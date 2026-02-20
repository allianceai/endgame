"""Neural embedding layers for Bayesian Network Classifiers.

This module provides neural network components for learning
conditional probability distributions from data.

The key innovation is replacing exponentially-sized CPTs with
neural conditional estimators that can generalize to unseen
parent configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalEmbeddingNet(nn.Module):
    """
    Neural network that estimates P(x_i | parents(x_i), y).

    Architecture:
    1. Embed each parent value and class to dense vectors
    2. Concatenate embeddings
    3. MLP outputs logits over x_i's possible values

    Parameters
    ----------
    target_cardinality : int
        Number of possible values for the target variable.

    parent_cardinalities : list[int]
        Number of possible values for each parent variable.
        The last entry should be the class cardinality if class is a parent.

    embedding_dim : int, default=16
        Dimensionality of value embeddings.

    hidden_dim : int, default=64
        Hidden layer size.

    n_hidden_layers : int, default=2
        Number of hidden layers.

    dropout : float, default=0.1
        Dropout rate.

    Examples
    --------
    >>> net = ConditionalEmbeddingNet(
    ...     target_cardinality=5,
    ...     parent_cardinalities=[3, 4, 2],  # 2 features + class
    ...     embedding_dim=16,
    ...     hidden_dim=64
    ... )
    >>> parent_values = torch.tensor([[0, 1, 0], [2, 3, 1]])
    >>> logits = net(parent_values)  # Shape: (2, 5)
    """

    def __init__(
        self,
        target_cardinality: int,
        parent_cardinalities: list[int],
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.target_cardinality = target_cardinality
        self.parent_cardinalities = parent_cardinalities
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding for each parent (including class)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim)
            for card in parent_cardinalities
        ])

        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.1)

        total_embed_dim = embedding_dim * len(parent_cardinalities)

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(total_embed_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, target_cardinality))

        self.net = nn.Sequential(*layers)

    def forward(self, parent_values: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for target values given parent values.

        Parameters
        ----------
        parent_values : torch.Tensor
            Integer tensor of shape (batch_size, n_parents).
            Each column corresponds to a parent variable.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, target_cardinality).
        """
        # Get embeddings for each parent
        embeds = []
        for i in range(parent_values.shape[1]):
            # Clamp to valid range to handle unseen values
            vals = parent_values[:, i].clamp(0, self.parent_cardinalities[i] - 1)
            embeds.append(self.embeddings[i](vals))

        # Concatenate embeddings
        x = torch.cat(embeds, dim=-1)

        # Pass through MLP
        return self.net(x)

    def get_proba(self, parent_values: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities (softmax of logits).

        Parameters
        ----------
        parent_values : torch.Tensor
            Integer tensor of shape (batch_size, n_parents).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (batch_size, target_cardinality).
        """
        logits = self.forward(parent_values)
        return F.softmax(logits, dim=-1)

    def get_log_proba(self, parent_values: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probabilities (log-softmax of logits).

        Parameters
        ----------
        parent_values : torch.Tensor
            Integer tensor of shape (batch_size, n_parents).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size, target_cardinality).
        """
        logits = self.forward(parent_values)
        return F.log_softmax(logits, dim=-1)


class SharedEmbeddingLayer(nn.Module):
    """
    Shared embedding layer for multiple categorical features.

    Uses a single embedding matrix for all features with
    feature-specific offsets. This is more parameter-efficient
    than separate embeddings when features share similar semantics.

    Parameters
    ----------
    total_categories : int
        Total number of categories across all features.

    embedding_dim : int
        Dimensionality of embeddings.

    feature_offsets : list[int]
        Starting index for each feature in the shared embedding.
    """

    def __init__(
        self,
        total_categories: int,
        embedding_dim: int,
        feature_offsets: list[int],
    ):
        super().__init__()

        self.embedding = nn.Embedding(total_categories, embedding_dim)
        self.feature_offsets = feature_offsets

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(
        self,
        values: torch.Tensor,
        feature_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get embeddings for feature values.

        Parameters
        ----------
        values : torch.Tensor
            Feature values of shape (batch_size, n_features) or (batch_size,).
        feature_indices : torch.Tensor, optional
            Feature indices if values is 1D.

        Returns
        -------
        torch.Tensor
            Embeddings.
        """
        if values.dim() == 1:
            # Single feature
            if feature_indices is None:
                raise ValueError("feature_indices required for 1D input")
            offset = self.feature_offsets[feature_indices[0].item()]
            return self.embedding(values + offset)

        # Multiple features
        batch_size, n_features = values.shape
        embeddings = []

        for i in range(n_features):
            offset = self.feature_offsets[i]
            embeddings.append(self.embedding(values[:, i] + offset))

        return torch.stack(embeddings, dim=1)


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregation of parent embeddings.

    Instead of simple concatenation, uses attention to weight
    the contribution of each parent embedding based on context.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of input embeddings.

    n_heads : int, default=4
        Number of attention heads.

    dropout : float, default=0.1
        Dropout rate.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate parent embeddings using attention.

        Parameters
        ----------
        embeddings : torch.Tensor
            Shape (batch_size, n_parents, embedding_dim).

        Returns
        -------
        torch.Tensor
            Aggregated embedding of shape (batch_size, embedding_dim).
        """
        # Self-attention
        attended, _ = self.attention(embeddings, embeddings, embeddings)

        # Residual connection
        attended = self.norm(embeddings + attended)

        # Pool across parents (mean)
        return attended.mean(dim=1)


class ConditionalAttentionNet(nn.Module):
    """
    Neural conditional estimator with attention aggregation.

    More expressive than ConditionalEmbeddingNet for complex
    parent dependencies, but more expensive.

    Parameters
    ----------
    target_cardinality : int
        Number of possible target values.

    parent_cardinalities : list[int]
        Cardinalities of parent variables.

    embedding_dim : int, default=32
        Embedding dimensionality.

    n_heads : int, default=4
        Number of attention heads.

    hidden_dim : int, default=128
        Hidden layer size.

    dropout : float, default=0.1
        Dropout rate.
    """

    def __init__(
        self,
        target_cardinality: int,
        parent_cardinalities: list[int],
        embedding_dim: int = 32,
        n_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.target_cardinality = target_cardinality
        self.parent_cardinalities = parent_cardinalities
        self.embedding_dim = embedding_dim

        # Parent embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim)
            for card in parent_cardinalities
        ])

        # Attention aggregation
        self.attention_aggregator = AttentionAggregator(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_cardinality),
        )

    def forward(self, parent_values: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for target values.

        Parameters
        ----------
        parent_values : torch.Tensor
            Shape (batch_size, n_parents).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, target_cardinality).
        """
        batch_size = parent_values.shape[0]

        # Get embeddings
        embeds = []
        for i in range(parent_values.shape[1]):
            vals = parent_values[:, i].clamp(0, self.parent_cardinalities[i] - 1)
            embeds.append(self.embeddings[i](vals))

        # Stack: (batch_size, n_parents, embedding_dim)
        embeds = torch.stack(embeds, dim=1)

        # Attention aggregation
        aggregated = self.attention_aggregator(embeds)

        # Output
        return self.mlp(aggregated)
