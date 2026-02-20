"""LLM utilities for inference and synthetic data generation.

Provides wrappers for large language models with quantization support
and utilities for generating synthetic training data.
"""

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator

# Transformers imports (lazy loaded)
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _check_transformers():
    """Check if transformers is available."""
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for LLM utilities. "
            "Install with: pip install transformers torch"
        )


@dataclass
class GenerationResult:
    """Result from LLM generation.

    Attributes
    ----------
    text : str
        Generated text.
    prompt : str
        Input prompt.
    tokens_generated : int
        Number of tokens generated.
    finish_reason : str
        Reason for stopping: 'length', 'stop', 'eos'.
    """

    text: str
    prompt: str
    tokens_generated: int
    finish_reason: str


class LLMWrapper(BaseEstimator):
    """Wrapper for LLM inference with quantization support.

    Provides easy access to causal language models with optional
    4-bit or 8-bit quantization for efficient inference.

    Parameters
    ----------
    model_name : str
        Model name from Hugging Face Hub, e.g.:
        - 'mistralai/Mistral-7B-Instruct-v0.2'
        - 'meta-llama/Llama-2-7b-chat-hf'
        - 'google/gemma-7b-it'
        - 'microsoft/phi-2'
    quantization : str, default='none'
        Quantization mode: '4bit', '8bit', or 'none'.
    max_new_tokens : int, default=512
        Maximum tokens to generate.
    temperature : float, default=0.7
        Sampling temperature (0.0 for greedy).
    top_p : float, default=0.9
        Top-p (nucleus) sampling parameter.
    top_k : int, default=50
        Top-k sampling parameter.
    repetition_penalty : float, default=1.1
        Penalty for repeating tokens.
    do_sample : bool, default=True
        Whether to use sampling (False for greedy).
    device_map : str, default='auto'
        Device mapping strategy.
    torch_dtype : str, default='auto'
        Torch dtype: 'auto', 'float16', 'bfloat16', 'float32'.
    trust_remote_code : bool, default=False
        Whether to trust remote code.
    use_flash_attention : bool, default=True
        Whether to use flash attention if available.

    Attributes
    ----------
    model_ : transformers model
        Loaded language model.
    tokenizer_ : transformers tokenizer
        Associated tokenizer.
    generation_config_ : GenerationConfig
        Generation configuration.

    Examples
    --------
    >>> from endgame.nlp import LLMWrapper
    >>> # Load Mistral with 4-bit quantization
    >>> llm = LLMWrapper(
    ...     model_name='mistralai/Mistral-7B-Instruct-v0.2',
    ...     quantization='4bit',
    ...     max_new_tokens=256
    ... )
    >>> # Generate text
    >>> response = llm.generate("What is machine learning?")
    >>> print(response.text)
    >>> # Batch generation
    >>> responses = llm.generate_batch(["Question 1?", "Question 2?"])
    """

    def __init__(
        self,
        model_name: str,
        quantization: str = "none",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        use_flash_attention: bool = True,
    ):
        _check_transformers()

        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_flash_attention = use_flash_attention

        self.model_: Any | None = None
        self.tokenizer_: Any | None = None
        self.generation_config_: GenerationConfig | None = None

        self._load_model()

    def _get_torch_dtype(self):
        """Get torch dtype from string."""
        if self.torch_dtype == "auto":
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Get quantization configuration."""
        if self.quantization == "none":
            return None

        try:
            import bitsandbytes
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for quantization. "
                "Install with: pip install bitsandbytes"
            )

        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unknown quantization: {self.quantization}")

    def _load_model(self):
        """Load model and tokenizer."""
        # Load tokenizer
        self.tokenizer_ = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # Set padding token if not set
        if self.tokenizer_.pad_token is None:
            self.tokenizer_.pad_token = self.tokenizer_.eos_token

        # Model kwargs
        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        # Quantization
        quant_config = self._get_quantization_config()
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = self._get_torch_dtype()

        # Flash attention
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass  # Fall back to default attention

        # Load model
        self.model_ = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Set up generation config
        self.generation_config_ = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.do_sample else 1.0,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer_.pad_token_id,
            eos_token_id=self.tokenizer_.eos_token_id,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.
        system_prompt : str, optional
            System prompt to prepend.
        **kwargs
            Override generation parameters.

        Returns
        -------
        GenerationResult
            Generation result with text and metadata.
        """
        # Format prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Tokenize
        inputs = self.tokenizer_(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model_.device)

        input_length = inputs.input_ids.shape[1]

        # Create generation config with overrides
        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature) if self.do_sample else 1.0,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            do_sample=kwargs.get("do_sample", self.do_sample),
            pad_token_id=self.tokenizer_.pad_token_id,
            eos_token_id=self.tokenizer_.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            outputs = self.model_.generate(
                **inputs,
                generation_config=gen_config,
            )

        # Decode
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer_.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        # Determine finish reason
        if generated_tokens[-1] == self.tokenizer_.eos_token_id:
            finish_reason = "eos"
        elif len(generated_tokens) >= gen_config.max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return GenerationResult(
            text=generated_text.strip(),
            prompt=full_prompt,
            tokens_generated=len(generated_tokens),
            finish_reason=finish_reason,
        )

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        batch_size: int = 4,
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate text for multiple prompts.

        Parameters
        ----------
        prompts : List[str]
            Input prompts.
        system_prompt : str, optional
            System prompt to prepend to all.
        batch_size : int, default=4
            Batch size for generation.
        **kwargs
            Override generation parameters.

        Returns
        -------
        List[GenerationResult]
            Generation results.
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            for prompt in batch:
                result = self.generate(prompt, system_prompt, **kwargs)
                results.append(result)
        return results

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """Generate response in chat format.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Chat messages with 'role' and 'content' keys.
            Roles: 'system', 'user', 'assistant'.
        **kwargs
            Override generation parameters.

        Returns
        -------
        GenerationResult
            Generation result.
        """
        # Apply chat template if available
        if hasattr(self.tokenizer_, "apply_chat_template"):
            prompt = self.tokenizer_.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"

        return self.generate(prompt, **kwargs)


class ChainOfThoughtPrompt:
    """Generates Chain-of-Thought prompts for reasoning tasks.

    Essential for math, logic, and multi-step reasoning competitions.

    Examples
    --------
    >>> from endgame.nlp import ChainOfThoughtPrompt
    >>> cot = ChainOfThoughtPrompt()
    >>> prompt = cot.create_prompt(
    ...     question="What is 15% of 240?",
    ...     examples=[
    ...         ("What is 10% of 100?", "10% means 10/100 = 0.1. 0.1 × 100 = 10. Answer: 10")
    ...     ]
    ... )
    """

    # Common CoT triggers
    COT_TRIGGERS = [
        "Let's think step by step.",
        "Let's work through this carefully.",
        "Let me break this down step by step.",
        "I'll solve this systematically.",
    ]

    def __init__(
        self,
        cot_trigger: str = "Let's think step by step.",
        answer_prefix: str = "Therefore, the answer is",
    ):
        """Initialize ChainOfThoughtPrompt.

        Parameters
        ----------
        cot_trigger : str
            Phrase to trigger step-by-step reasoning.
        answer_prefix : str
            Prefix for final answer extraction.
        """
        self.cot_trigger = cot_trigger
        self.answer_prefix = answer_prefix

    def create_prompt(
        self,
        question: str,
        examples: list[tuple] | None = None,
        include_trigger: bool = True,
    ) -> str:
        """Create a CoT prompt.

        Parameters
        ----------
        question : str
            The question to answer.
        examples : List[tuple], optional
            Few-shot examples as (question, reasoning) tuples.
        include_trigger : bool, default=True
            Whether to include the CoT trigger.

        Returns
        -------
        str
            Formatted CoT prompt.
        """
        prompt_parts = []

        # Add examples
        if examples:
            for i, (ex_q, ex_reasoning) in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Q: {ex_q}")
                prompt_parts.append(f"A: {ex_reasoning}")
                prompt_parts.append("")

        # Add question
        prompt_parts.append("Question:")
        prompt_parts.append(f"Q: {question}")

        if include_trigger:
            prompt_parts.append(f"A: {self.cot_trigger}")
        else:
            prompt_parts.append("A:")

        return "\n".join(prompt_parts)

    def create_self_consistency_prompts(
        self,
        question: str,
        n_samples: int = 5,
        examples: list[tuple] | None = None,
    ) -> list[str]:
        """Create multiple prompts for self-consistency.

        Self-consistency involves sampling multiple reasoning paths
        and taking the majority vote answer.

        Parameters
        ----------
        question : str
            The question.
        n_samples : int, default=5
            Number of prompt variations.
        examples : List[tuple], optional
            Few-shot examples.

        Returns
        -------
        List[str]
            Multiple prompts with different triggers.
        """
        prompts = []
        triggers = self.COT_TRIGGERS[:n_samples]

        # Pad with default if needed
        while len(triggers) < n_samples:
            triggers.append(self.cot_trigger)

        for trigger in triggers:
            original_trigger = self.cot_trigger
            self.cot_trigger = trigger
            prompts.append(self.create_prompt(question, examples))
            self.cot_trigger = original_trigger

        return prompts

    def extract_answer(
        self,
        response: str,
        answer_format: str = "number",
    ) -> str | None:
        """Extract final answer from CoT response.

        Parameters
        ----------
        response : str
            Full response with reasoning.
        answer_format : str, default='number'
            Expected format: 'number', 'letter', 'text'.

        Returns
        -------
        str or None
            Extracted answer.
        """
        import re

        # Look for answer after prefix
        if self.answer_prefix in response:
            answer_part = response.split(self.answer_prefix)[-1].strip()
        else:
            # Try to find last sentence
            sentences = response.split(".")
            answer_part = sentences[-1].strip() if sentences else response

        if answer_format == "number":
            # Extract numbers
            numbers = re.findall(r"-?\d+\.?\d*", answer_part)
            return numbers[0] if numbers else None

        elif answer_format == "letter":
            # Extract single letter answer (A, B, C, D)
            letters = re.findall(r"\b([A-D])\b", answer_part.upper())
            return letters[0] if letters else None

        else:
            # Return cleaned text
            return answer_part.split("\n")[0].strip()


class SyntheticDataGenerator:
    """Generate synthetic training data using LLMs.

    Key technique in 2024 competitions: distill GPT-4/Claude
    reasoning into smaller models.

    Parameters
    ----------
    llm : LLMWrapper
        LLM to use for generation.
    diversity_factor : float, default=0.7
        Temperature for generation (higher = more diverse).

    Examples
    --------
    >>> from endgame.nlp import LLMWrapper, SyntheticDataGenerator
    >>> llm = LLMWrapper('mistralai/Mistral-7B-Instruct-v0.2', quantization='4bit')
    >>> generator = SyntheticDataGenerator(llm)
    >>> samples = generator.generate_classification_data(
    ...     task_description="Classify text as positive or negative sentiment",
    ...     labels=["positive", "negative"],
    ...     examples_per_label=100
    ... )
    """

    def __init__(
        self,
        llm: LLMWrapper,
        diversity_factor: float = 0.7,
    ):
        self.llm = llm
        self.diversity_factor = diversity_factor

    def generate_classification_data(
        self,
        task_description: str,
        labels: list[str],
        examples_per_label: int = 50,
        seed_examples: dict[str, list[str]] | None = None,
    ) -> list[dict[str, str]]:
        """Generate synthetic classification data.

        Parameters
        ----------
        task_description : str
            Description of the classification task.
        labels : List[str]
            Class labels.
        examples_per_label : int, default=50
            Number of examples per label.
        seed_examples : Dict[str, List[str]], optional
            Seed examples for each label to guide generation.

        Returns
        -------
        List[Dict[str, str]]
            Generated samples with 'text' and 'label' keys.
        """
        samples = []

        for label in labels:
            prompt = self._build_generation_prompt(
                task_description, label, seed_examples, examples_per_label
            )

            result = self.llm.generate(
                prompt,
                temperature=self.diversity_factor,
                max_new_tokens=1024,
            )

            # Parse generated examples
            generated = self._parse_examples(result.text)

            for text in generated[:examples_per_label]:
                samples.append({"text": text, "label": label})

        return samples

    def _build_generation_prompt(
        self,
        task_description: str,
        label: str,
        seed_examples: dict[str, list[str]] | None,
        n_examples: int,
    ) -> str:
        """Build prompt for generating examples."""
        prompt = f"""Task: {task_description}

Generate {n_examples} diverse examples of text that would be classified as "{label}".

Requirements:
- Each example should be unique and realistic
- Examples should vary in length and style
- Cover different aspects of the "{label}" category

"""

        if seed_examples and label in seed_examples:
            prompt += "Here are some example texts for this category:\n"
            for ex in seed_examples[label][:3]:
                prompt += f"- {ex}\n"
            prompt += "\n"

        prompt += f"Generate {n_examples} new examples, one per line:\n"

        return prompt

    def _parse_examples(self, text: str) -> list[str]:
        """Parse generated examples from text."""
        lines = text.strip().split("\n")
        examples = []

        for line in lines:
            line = line.strip()
            # Remove common prefixes
            for prefix in ["- ", "* ", "• ", "> "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]

            # Remove numbering
            import re
            line = re.sub(r"^\d+[\.\)]\s*", "", line)

            if line and len(line) > 10:  # Skip very short lines
                examples.append(line)

        return examples

    def generate_qa_pairs(
        self,
        context: str,
        n_pairs: int = 10,
        question_types: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Generate question-answer pairs from context.

        Parameters
        ----------
        context : str
            Source text to generate Q&A from.
        n_pairs : int, default=10
            Number of Q&A pairs to generate.
        question_types : List[str], optional
            Types of questions: 'factual', 'inferential', 'opinion'.

        Returns
        -------
        List[Dict[str, str]]
            Generated Q&A pairs with 'question', 'answer', 'context' keys.
        """
        if question_types is None:
            question_types = ["factual", "inferential"]

        prompt = f"""Based on the following context, generate {n_pairs} question-answer pairs.

Context:
{context}

Requirements:
- Questions should be answerable from the context
- Include different question types: {', '.join(question_types)}
- Answers should be concise but complete

Format each Q&A pair as:
Q: [question]
A: [answer]

Generate {n_pairs} Q&A pairs:
"""

        result = self.llm.generate(
            prompt,
            temperature=self.diversity_factor,
            max_new_tokens=1024,
        )

        # Parse Q&A pairs
        pairs = []
        lines = result.text.strip().split("\n")

        current_q = None
        for line in lines:
            line = line.strip()
            if line.startswith("Q:"):
                current_q = line[2:].strip()
            elif line.startswith("A:") and current_q:
                answer = line[2:].strip()
                pairs.append({
                    "question": current_q,
                    "answer": answer,
                    "context": context,
                })
                current_q = None

        return pairs
