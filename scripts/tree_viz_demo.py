"""Train a decision tree on an OpenML dataset and save an interactive visualization."""

from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from endgame.visualization import TreeVisualizer

# Load the 'credit-g' dataset (German credit, well-labeled, 1000 samples, 20 features)
data = fetch_openml(data_id=31, as_frame=True, parser="auto")
X, y = data.data, data.target
feature_names = list(X.columns)
class_names = sorted(y.unique().tolist())

# One-hot encode categoricals so the tree can split on them
X = X.apply(lambda col: col.cat.codes if col.dtype.name == "category" else col)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
clf.fit(X_train, y_train)

acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)
print(f"Train accuracy: {acc_train:.3f}")
print(f"Test accuracy:  {acc_test:.3f}")

viz = TreeVisualizer(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    title="German Credit — Decision Tree (max_depth=5)",
    color_by="prediction",
)

out = viz.save("tree_viz_demo.html")
print(f"Saved to {out}")
