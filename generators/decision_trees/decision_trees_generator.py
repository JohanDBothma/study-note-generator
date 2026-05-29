#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Decision Trees — Study Sheet
=============================
Covers tree structure, training, entropy/information gain, Gini impurity,
and the full sklearn workflow from raw data to a visualised tree.

Usage (run from project root):
    python -m generators.decision_trees.decision_trees_generator
    python -m generators.decision_trees.decision_trees_generator --output pdfs/decision_trees/decision_trees.pdf
"""

import argparse
import base64
from shared.cards import build_code_card
from shared.renderer import render


def _img(path):
    """Load a PNG/JPG from path and return a base64 string, or None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


ENTROPY_TABLE = """\
| Entropy | Meaning | Example split |
|---------|---------|---------------|
| 0 | Pure — all samples same class | 10/10 drugA |
| ~0.5 | Low — mostly one class | 8/10 drugA, 2/10 drugB |
| ~1.0 | High — classes roughly equal | 5/10 drugA, 5/10 drugB |
| Max | Completely mixed | Equal spread across all classes |
"""

ITEMS = [
    {
        "title": "Tree Structure",
        "type": "Rule-based tree",
        "image_b64": _img("generators/decision_trees/assets/tree_structure.png"),
        "image_caption": "Example decision tree — root node, decision nodes, and leaf nodes.",
        "image_width": "55%",
        "description": (
            "A decision tree is a flowchart-like graph that maps feature conditions to class predictions. "
            "It has three kinds of nodes:\n\n"
            "• Root node — the very first split; uses the most informative feature in the whole dataset.\n"
            "• Decision nodes — internal nodes that test one feature and branch left or right based on the result.\n"
            "• Leaf nodes — terminal nodes that hold the final predicted class (no more splits below them).\n\n"
            "Each path from root to leaf is a human-readable rule, e.g. "
            '"If Na_to_K > 14.8 → drugY".'
        ),
        "code": """\
# Print a text representation of the tree — shows root, decision, and leaf nodes
from sklearn.tree import export_text

tree_rules = export_text(
    drugTree,
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
)
print(tree_rules)

# Example output:
# |--- Na_to_K <= 14.82
# |   |--- BP <= 0.50
# |   |   |--- class: drugX      <- leaf node
# |   |--- BP > 0.50
# |   |   |--- class: drugA      <- leaf node
# |--- Na_to_K > 14.82
# |   |--- class: drugY          <- leaf node
""",
    },
    {
        "title": "Training a Decision Tree",
        "type": "Rule-based tree",
        "description": (
            "Training builds the tree top-down by greedily picking the best split at each node:\n\n"
            "1. Start with the root node containing ALL training samples.\n"
            "2. Search every feature for the threshold that best separates the classes "
            "(measured by information gain or Gini impurity).\n"
            "3. Split the node's data into two child nodes on that threshold.\n"
            "4. Repeat recursively for each child until a stopping criterion is met "
            "(e.g. max_depth, pure leaf, or too few samples).\n\n"
            "The result is a tree where each internal node asks exactly one yes/no question "
            "about one feature."
        ),
        "code": """\
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.3, random_state=3
)

# criterion="entropy" uses information gain to choose splits
# max_depth limits tree size to avoid overfitting
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)
""",
    },
    {
        "title": "Entropy & Information Gain",
        "type": "Rule-based tree",
        "description": (
            "Entropy measures the impurity (disorder) of a node's class distribution:\n\n"
            "    H = -Σ p_i · log₂(p_i)\n\n"
            "Information gain is the drop in entropy after a split — "
            "the algorithm picks whichever feature maximises this drop:\n\n"
            "    IG = H(parent) − weighted_avg(H(children))\n\n"
            "Higher information gain → the split tells us more → better feature to split on."
        ),
        "dataset_md": ENTROPY_TABLE,
        "code": """\
# sklearn computes entropy internally; you can inspect it via the tree object
import numpy as np

def entropy(labels):
    classes, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

print("Root entropy:", entropy(y_trainset))
# After fitting, each node's impurity is stored in:
print(drugTree.tree_.impurity)   # array of entropy values per node
""",
    },
    {
        "title": "Gini Impurity",
        "type": "Rule-based tree",
        "description": (
            "Gini impurity is an alternative splitting criterion to entropy. "
            "It measures the probability of mislabelling a randomly chosen sample:\n\n"
            "    Gini = 1 − Σ p_i²\n\n"
            "• Gini = 0 → perfectly pure node (all one class).\n"
            "• Gini = 0.5 → maximally impure for a binary problem.\n\n"
            "Gini is slightly faster to compute (no logarithm) and tends to produce "
            "similar trees to entropy in practice. Use criterion='gini' in sklearn."
        ),
        "code": """\
# Switch between criteria — results are usually very close
tree_gini    = DecisionTreeClassifier(criterion="gini",    max_depth=4)
tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4)

tree_gini.fit(X_trainset, y_trainset)
tree_entropy.fit(X_trainset, y_trainset)

from sklearn import metrics
for name, model in [("Gini", tree_gini), ("Entropy", tree_entropy)]:
    preds = model.predict(X_testset)
    print(f"{name} accuracy: {metrics.accuracy_score(y_testset, preds):.3f}")
""",
    },
    {
        "title": "Encoding Categorical Features",
        "type": "Categorical encoding",
        "description": (
            "Decision trees in sklearn only accept numeric inputs. "
            "Categorical columns (e.g. Sex = 'M'/'F', BP = 'LOW'/'NORMAL'/'HIGH') "
            "must be converted to integers before fitting.\n\n"
            "LabelEncoder assigns each unique string a stable integer (alphabetical order). "
            "It's fine for ordinal or small-cardinality columns used in a tree — "
            "the tree only checks thresholds, so the exact integer values don't need to be meaningful."
        ),
        "code": """\
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Encode each categorical column in-place
my_data['Sex']         = label_encoder.fit_transform(my_data['Sex'])
my_data['BP']          = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

# Inspect the dataset after encoding
my_data.info()
print(my_data.head())
""",
    },
    {
        "title": "Visualising the Tree",
        "type": "Rule-based tree",
        "image_b64": _img("generators/decision_trees/assets/tree_plot.png"),
        "image_caption": "Output of plot_tree() — coloured nodes show majority class at each split.",
        "description": (
            "plot_tree() renders the fitted tree as a graphical diagram. "
            "Each node shows the split condition, sample count, impurity, and majority class. "
            "Setting filled=True colours nodes by their majority class — making leaf nodes "
            "instantly readable at a glance.\n\n"
            "Tips: use figsize to give the canvas enough room for deep trees, "
            "and feature_names / class_names to replace the default x[0], x[1] placeholders "
            "with your actual column and class names."
        ),
        "code": """\
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    drugTree,
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
    class_names=drugTree.classes_,
    filled=True,      # colour nodes by majority class
    rounded=True,
    fontsize=12,
    proportion=False,
)
plt.tight_layout()
plt.show()
""",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate Decision Trees study sheet PDF")
    parser.add_argument("--output", default="pdfs/decision_trees/decision_trees.pdf")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    render(
        ITEMS,
        title="Decision Trees",
        subtitle="Tree structure, training, entropy, Gini impurity, and the full sklearn workflow.",
        output_path=args.output,
        save_html=args.html,
        card_builder=build_code_card,
        show_header=True,
    )


if __name__ == "__main__":
    main()
