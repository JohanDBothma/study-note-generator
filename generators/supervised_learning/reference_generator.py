#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Supervised Learning — Reference Sheet
======================================
Usage (from project root):
    python -m generators.supervised_learning.reference_generator
    python -m generators.supervised_learning.reference_generator --output pdfs/supervised_learning/reference.pdf --html
"""

import argparse
from shared.cards import build_reference_card
from shared.renderer import render
from shared.styles import BADGE_COLORS

# Register badge colours used in this sheet
BADGE_COLORS.update({
    "Multiclass strategy": ("#e0f2fe", "#0c4a6e"),
    "Evaluation metric":   ("#fef3c7", "#92400e"),
    "Visualisation":       ("#ede9fe", "#5b21b6"),
    "Class imbalance":     ("#fff1f2", "#9f1239"),
})

ITEMS = [
    # ── Models ────────────────────────────────────────────────────────────────
    {
        "title": "One vs One Classifier",
        "type": "Multiclass strategy",
        "process": "Trains one classifier for each pair of classes. Each classifier votes, and the class with the most votes wins.",
        "hyperparams": [
            ("estimator", "The base classifier to use, e.g. LogisticRegression()"),
        ],
        "pros": "Works well for small datasets.",
        "cons": "Gets slow quickly — needs k² classifiers for k classes.",
        "applications": "Multiclass problems with few classes.",
        "code": """\
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression

model = OneVsOneClassifier(LogisticRegression())
model.fit(X_train, y_train)""",
    },
    {
        "title": "One vs All Classifier",
        "type": "Multiclass strategy",
        "process": "Trains one classifier per class (this class vs. everything else). The class with the highest score wins.",
        "hyperparams": [
            ("estimator", "The base classifier to use, e.g. LogisticRegression()"),
        ],
        "pros": "Simpler and faster than One vs One.",
        "cons": "Struggles when classes are very unbalanced.",
        "applications": "Image and text classification.",
        "code": """\
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)""",
    },
    {
        "title": "Decision Tree Classifier",
        "type": "Rule-based tree",
        "process": "Splits data into groups by asking yes/no questions about features. Each split tries to keep the resulting groups as pure as possible.",
        "hyperparams": [
            ("max_depth",        "How many levels deep the tree can go"),
            ("min_samples_leaf", "Minimum samples needed to form a leaf"),
        ],
        "pros": "Easy to read and explain; no need to scale features.",
        "cons": "Overfits easily — always set max_depth.",
        "applications": "Credit scoring, medical diagnosis.",
        "code": """\
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)""",
    },
    {
        "title": "Decision Tree Regressor",
        "type": "Rule-based tree",
        "process": "Same as the classifier, but instead of a class label each leaf returns the average value of its training samples.",
        "hyperparams": [
            ("max_depth",        "How many levels deep the tree can go"),
            ("min_samples_leaf", "Minimum samples needed to form a leaf"),
        ],
        "pros": "Easy to interpret; handles nonlinear data.",
        "cons": "Overfits easily; sensitive to noisy data.",
        "applications": "House price prediction, demand forecasting.",
        "code": """\
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)""",
    },
    {
        "title": "Linear SVM Classifier",
        "type": "Maximum margin classifier",
        "process": "Draws a line (or plane) between classes with the widest possible gap on either side. Kernels let it handle curved boundaries.",
        "hyperparams": [
            ("C",      "How strict the boundary is — lower = more tolerant of errors"),
            ("kernel", "Shape of the boundary: 'linear', 'rbf', 'poly'"),
            ("gamma",  "How far each point's influence reaches (rbf/poly only)"),
        ],
        "pros": "Works well with lots of features.",
        "cons": "Slow on large datasets; features need to be scaled.",
        "applications": "Text and image classification.",
        "code": """\
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)""",
    },
    {
        "title": "K-Nearest Neighbors Classifier",
        "type": "Instance-based learner",
        "process": "Finds the k most similar training points and returns their majority class. No training needed — it just memorises the data.",
        "hyperparams": [
            ("n_neighbors", "How many neighbours to look at"),
            ("weights",     "'uniform' = equal vote, 'distance' = closer = more weight"),
            ("algorithm",   "How to search for neighbours: 'auto', 'kd_tree', 'brute'"),
        ],
        "pros": "Simple and needs no training step.",
        "cons": "Slow to predict on large data; scale your features first.",
        "applications": "Recommendations, image recognition.",
        "code": """\
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train, y_train)""",
    },
    {
        "title": "Random Forest Regressor",
        "type": "Ensemble — parallel trees",
        "process": "Trains many decision trees on random subsets of the data, then averages their predictions. The randomness makes it much harder to overfit.",
        "hyperparams": [
            ("n_estimators", "Number of trees — more = better, but slower"),
            ("max_depth",    "Max depth of each tree"),
        ],
        "pros": "Much less likely to overfit than a single tree.",
        "cons": "Uses more memory and takes longer to train as trees increase.",
        "applications": "Sales and price forecasting.",
        "code": """\
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)""",
    },
    {
        "title": "XGBoost Regressor",
        "type": "Ensemble — sequential trees",
        "process": "Builds trees one by one, where each new tree fixes the mistakes of the previous ones. Often the most accurate option for structured data.",
        "hyperparams": [
            ("n_estimators",  "Number of trees to build"),
            ("learning_rate", "How much each tree contributes — smaller = more careful"),
            ("max_depth",     "Max depth of each tree"),
        ],
        "pros": "Very accurate; handles missing values automatically.",
        "cons": "Slow to train; many settings to tune.",
        "applications": "Competition ML (Kaggle), any structured data problem.",
        "code": """\
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)""",
    },

    # ── Associated functions ───────────────────────────────────────────────────
    {
        "title": "OneHotEncoder",
        "type": "Categorical encoding",
        "process": "Turns a categorical column into one binary column per category. E.g. 'Red/Green/Blue' becomes three 0/1 columns.",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Encoding categorical input features before model training.",
        "code": """\
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(categorical_data)""",
    },
    {
        "title": "accuracy_score",
        "type": "Evaluation metric",
        "process": "What percentage of predictions were correct. Simple, but misleading when one class is much more common than others.",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Quick check on balanced classification results.",
        "code": """\
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)""",
    },
    {
        "title": "LabelEncoder",
        "type": "Categorical encoding",
        "process": "Converts string class labels to numbers (alphabetical order). Only use on the target column, not on input features.",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Encoding the target (y) column before training.",
        "code": """\
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)""",
    },
    {
        "title": "plot_tree",
        "type": "Visualisation",
        "process": "Draws a decision tree as a diagram so you can see exactly which questions it asks at each step.",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Checking and explaining a decision tree model.",
        "code": """\
from sklearn.tree import plot_tree

plot_tree(model, max_depth=3, filled=True)""",
    },
    {
        "title": "normalize",
        "type": "Numerical scaling",
        "process": "Scales each row to length 1. Different from StandardScaler which works column by column — use this when the direction of a row matters more than its size.",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Text vectors, any model that uses cosine similarity.",
        "code": """\
from sklearn.preprocessing import normalize

normalized_data = normalize(data, norm='l2')""",
    },
    {
        "title": "compute_sample_weight",
        "type": "Class imbalance",
        "process": "Gives higher weight to samples from rare classes so the model doesn't ignore them. Pass the result into model.fit().",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Training on datasets where some classes have far fewer examples.",
        "code": """\
from sklearn.utils.class_weight import compute_sample_weight

weights = compute_sample_weight(class_weight='balanced', y=y)
model.fit(X_train, y_train, sample_weight=weights)""",
    },
    {
        "title": "roc_auc_score",
        "type": "Evaluation metric",
        "process": "Measures how well a classifier separates the two classes, regardless of where you set the decision threshold. 1.0 = perfect, 0.5 = random.",
        "hyperparams": [],
        "pros": None,
        "cons": None,
        "applications": "Comparing classifiers, especially when classes are unbalanced.",
        "code": """\
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_score)""",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate supervised learning reference PDF")
    parser.add_argument("--output", default="pdfs/supervised_learning/supervised_learning_reference.pdf")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    render(
        ITEMS,
        title="Supervised Learning — Quick Reference",
        subtitle="Models, multiclass strategies, and utility functions",
        output_path=args.output,
        save_html=args.html,
        card_builder=build_reference_card,
        show_header=True,
    )


if __name__ == "__main__":
    main()
