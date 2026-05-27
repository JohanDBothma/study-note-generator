#!/usr/bin/env python3
"""
ML Study Sheet Generator
========================
Generates a clean PDF study sheet for ML classification algorithms.

Usage:
    python ml_study_sheet.py
    python ml_study_sheet.py --output my_notes.pdf

Requirements:
    pip install weasyprint matplotlib
"""

import argparse
import base64
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#378add"
C_AMBER  = "#ef9f27"
C_GREEN  = "#34d399"
C_RED    = "#b41e1e"
C_GRAY   = "#888888"
C_QUERY  = "#c03030"

BADGE_COLORS = {
    "Probabilistic":  ("#fce7f3", "#9d174d"),
    "Linear":         ("#dbeafe", "#1e40af"),
    "Tree-based":     ("#fef3c7", "#92400e"),
    "Instance-based": ("#ede9fe", "#5b21b6"),
    "Margin-based":   ("#e0f2fe", "#0c4a6e"),
    "Deep learning":  ("#fff1f2", "#9f1239"),
    "Ensemble":       ("#dcfce7", "#166534"),
    "Statistical":    ("#f0fdf4", "#14532d"),
}

# ── chart helpers ──────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def rng_seq(seed):
    """Tiny deterministic pseudo-random generator (no numpy dependency for seeding)."""
    x = float(seed)
    def _next():
        nonlocal x
        x = (x * 9301 + 49297) % 233280
        return x / 233280
    return _next


def chart_nb():
    words  = ["free", "win", "money", "meeting", "click", "prize"]
    spam   = [0.92, 0.88, 0.79, 0.05, 0.81, 0.95]
    ham    = [0.08, 0.12, 0.21, 0.95, 0.19, 0.05]
    x      = np.arange(len(words))
    w      = 0.35
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    ax.bar(x - w/2, spam, w, color=C_AMBER, alpha=0.85, label="Spam prob")
    ax.bar(x + w/2, ham,  w, color=C_BLUE,  alpha=0.75, label="Ham prob")
    ax.set_xticks(x); ax.set_xticklabels(words, fontsize=9)
    ax.set_ylim(0, 1.05); ax.tick_params(labelsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def chart_lr():
    r0, r1 = rng_seq(1), rng_seq(2)
    pts0, pts1 = [], []
    for _ in range(50):
        xi, yi = r0()*3-2.5, r0()*3-2
        if xi - yi < 0.3: pts0.append((xi, yi))
    for _ in range(50):
        xi, yi = r1()*3-0.5, r1()*3-2.5
        if xi - yi > -0.3: pts1.append((xi, yi))
    pts0 = pts0[:30]; pts1 = pts1[:30]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    if pts0: ax.scatter(*zip(*pts0), color=C_BLUE,  alpha=0.8, s=22, label="Class 0")
    if pts1: ax.scatter(*zip(*pts1), color=C_AMBER, alpha=0.8, s=22, label="Class 1")
    lx = np.array([-3, 3])
    ax.plot(lx, lx - 0.6, color=C_RED, lw=1.8, ls="--", label="Boundary")
    ax.set_xlim(-3, 3); ax.set_ylim(-3.2, 3)
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def chart_dt():
    """Decision tree rendered as a simple diagram using matplotlib patches."""
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, sub, fc, ec, tc, sc):
        rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x, y+0.18, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=tc)
        ax.text(x, y-0.28, sub, ha="center", va="center",
                fontsize=7, color=sc)

    def leaf(x, y, label, fc, ec, tc):
        rect = mpatches.FancyBboxPatch((x-1.1, y-0.45), 2.2, 0.9,
            boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=tc)

    def arrow(x1, y1, x2, y2, label, lc):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=0.8))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx-0.15, my, label, ha="right", va="center",
                fontsize=7, color=lc)

    # root
    box(5, 4.2, 2.8, 0.9, "petal length", "≤ 2.45 ?",
        "#dbeafe", "#93c5fd", "#1e40af", "#3b82f6")
    # left branch → setosa
    arrow(5, 3.75, 2.5, 2.85, "yes", "#16a34a")
    leaf(2.5, 2.5, "Setosa", "#dcfce7", "#86efac", "#166534")
    # right branch → inner node
    arrow(5, 3.75, 7.5, 2.85, "no", "#dc2626")
    box(7.5, 2.5, 2.8, 0.9, "petal width", "≤ 1.75 ?",
        "#dbeafe", "#93c5fd", "#1e40af", "#3b82f6")
    # inner → versicolor
    arrow(7.5, 2.05, 6, 1.15, "yes", "#16a34a")
    leaf(6, 0.8, "Versicolor", "#ede9fe", "#c4b5fd", "#5b21b6")
    # inner → virginica
    arrow(7.5, 2.05, 9, 1.15, "no", "#dc2626")
    leaf(9, 0.8, "Virginica", "#fef3c7", "#fcd34d", "#92400e")

    fig.tight_layout(pad=0.2)
    return fig_to_b64(fig)


def chart_knn():
    r0, r1 = rng_seq(3), rng_seq(5)
    c0 = [(r0()*1.5-2,   r0()*1.5-0.5) for _ in range(8)]
    c1 = [(r1()*1.5+0.2, r1()*1.5-1.5) for _ in range(8)]
    qx, qy = -0.4, -0.3

    # find 3 nearest from c0
    dists = sorted(c0, key=lambda p: (p[0]-qx)**2+(p[1]-qy)**2)
    k3 = dists[:3]
    radius = max((p[0]-qx)**2+(p[1]-qy)**2 for p in k3)**0.5 + 0.15

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    circle = plt.Circle((qx, qy), radius, fill=False,
                         linestyle="--", color="#bbbbbb", lw=1)
    ax.add_patch(circle)
    for p in k3:
        ax.plot([qx, p[0]], [qy, p[1]], color="#aaaaaa",
                lw=1, ls="--", zorder=1)
    x0, y0 = zip(*c0); x1, y1 = zip(*c1)
    ax.scatter(x0, y0, color=C_BLUE,  alpha=0.85, s=30, zorder=2, label="Class 0")
    ax.scatter(x1, y1, color=C_AMBER, alpha=0.85, s=30, zorder=2, label="Class 1")
    ax.scatter([qx], [qy], color=C_QUERY, s=60, zorder=3,
               edgecolors="white", lw=1.2, label="Query → class 0")
    ax.set_xlim(-2.5, 2); ax.set_ylim(-2, 1.5)
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def chart_svm():
    r0, r1 = rng_seq(7), rng_seq(9)
    c0 = [(r0()*1.8-3,   r0()*2-1)   for _ in range(10)]
    c1 = [(r1()*1.8+0.2, r1()*2-1.5) for _ in range(10)]
    bx = -0.5   # decision boundary x
    margin = 0.7

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    ax.axvline(bx, color=C_RED, lw=1.8, zorder=2)
    ax.axvline(bx - margin, color=C_BLUE,  lw=1, ls="--", alpha=0.5, zorder=1)
    ax.axvline(bx + margin, color=C_AMBER, lw=1, ls="--", alpha=0.5, zorder=1)
    ax.axvspan(bx-margin, bx+margin, alpha=0.04, color="#888888")
    x0, y0 = zip(*c0); x1, y1 = zip(*c1)
    ax.scatter(x0, y0, color=C_BLUE,  alpha=0.85, s=28, zorder=3, label="Class 0")
    ax.scatter(x1, y1, color=C_AMBER, alpha=0.85, s=28, zorder=3, label="Class 1")
    # support vectors: closest point from each class
    sv0 = min(c0, key=lambda p: abs(p[0]-bx))
    sv1 = min(c1, key=lambda p: abs(p[0]-bx))
    for sv in (sv0, sv1):
        circle = plt.Circle(sv, 0.18, fill=False, color="#444", lw=1.2, zorder=4)
        ax.add_patch(circle)
    ax.text(bx, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            "boundary", ha="center", va="bottom", fontsize=7, color=C_RED)
    ax.text((bx-margin + bx)/2, -1.6, "← margin →",
            ha="center", fontsize=7, color=C_GRAY)
    ax.set_xlim(-3, 2.5); ax.set_ylim(-2, 1.8)
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    sv_patch = mpatches.Patch(facecolor="none", edgecolor="#444",
                               lw=1.2, label="Support vectors")
    ax.legend(handles=[
        mpatches.Patch(color=C_BLUE,  label="Class 0"),
        mpatches.Patch(color=C_AMBER, label="Class 1"),
        sv_patch
    ], fontsize=8, framealpha=0)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def chart_nn():
    """Draw NN architecture with matplotlib."""
    layer_sizes = [4, 5, 5, 3, 2]
    layer_labels = ["Input\n(4)", "Hidden 1\n(5)", "Hidden 2\n(5)", "Hidden 3\n(3)", "Output\n(2)"]
    colors = [C_BLUE, "#a78bfa", "#a78bfa", "#a78bfa", C_AMBER]
    x_positions = np.linspace(0.1, 0.9, len(layer_sizes))
    max_n = max(layer_sizes)

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.1); ax.axis("off")

    node_pos = []
    for li, (n, xp) in enumerate(zip(layer_sizes, x_positions)):
        ys = np.linspace(0.1, 0.9, n) if n > 1 else [0.5]
        node_pos.append(list(zip([xp]*n, ys)))
        for _, yp in zip(range(n), ys):
            circle = plt.Circle((xp, yp), 0.035,
                color=colors[li], alpha=0.85, zorder=3)
            ax.add_patch(circle)
            circle2 = plt.Circle((xp, yp), 0.035,
                fill=False, edgecolor=colors[li], lw=0.8, zorder=4)
            ax.add_patch(circle2)
        ax.text(xp, -0.02, layer_labels[li], ha="center", va="top",
                fontsize=6.5, color=C_GRAY, multialignment="center")

    for li in range(len(layer_sizes)-1):
        for (x1, y1) in node_pos[li]:
            for (x2, y2) in node_pos[li+1]:
                ax.plot([x1, x2], [y1, y2], color="#cccccc", lw=0.4,
                        alpha=0.6, zorder=1)

    fig.tight_layout(pad=0.2)
    return fig_to_b64(fig)


def chart_rf():
    labels = ["1 tree", "50 trees", "100 trees", "200 trees"]
    vals   = [0.78, 0.89, 0.93, 0.94]
    alphas = [0.35, 0.55, 0.75, 0.95]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    bars = ax.bar(labels, vals, color=[C_BLUE]*4, alpha=1)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.set_ylim(0.6, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Test accuracy vs number of trees", fontsize=9, color=C_GRAY, pad=4)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def chart_xgb():
    rounds = list(range(1, 11))
    train  = [0.62, 0.48, 0.38, 0.31, 0.26, 0.22, 0.19, 0.17, 0.15, 0.14]
    val    = [0.64, 0.52, 0.43, 0.37, 0.33, 0.31, 0.29, 0.28, 0.28, 0.27]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    ax.plot(rounds, train, color=C_BLUE,  lw=2,   marker="o", ms=4, label="Training loss")
    ax.plot(rounds, val,   color=C_AMBER, lw=2,   marker="o", ms=4,
            ls="--", label="Validation loss")
    ax.set_xticks(rounds); ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=8)
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def chart_lda():
    r0, r1, r2 = rng_seq(1), rng_seq(7), rng_seq(3)
    c0 = [(r0()*1.8-2.8, r0()*1.8-1)   for _ in range(18)]
    c1 = [(r1()*1.8-0.5, r1()*1.8-0.8) for _ in range(18)]
    c2 = [(r2()*1.8+0.5, r2()*1.8-2)   for _ in range(18)]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#f5f4f0")
    ax.set_facecolor("#f5f4f0")
    for pts, col, lbl in [(c0, C_BLUE, "Class 0"),
                           (c1, C_AMBER,"Class 1"),
                           (c2, C_GREEN,"Class 2")]:
        x, y = zip(*pts)
        ax.scatter(x, y, color=col, alpha=0.82, s=22, label=lbl)
    lx = np.array([-4, 3])
    ax.plot(lx, lx*0.55 + 0.4, color=C_RED, lw=1.5, ls="--", label="LD1 axis")
    ax.set_xlim(-4, 3); ax.set_ylim(-2.5, 2)
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


# ── algorithm data ─────────────────────────────────────────────────────────────

ALGORITHMS = [
    {
        "title": "Naive Bayes",
        "sub": "Probability-based classifier",
        "type": "Probabilistic",
        "cliff": (
            "Uses Bayes' theorem to predict class probabilities. Assumes all features "
            "are independent — which is rarely true in real life, hence \"naive\". "
            "Despite this, it works surprisingly well for text data."
        ),
        "use_case_title": "Email spam detection",
        "use_case": (
            "Given the words in an email, what's the probability it's spam? "
            "Each word is treated independently, e.g. P(spam | 'free', 'win', 'click')."
        ),
        "code": """\
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = ["win free money now", "meeting at 3pm",
          "click here for prize"]
labels = [1, 0, 1]  # 1=spam, 0=ham

vec = CountVectorizer()
X   = vec.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

pred = model.predict(vec.transform(["free prize click"]))
print(pred)  # [1] → spam""",
        "viz_label": "Word probabilities — spam vs ham",
        "chart_fn": chart_nb,
    },
    {
        "title": "Logistic Regression",
        "sub": "Linear decision boundary",
        "type": "Linear",
        "cliff": (
            "Fits a linear boundary between classes, then squashes the output through "
            "a sigmoid function to get a probability between 0 and 1. "
            "Fast, interpretable, and a great baseline for any classification problem."
        ),
        "use_case_title": "Customer churn prediction",
        "use_case": (
            "Given features like usage frequency, support tickets, and contract length — "
            "what's the probability a customer will cancel?"
        ),
        "code": """\
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2,
                            n_redundant=0, random_state=42)
model = LogisticRegression()
model.fit(X, y)

print(model.predict([[0.5, -1.2]]))
# → [0] or [1]
print(model.predict_proba([[0.5, -1.2]]))
# → [[0.73, 0.27]]""",
        "viz_label": "2D scatter with decision boundary",
        "chart_fn": chart_lr,
    },
    {
        "title": "Decision Trees",
        "sub": "Rule-based splitting",
        "type": "Tree-based",
        "cliff": (
            "Recursively splits data into branches by asking yes/no questions on features. "
            "Very interpretable — you can literally read the rules. "
            "Prone to overfitting without pruning."
        ),
        "use_case_title": "Loan approval",
        "use_case": (
            "If income > 50k AND credit score > 700 AND debt ratio < 0.3 → approve. "
            "Each split mirrors how a loan officer actually thinks."
        ),
        "code": """\
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

rules = export_text(model,
    feature_names=load_iris().feature_names)
print(rules)

print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
# → [0] = setosa""",
        "viz_label": "Iris decision tree structure",
        "chart_fn": chart_dt,
    },
    {
        "title": "K-Nearest Neighbors",
        "sub": "Majority vote from neighbors",
        "type": "Instance-based",
        "cliff": (
            "No training phase — at prediction time it finds the K closest data points "
            "and takes a majority vote. Simple and intuitive, but slow on large datasets "
            "and sensitive to irrelevant features."
        ),
        "use_case_title": "Movie recommendations",
        "use_case": (
            "Find the K users most similar to you based on viewing history. "
            "Recommend what they watched that you haven't."
        ),
        "code": """\
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=300, n_features=2,
                            n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))   # accuracy
print(model.predict([[0.0, 0.5]]))   # class""",
        "viz_label": "K=3 nearest neighbors — query → class 0",
        "chart_fn": chart_knn,
    },
    {
        "title": "Support Vector Machines",
        "sub": "Maximum margin classifier",
        "type": "Margin-based",
        "cliff": (
            "Finds the hyperplane that maximises the margin between classes. "
            "Uses only the nearest data points (support vectors) to define the boundary. "
            "Kernels let it handle non-linear data."
        ),
        "use_case_title": "Handwritten digit classification",
        "use_case": (
            "Works well in high-dimensional spaces with a clear margin. "
            "A classic approach for MNIST digit recognition."
        ),
        "code": """\
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=2,
                            n_redundant=0, random_state=42)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(X_scaled, y)

print(model.predict(scaler.transform([[0.5, -1.0]])))
print(f"Support vectors: {model.support_vectors_.shape[0]}")""",
        "viz_label": "Maximum margin + support vectors",
        "chart_fn": chart_svm,
    },
    {
        "title": "Neural Networks",
        "sub": "Layered feature learning",
        "type": "Deep learning",
        "cliff": (
            "Stacked layers of neurons learn increasingly abstract features. "
            "Each layer transforms input via weights + activation functions. "
            "The network adjusts weights through backpropagation during training."
        ),
        "use_case_title": "Sentiment analysis",
        "use_case": (
            "Classify customer reviews as positive or negative. "
            "The layers learn to detect words, phrases, and context automatically."
        ),
        "code": """\
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=4,
                            n_redundant=0, random_state=42)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2 hidden layers: 64 and 32 neurons
model = MLPClassifier(hidden_layer_sizes=(64, 32),
                      activation='relu', max_iter=500,
                      random_state=42)
model.fit(X_scaled, y)

print(model.score(X_scaled, y))   # training accuracy""",
        "viz_label": "Neural network layer architecture",
        "chart_fn": chart_nn,
    },
    {
        "title": "Random Forest",
        "sub": "Ensemble of diverse trees",
        "type": "Ensemble",
        "cliff": (
            "Builds hundreds of decision trees on random subsets of data and features, "
            "then aggregates their votes. The diversity between trees reduces overfitting. "
            "One of the most reliable all-round classifiers."
        ),
        "use_case_title": "Medical diagnosis",
        "use_case": (
            "Predict whether a patient has a disease based on many lab values. "
            "The ensemble smooths out noise in individual test results."
        ),
        "code": """\
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=10,
                            n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.3f}")

importances = model.feature_importances_
top3 = sorted(enumerate(importances), key=lambda x:-x[1])[:3]
print(top3)  # top 3 most important features""",
        "viz_label": "Accuracy vs number of trees",
        "chart_fn": chart_rf,
    },
    {
        "title": "XGBoost / Gradient Boosting",
        "sub": "Sequential error correction",
        "type": "Ensemble",
        "cliff": (
            "Builds trees one at a time, where each tree focuses on correcting the "
            "errors of the previous. Uses gradient descent to minimise a loss function. "
            "Fast, powerful, regularised — Kaggle's favourite."
        ),
        "use_case_title": "Credit scoring",
        "use_case": (
            "Predict whether a loan applicant will default. "
            "Handles missing data natively, works well with mixed feature types."
        ),
        "code": """\
# pip install xgboost
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=10,
                            n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

model = XGBClassifier(n_estimators=100, learning_rate=0.1,
                      max_depth=4, eval_metric='logloss',
                      random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=False)

print(f"Accuracy: {model.score(X_test, y_test):.3f}")""",
        "viz_label": "Training vs validation loss per round",
        "chart_fn": chart_xgb,
    },
    {
        "title": "LDA / QDA",
        "sub": "Discriminant analysis",
        "type": "Statistical",
        "cliff": (
            "LDA finds a linear combination of features that best separates classes "
            "(assumes equal covariance). QDA is the flexible variant allowing each class "
            "its own covariance. Both are fast and work well with small datasets."
        ),
        "use_case_title": "Face recognition",
        "use_case": (
            "LDA (Fisherfaces) was a classic approach — project face images onto the "
            "dimensions that best separate identities."
        ),
        "code": """\
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
print(f"LDA accuracy: {lda.score(X, y):.3f}")

qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)
print(f"QDA accuracy: {qda.score(X, y):.3f}")

X_2d = lda.transform(X)   # collapse to (n_classes-1) dims
print(X_2d.shape)          # (178, 2)""",
        "viz_label": "3-class separation + LD1 axis",
        "chart_fn": chart_lda,
    },
]


# ── HTML builder ───────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

@page {
    size: A4;
    margin: 15mm 12mm 15mm 12mm;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:      #f5f4f0;
    --surface: #ffffff;
    --border:  #e0ddd6;
    --text:    #1a1917;
    --muted:   #6b6860;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
    --code-bg: #1c1b18;
    --code-fg: #e8e4d9;
}

body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    font-size: 10pt;
    line-height: 1.55;
    padding: 0;
}

header {
    border-bottom: 2pt solid var(--text);
    padding-bottom: 8pt;
    margin-bottom: 14pt;
}

header h1 { font-size: 18pt; font-weight: 600; letter-spacing: -0.02em; }
header p  { color: var(--muted); font-size: 9pt; margin-top: 2pt; font-weight: 300; }

.algo {
    background: var(--surface);
    border: 0.5pt solid var(--border);
    border-radius: 5pt;
    margin-bottom: 12pt;
    page-break-inside: avoid;
    overflow: hidden;
    width: 100%;
}

.algo-header {
    padding: 7pt 10pt 6pt;
    border-bottom: 0.5pt solid var(--border);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8pt;
}

.algo-title { font-size: 12pt; font-weight: 600; letter-spacing: -0.01em; }
.algo-sub   { font-size: 8pt;  color: var(--muted); font-weight: 300; margin-top: 1pt; }

.badge {
    font-size: 7.5pt;
    font-weight: 500;
    padding: 2pt 7pt;
    border-radius: 20pt;
    white-space: nowrap;
    font-family: var(--mono);
    flex-shrink: 0;
}

/* ── three-row card layout ── */
.info-row {
    display: table;
    width: 100%;
    table-layout: fixed;
    border-bottom: 0.5pt solid var(--border);
}

.cliff-col, .use-col {
    display: table-cell;
    width: 50%;
    vertical-align: top;
    padding: 8pt 10pt;
}

.cliff-col { border-right: 0.5pt solid var(--border); }

.code-row {
    border-bottom: 0.5pt solid var(--border);
    padding: 8pt 10pt;
}

.viz-row {
    padding: 8pt 10pt;
}

.section-label {
    font-size: 7pt;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4pt;
}

.cliff-note {
    font-size: 9pt;
    line-height: 1.6;
    color: var(--text);
    font-weight: 300;
}

.use-case {
    font-size: 8.5pt;
    line-height: 1.55;
    color: var(--muted);
    font-weight: 300;
    border-left: 2pt solid var(--border);
    padding-left: 7pt;
}

.use-case strong {
    color: var(--text);
    font-weight: 500;
    display: block;
    margin-bottom: 1pt;
}

pre {
    background: var(--code-bg);
    color: var(--code-fg);
    font-family: var(--mono);
    font-size: 7.5pt;
    line-height: 1.65;
    padding: 8pt 10pt;
    border-radius: 4pt;
    white-space: pre;
    overflow: hidden;
    word-break: normal;
}

.chart-wrap {
    background: var(--bg);
    border-radius: 4pt;
    padding: 4pt;
    text-align: center;
}

.chart-wrap img {
    width: 100%;
    height: auto;
    display: block;
}

.viz-label {
    font-size: 7pt;
    color: var(--muted);
    font-family: var(--mono);
    margin-top: 4pt;
    text-align: center;
}
"""


def escape(text):
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def build_card(algo, chart_b64):
    bg, tc = BADGE_COLORS.get(algo["type"], ("#e5e7eb", "#374151"))
    return f"""
<div class="algo">
  <div class="algo-header">
    <div>
      <div class="algo-title">{escape(algo['title'])}</div>
      <div class="algo-sub">{escape(algo['sub'])}</div>
    </div>
    <span class="badge" style="background:{bg};color:{tc}">{escape(algo['type'])}</span>
  </div>
  <div class="info-row">
    <div class="cliff-col">
      <div class="section-label">Cliff note</div>
      <p class="cliff-note">{escape(algo['cliff'])}</p>
    </div>
    <div class="use-col">
      <div class="section-label">Use case</div>
      <div class="use-case">
        <strong>{escape(algo['use_case_title'])}</strong>
        {escape(algo['use_case'])}
      </div>
    </div>
  </div>
  <div class="code-row">
    <div class="section-label">Code</div>
    <pre>{escape(algo['code'])}</pre>
  </div>
  <div class="viz-row">
    <div class="section-label">Visual — {escape(algo['viz_label'])}</div>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{chart_b64}" alt="{escape(algo['viz_label'])}">
    </div>
  </div>
</div>"""


def build_html(title, subtitle, cards_html):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <p>{escape(subtitle)}</p>
</header>
{''.join(cards_html)}
</body>
</html>"""


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ML study sheet PDF")
    parser.add_argument("--output", default="classification_algorithms.pdf",
                        help="Output PDF filename (default: classification_algorithms.pdf)")
    parser.add_argument("--html", action="store_true",
                        help="Also save the intermediate HTML file")
    args = parser.parse_args()

    print("Generating charts...")
    cards = []
    for i, algo in enumerate(ALGORITHMS, 1):
        print(f"  [{i}/{len(ALGORITHMS)}] {algo['title']}")
        chart_b64 = algo["chart_fn"]()
        cards.append(build_card(algo, chart_b64))

    title    = "ML Classification Algorithms"
    subtitle = "Study sheet — cliff notes, use cases, scikit-learn code, visuals"
    html     = build_html(title, subtitle, cards)

    if args.html:
        html_path = args.output.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML saved → {html_path}")

    print(f"Rendering PDF → {args.output} ...")
    try:
        from weasyprint import HTML as WP_HTML
        WP_HTML(string=html).write_pdf(args.output)
        print(f"Done! ✓  {args.output}")
    except ImportError:
        print("WeasyPrint not installed. Run:  pip install weasyprint")
        print("Saving HTML instead...")
        html_path = args.output.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML saved → {html_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()