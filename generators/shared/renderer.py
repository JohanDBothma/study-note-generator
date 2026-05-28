import base64
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .styles import CSS, escape


# ── chart helpers ──────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def rng_seq(seed):
    x = float(seed)
    def _next():
        nonlocal x
        x = (x * 9301 + 49297) % 233280
        return x / 233280
    return _next


def new_fig():
    return plt.subplots(figsize=(10, 3.2), facecolor="white")


def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


# ── math block renderer ────────────────────────────────────────────────────────

def render_math_block(formula_lines):
    """Render pure $...$ formula lines as a PNG image using matplotlib mathtext."""
    n = len(formula_lines)
    fig_h = max(0.5, 0.5 * n)
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)

    for i, line in enumerate(formula_lines):
        y = 1.0 - (i + 0.5) / n
        ax.text(0.5, y, line, ha="center", va="center", fontsize=12,
                transform=ax.transAxes, color="#1a1917")

    return fig_to_b64(fig)


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(title, subtitle, cards_html, show_header=True):
    header = f"""<header>
  <h1>{escape(title)}</h1>
  <p>{escape(subtitle)}</p>
</header>""" if show_header else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
{header}
{''.join(cards_html)}
</body>
</html>"""


# ── PDF renderer ───────────────────────────────────────────────────────────────

def render(items, title, subtitle, output_path, save_html=False,
           card_builder=None, show_header=True):
    if card_builder is None:
        from .cards import build_algorithm_card
        card_builder = build_algorithm_card

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Generating cards...")
    cards = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['title']}")
        chart_b64 = item["chart_fn"]() if "chart_fn" in item else None
        cards.append(card_builder(item, chart_b64))

    html = build_html(title, subtitle, cards, show_header=show_header)

    if save_html:
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML saved -> {html_path}")

    print(f"Rendering PDF -> {output_path} ...")
    try:
        from weasyprint import HTML as WP_HTML
        WP_HTML(string=html).write_pdf(output_path)
        print(f"Done! {output_path}")
    except ImportError:
        print("WeasyPrint not installed. Run:  pip install weasyprint")
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML saved -> {html_path}")
        sys.exit(1)
