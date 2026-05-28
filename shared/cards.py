from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from .styles import BADGE_COLORS, escape

_PY_FORMATTER = HtmlFormatter(style="one-dark", noclasses=True, nowrap=True)


def _highlight_python(code):
    """Return syntax-highlighted HTML for a Python code string."""
    inner = highlight(code, PythonLexer(), _PY_FORMATTER)
    bg = _PY_FORMATTER.style.background_color
    return (
        f'<pre style="background:{bg};padding:8pt 10pt;border-radius:4pt;'
        f'font-family:var(--mono);font-size:7.5pt;line-height:1.65;'
        f'white-space:pre;overflow:hidden">{inner}</pre>'
    )


def _md_table(md_text):
    """Convert a pipe-delimited markdown table string into an HTML table."""
    lines = [l.strip() for l in md_text.strip().splitlines()]
    # filter out separator rows (---|---) and blank lines
    rows = [l for l in lines if l and not set(l.replace("|", "").replace("-", "").replace(" ", "")) == set()]
    if not rows:
        return ""
    header_cells = [c.strip() for c in rows[0].strip("|").split("|")]
    th = "".join(f"<th>{escape(c)}</th>" for c in header_cells)
    body_rows = ""
    for row in rows[1:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        body_rows += "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>"
    return f'<table class="data-table"><thead><tr>{th}</tr></thead><tbody>{body_rows}</tbody></table>'


# ── algorithm card ─────────────────────────────────────────────────────────────
# Fields: title, type, cliff, use_case_title, use_case,
#         math   — list of str: "$...$" for LaTeX, plain text otherwise
#         use_when — list of str
#         code, viz_label, chart_fn (zero-arg callable → matplotlib figure)

def build_algorithm_card(algo, chart_b64):
    bg, tc = BADGE_COLORS.get(algo["type"], ("#e5e7eb", "#374151"))
    when_items = "".join(f'<li>{escape(i)}</li>' for i in algo["use_when"])
    return f"""
<div class="algo">
  <div class="algo-header">
    <div class="algo-title">{escape(algo['title'])}</div>
    <span class="badge" style="background:{bg};color:{tc}">{escape(algo['type'])}</span>
  </div>
  <div class="cliff-row">
    <div class="section-label">Cliff note</div>
    <p class="cliff-note">{escape(algo['cliff'])}</p>
  </div>
  <div class="info-row">
    <div class="cliff-col">
      <div class="section-label">Use case</div>
      <div class="use-case">
        <strong>{escape(algo['use_case_title'])}</strong>
        {escape(algo['use_case'])}
      </div>
    </div>
    <div class="use-col">
      <div class="section-label">Use this when</div>
      <ul class="when-list">{when_items}</ul>
    </div>
  </div>
  <div class="math-row">
    <div class="section-label">How it works</div>
    <p class="math-note">{escape(algo['math'])}</p>
  </div>
  <div class="code-row">
    <div class="section-label">Code</div>
    {_highlight_python(algo['code'])}
  </div>
  <div class="viz-row">
    <div class="section-label">Visual — {escape(algo['viz_label'])}</div>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{chart_b64}" alt="{escape(algo['viz_label'])}">
    </div>
  </div>
</div>"""


# ── preprocessing card ─────────────────────────────────────────────────────────
# Fields: title, type, cliff, use_case_title, use_case, use_when, code,
#         dataset: {
#             before: { label: str, columns: [...], rows: [[...], ...] },
#             after:  { label: str, columns: [...], rows: [[...], ...] },
#         }

def _table(data):
    cols = "".join(f'<th>{escape(c)}</th>' for c in data["columns"])
    rows = "".join(
        "<tr>" + "".join(f'<td>{escape(v)}</td>' for v in row) + "</tr>"
        for row in data["rows"]
    )
    return f'<table class="data-table"><thead><tr>{cols}</tr></thead><tbody>{rows}</tbody></table>'


def build_code_card(item, _chart_b64=None):
    """Card layout: title + badge | description | optional dataset table | code block."""
    bg, tc = BADGE_COLORS.get(item["type"], ("#e5e7eb", "#374151"))
    dataset_html = ""
    if item.get("dataset_md"):
        dataset_html = f"""
  <div class="cliff-row">
    <div class="section-label">Sample data</div>
    {_md_table(item['dataset_md'])}
  </div>"""
    return f"""
<div class="algo">
  <div class="algo-header">
    <div class="algo-title">{escape(item['title'])}</div>
    <span class="badge" style="background:{bg};color:{tc}">{escape(item['type'])}</span>
  </div>
  <div class="cliff-row">
    <div class="section-label">Description</div>
    <p class="cliff-note">{escape(item['description'])}</p>
  </div>{dataset_html}
  <div class="code-row" style="border-bottom:none">
    <div class="section-label">Code</div>
    {_highlight_python(item['code'])}
  </div>
</div>"""


def build_preprocessing_card(item, _chart_b64=None):
    bg, tc = BADGE_COLORS.get(item["type"], ("#e5e7eb", "#374151"))
    when_items = "".join(f'<li>{escape(i)}</li>' for i in item["use_when"])
    dataset = item.get("dataset", {})
    before = dataset.get("before", {})
    after  = dataset.get("after",  {})
    return f"""
<div class="algo">
  <div class="algo-header">
    <div class="algo-title">{escape(item['title'])}</div>
    <span class="badge" style="background:{bg};color:{tc}">{escape(item['type'])}</span>
  </div>
  <div class="cliff-row">
    <div class="section-label">Cliff note</div>
    <p class="cliff-note">{escape(item['cliff'])}</p>
  </div>
  <div class="info-row">
    <div class="cliff-col">
      <div class="section-label">Use case</div>
      <div class="use-case">
        <strong>{escape(item['use_case_title'])}</strong>
        {escape(item['use_case'])}
      </div>
    </div>
    <div class="use-col">
      <div class="section-label">Use this when</div>
      <ul class="when-list">{when_items}</ul>
    </div>
  </div>
  <div class="meta-row">
    <div class="math-col">
      <div class="section-label">{escape(before.get('label', 'Before'))}</div>
      {_table(before) if before else ''}
    </div>
    <div class="when-col">
      <div class="section-label">{escape(after.get('label', 'After'))}</div>
      {_table(after) if after else ''}
    </div>
  </div>
  <div class="code-row">
    <div class="section-label">Code</div>
    <pre>{escape(item['code'])}</pre>
  </div>
</div>"""
