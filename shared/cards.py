from .styles import BADGE_COLORS, escape


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
    <pre>{escape(algo['code'])}</pre>
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
