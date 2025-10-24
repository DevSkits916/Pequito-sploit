from __future__ import annotations

from collections import OrderedDict
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from flask import Flask, request

try:
  from rng_algorithms import (
    ALGORITHMS,
    ALGORITHM_MAP,
    DETECTABLE_ALGORITHMS,
    AlgorithmHandler,
    AnalysisResult,
  )
except ModuleNotFoundError:  # pragma: no cover - fallback for package execution
  from server.rng_algorithms import (  # type: ignore[import-not-found]
    ALGORITHMS,
    ALGORITHM_MAP,
    DETECTABLE_ALGORITHMS,
    AlgorithmHandler,
    AnalysisResult,
  )

Number = Union[int, float]

app = Flask(__name__)


@dataclass
class Result:
  parameters: str = '—'
  forward: str = '—'
  backward: str = '—'
  generated: str = '—'
  error: Optional[str] = None
  algorithm: Optional[str] = None


_GROUPED: OrderedDict[str, List[AlgorithmHandler]] = OrderedDict()
for algorithm in ALGORITHMS:
  _GROUPED.setdefault(algorithm.category, []).append(algorithm)
ALGORITHM_GROUPS: List[Tuple[str, List[AlgorithmHandler]]] = list(_GROUPED.items())
DEFAULT_ALGORITHM = ALGORITHM_MAP['lcg']
ALGORITHM_DETAILS = {
  handler.key: {
    'label': handler.label,
    'category': handler.category,
    'supports_detection': handler.supports_detection,
    'description': handler.description,
  }
  for handler in ALGORITHMS
}


def parse_sequence(raw: str) -> List[Number]:
  if not raw:
    return []
  clean = raw.replace('\n', ' ').replace(',', ' ')
  tokens = [token.strip() for token in clean.split()]
  result: List[Number] = []
  for token in tokens:
    if not token:
      continue
    try:
      value = int(token, 0)
      result.append(value)
      continue
    except ValueError:
      pass
    try:
      value = float(token)
    except ValueError:
      continue
    result.append(value)
  return result


def _value_to_string(value: Number) -> str:
  if isinstance(value, float):
    return f'{value:.12g}'
  return str(value)


def format_values(values: Iterable[Number]) -> str:
  materialised = list(values)
  if not materialised:
    return '—'
  return ', '.join(_value_to_string(value) for value in materialised)


def analysis_to_result(label: str, analysis: AnalysisResult) -> Result:
  forward_text = format_values(analysis.forward)
  generated_text = format_values(analysis.generated)
  if forward_text == '—' and generated_text != '—':
    forward_text = generated_text
  return Result(
    algorithm=label,
    parameters=f'{label}: {analysis.parameters}',
    forward=forward_text,
    backward=format_values(analysis.backward),
    generated=generated_text,
  )


def load_sample_defaults(form_data: dict, algorithm: AlgorithmHandler) -> None:
  sample = algorithm.sample or {}
  if algorithm.supports_detection and 'sequence' in sample:
    sequence_values = sample.get('sequence', [])
    form_data['sequence'] = ', '.join(_value_to_string(value) for value in sequence_values)
    form_data['forward'] = sample.get('forward', form_data.get('forward', 5))
    form_data['backward'] = sample.get('backward', form_data.get('backward', 0))
  else:
    seed_values = sample.get('seed')
    if isinstance(seed_values, (list, tuple)):
      form_data['sequence'] = ', '.join(_value_to_string(value) for value in seed_values)
    elif seed_values is None:
      form_data['sequence'] = ''
    else:
      form_data['sequence'] = _value_to_string(seed_values)
    form_data['forward'] = sample.get('count', form_data.get('forward', 10))
    form_data['backward'] = 0


def attempt_auto_detection(sequence: List[Number], forward: int, backward: int) -> Tuple[AnalysisResult, AlgorithmHandler]:
  errors: List[str] = []
  for algorithm in DETECTABLE_ALGORITHMS:
    try:
      analysis = algorithm.analyzer(sequence, forward, backward)
      return analysis, algorithm
    except Exception as exc:  # noqa: BLE001
      errors.append(f"{algorithm.label}: {exc}")
  joined = ' | '.join(errors)
  message = 'Unable to match sequence to a supported algorithm.'
  if joined:
    message = f'{message} Details: {joined}'
  raise ValueError(message)


def render_page(form_data: dict, result: Result) -> str:
  return TEMPLATE.render(
    sequence=form_data.get('sequence', ''),
    forward=form_data.get('forward', 5),
    backward=form_data.get('backward', 0),
    mode=form_data.get('mode', 'auto'),
    result=result,
    algorithm_groups=ALGORITHM_GROUPS,
    algorithm_details_json=json.dumps(ALGORITHM_DETAILS),
    detectable_count=len(DETECTABLE_ALGORITHMS),
    total_count=len(ALGORITHMS),
    input_count=len(parse_sequence(str(form_data.get('sequence', '')))),
  )


@app.route('/', methods=['GET', 'POST'])
def index():
  form_data = {
    'sequence': ', '.join(_value_to_string(value) for value in DEFAULT_ALGORITHM.sample['sequence']),
    'forward': DEFAULT_ALGORITHM.sample.get('forward', 5),
    'backward': DEFAULT_ALGORITHM.sample.get('backward', 5),
    'mode': 'auto',
  }
  result = Result()

  if request.method == 'GET':
    return render_page(form_data, result)

  form_data['sequence'] = request.form.get('sequence', form_data['sequence'])
  form_data['forward'] = request.form.get('forward', form_data['forward'])
  form_data['backward'] = request.form.get('backward', form_data['backward'])
  form_data['mode'] = request.form.get('mode', form_data['mode'])

  selected_key = form_data['mode']
  selected_algorithm = ALGORITHM_MAP.get(selected_key, DEFAULT_ALGORITHM)

  if request.form.get('load_sample'):
    load_sample_defaults(form_data, selected_algorithm)
    return render_page(form_data, result)

  try:
    forward = max(0, int(form_data['forward'] or 0))
    backward = max(0, int(form_data['backward'] or 0))
  except ValueError:
    result.error = 'Forward and backward counts must be integers.'
    return render_page(form_data, result)

  form_data['forward'] = forward
  form_data['backward'] = backward

  sequence = parse_sequence(form_data['sequence'])

  try:
    if selected_key == 'auto':
      if not sequence:
        raise ValueError('Please provide at least one number to analyze.')
      analysis, detected_algorithm = attempt_auto_detection(sequence, forward, backward)
      result = analysis_to_result(detected_algorithm.label, analysis)
    else:
      algorithm = ALGORITHM_MAP.get(selected_key)
      if algorithm is None:
        raise ValueError('Unknown algorithm requested.')
      if algorithm.supports_detection and algorithm.analyzer:
        if not sequence:
          raise ValueError('Please provide at least one number to analyze.')
        analysis = algorithm.analyzer(sequence, forward, backward)
        result = analysis_to_result(algorithm.label, analysis)
      else:
        if algorithm.generator is None:
          raise ValueError('Generation routine not available for this algorithm.')
        count = forward or int(algorithm.sample.get('count', 10))
        form_data['forward'] = count
        analysis = algorithm.generator(sequence, count)
        result = analysis_to_result(algorithm.label, analysis)
        result.backward = '—'
  except Exception as exc:  # noqa: BLE001
    result = Result(error=str(exc))

  return render_page(form_data, result)


class TemplateWrapper:
  def __init__(self, text: str) -> None:
    self.text = text

  def render(self, **context):
    from jinja2 import Template

    template = Template(self.text)
    return template.render(**context)


TEMPLATE = TemplateWrapper(
  """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RNG Analyzer</title>
  <style>
:root {
  color-scheme: dark;
  --bg: #050816;
  --bg-alt: rgba(15, 23, 42, 0.78);
  --bg-alt-strong: rgba(15, 23, 42, 0.92);
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #7c3aed;
  --accent-soft: rgba(124, 58, 237, 0.25);
  --success: #34d399;
  --error: #f87171;
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  min-height: 100vh;
  background: radial-gradient(circle at 20% -10%, rgba(147, 51, 234, 0.38), transparent 55%),
    radial-gradient(circle at 90% 10%, rgba(14, 116, 144, 0.34), transparent 55%),
    linear-gradient(180deg, #050816 0%, #0f172a 100%);
  color: var(--text);
  display: flex;
  justify-content: center;
  padding: 3.5rem 1.2rem 4rem;
}
.app-shell {
  width: min(1100px, 100%);
  display: grid;
  gap: 1.5rem;
}
.app-header {
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.65));
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  padding: 2.1rem 2rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 25px 65px rgba(2, 6, 23, 0.55);
}
.app-header::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at top right, rgba(124, 58, 237, 0.35), transparent 60%);
  pointer-events: none;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.9rem;
  border-radius: 999px;
  background: rgba(124, 58, 237, 0.18);
  border: 1px solid rgba(124, 58, 237, 0.45);
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #c4b5fd;
  position: relative;
  z-index: 1;
}
.app-header h1 {
  margin: 0.4rem 0 0;
  font-size: clamp(2.4rem, 5vw, 3.1rem);
  letter-spacing: -0.015em;
  position: relative;
  z-index: 1;
}
.subtitle {
  margin: 0.6rem 0 0;
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 60ch;
  position: relative;
  z-index: 1;
}
.status-card {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  align-items: stretch;
  background: var(--bg-alt);
  border-radius: 18px;
  padding: 1.4rem 1.8rem;
  border: 1px solid rgba(148, 163, 184, 0.14);
  box-shadow: 0 20px 55px rgba(15, 23, 42, 0.45);
}
.status-tile {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.status-label {
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.55);
}
.status-value {
  font-size: 1.35rem;
  font-weight: 600;
}
.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: rgba(52, 211, 153, 0.16);
  border: 1px solid rgba(52, 211, 153, 0.45);
  color: #6ee7b7;
  font-size: 0.82rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.layout-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
  align-items: start;
}
.card {
  background: var(--bg-alt-strong);
  border-radius: 18px;
  padding: 1.8rem;
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 18px 45px rgba(2, 6, 23, 0.55);
}
.card h2 {
  margin: 0 0 1rem;
  font-size: 1.35rem;
}
.field-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.88rem;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
textarea,
input,
select {
  width: 100%;
  background: rgba(10, 16, 32, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 12px;
  padding: 0.75rem 0.9rem;
  color: var(--text);
  font-size: 1rem;
  font-family: inherit;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
textarea:focus,
input:focus,
select:focus {
  outline: none;
  border-color: rgba(124, 58, 237, 0.7);
  box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2);
  transform: translateY(-1px);
}
textarea {
  min-height: 180px;
  resize: vertical;
}
.options-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  margin-top: 1.2rem;
}
.button-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 1.4rem;
}
button {
  cursor: pointer;
  border-radius: 12px;
  padding: 0.8rem 1.6rem;
  border: none;
  font-weight: 600;
  letter-spacing: 0.03em;
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}
button.primary {
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.95), rgba(79, 70, 229, 0.95));
  color: white;
  box-shadow: 0 18px 35px rgba(76, 29, 149, 0.45);
}
button:not(.primary) {
  background: rgba(15, 23, 42, 0.75);
  color: var(--text);
  border: 1px solid rgba(148, 163, 184, 0.22);
}
button:hover {
  transform: translateY(-2px);
}
button:active {
  transform: translateY(0);
}
.info-panel {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}
.algorithm-list {
  display: grid;
  gap: 0.6rem;
  max-height: 280px;
  overflow-y: auto;
  padding-right: 0.2rem;
}
.algorithm-group {
  display: grid;
  gap: 0.45rem;
}
.algorithm-group-title {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(226, 232, 240, 0.45);
}
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.algo-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.2);
  color: var(--text);
  font-size: 0.78rem;
  transition: border 0.18s ease, background 0.18s ease, color 0.18s ease;
}
.algo-chip[data-active='true'] {
  border-color: rgba(124, 58, 237, 0.7);
  background: rgba(124, 58, 237, 0.2);
  color: #c4b5fd;
}
.algorithm-meta {
  display: grid;
  gap: 0.65rem;
  padding: 1rem 1.1rem;
  border-radius: 16px;
  background: rgba(15, 23, 42, 0.65);
  border: 1px solid rgba(124, 58, 237, 0.2);
}
.algorithm-meta h3 {
  margin: 0;
  font-size: 1.1rem;
}
.algorithm-meta p {
  margin: 0;
  color: rgba(226, 232, 240, 0.7);
  line-height: 1.5;
}
.meta-badges {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.meta-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.7);
}
.results-card {
  display: grid;
  gap: 1.4rem;
}
.results-headline {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.8rem;
}
.results-headline h2 {
  margin: 0;
}
.hint {
  margin: 0;
  color: var(--muted);
  font-size: 0.88rem;
}
.results-section {
  display: grid;
  gap: 0.45rem;
}
pre {
  margin: 0;
  background: rgba(10, 16, 32, 0.82);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  border: 1px solid rgba(148, 163, 184, 0.18);
  font-size: 0.95rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  overflow-x: auto;
}
.hidden {
  display: none;
}
#error-card {
  border: 1px solid rgba(248, 113, 113, 0.45);
}
#error-output {
  color: var(--error);
}
.app-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
  color: rgba(226, 232, 240, 0.55);
  padding: 0 0.4rem;
}
.footer-links {
  display: flex;
  gap: 0.9rem;
}
.footer-links span {
  opacity: 0.7;
}
@media (max-width: 980px) {
  .layout-grid {
    grid-template-columns: 1fr;
  }
  .algorithm-list {
    max-height: none;
  }
}
@media (max-width: 640px) {
  body {
    padding: 2.4rem 0.8rem 3rem;
  }
  .card {
    padding: 1.4rem;
  }
  .button-row {
    flex-direction: column;
    align-items: stretch;
  }
  button {
    width: 100%;
  }
}
  </style>
</head>
<body>
  <main class="app-shell">
    <header class="app-header">
      <div class="badge">DevSkits916</div>
      <h1>RNG Intelligence Console</h1>
      <p class="subtitle">Identify, interrogate, and synthesise sequences across congruential, xor-based, cryptographic, quasi-random, and chaotic generators.</p>
    </header>

    <section class="status-card">
      <div class="status-tile">
        <span class="status-label">Current mode</span>
        <span class="status-value" id="status-mode">—</span>
      </div>
      <div class="status-tile">
        <span class="status-label">Detection coverage</span>
        <span class="status-value">{{ detectable_count }} / {{ total_count }}</span>
      </div>
      <div class="status-tile">
        <span class="status-label">Sequence length</span>
        <span class="status-value">{{ input_count }}</span>
      </div>
      <div class="status-tile">
        <span class="status-label">Auto detection</span>
        <span class="status-chip">{{ 'Enabled' if mode == 'auto' else 'Manual' }}</span>
      </div>
    </section>

    <div class="layout-grid">
      <form method="post" class="card">
        <h2>Sequence intake</h2>
        <label class="field-label" for="sequence">
          Observed sequence or seed material
          <span>Comma, space, or newline separated. Hex like 0x123 welcome.</span>
        </label>
        <textarea id="sequence" name="sequence" rows="6">{{ sequence }}</textarea>
        <div class="options-grid">
          <label>
            <span class="field-label">Forward predictions / count</span>
            <input type="number" name="forward" min="0" value="{{ forward }}" />
          </label>
          <label>
            <span class="field-label">Backward predictions</span>
            <input type="number" name="backward" min="0" value="{{ backward }}" />
          </label>
          <label>
            <span class="field-label">Algorithm mode</span>
            <select name="mode">
              <option value="auto" {% if mode == 'auto' %}selected{% endif %}>Auto Detect / Infer</option>
              {% for category, items in algorithm_groups %}
                <optgroup label="{{ category }}">
                  {% for entry in items %}
                    <option value="{{ entry.key }}" {% if mode == entry.key %}selected{% endif %}>{{ entry.label }}</option>
                  {% endfor %}
                </optgroup>
              {% endfor %}
            </select>
          </label>
        </div>
        <div class="button-row">
          <button type="submit" class="primary">Analyze / Predict</button>
          <button type="submit" name="load_sample" value="1">Load Sample Configuration</button>
        </div>
      </form>

      <aside class="card info-panel">
        <h2>Algorithm Atlas</h2>
        <div class="algorithm-list">
          {% for category, items in algorithm_groups %}
            <div class="algorithm-group">
              <span class="algorithm-group-title">{{ category }}</span>
              <div class="chip-row">
                {% for entry in items %}
                  <button type="button" class="algo-chip" data-key="{{ entry.key }}">{{ entry.label }}</button>
                {% endfor %}
              </div>
            </div>
          {% endfor %}
        </div>
        <div class="algorithm-meta">
          <h3 id="algorithm-title">Select an algorithm</h3>
          <div class="meta-badges">
            <span class="meta-pill" id="algorithm-category">—</span>
            <span class="meta-pill" id="algorithm-detection">—</span>
          </div>
          <p id="algorithm-description">Choose a generator to review its role and capabilities. Selecting a chip also tunes the form to that mode.</p>
        </div>
      </aside>
    </div>

    <section class="card results-card" id="results-card">
      <div class="results-headline">
        <h2>Results</h2>
        {% if result.algorithm %}<span class="hint">Active algorithm: {{ result.algorithm }}</span>{% endif %}
      </div>
      <div class="results-section">
        <h3>Parameters</h3>
        <pre id="parameters-output">{{ result.parameters }}</pre>
      </div>
      <div class="results-section">
        <h3>Forward predictions</h3>
        <p class="hint">Predicted or generated states after your last provided number.</p>
        <pre id="forward-output">{{ result.forward }}</pre>
      </div>
      <div class="results-section">
        <h3>Backward predictions</h3>
        <p class="hint">States before your first provided number (where supported).</p>
        <pre id="backward-output">{{ result.backward }}</pre>
      </div>
      <div class="results-section">
        <h3>Generated sequence</h3>
        <pre id="generated-output">{{ result.generated }}</pre>
      </div>
    </section>

    <section class="card {% if not result.error %}hidden{% endif %}" id="error-card">
      <h2>Error</h2>
      <pre id="error-output">{{ result.error or '' }}</pre>
    </section>

    <footer class="app-footer">
      <span>RNG Analyzer · Precision tooling for entropy sleuths</span>
      <div class="footer-links">
        <span>{{ total_count }} algorithms bundled</span>
        <span>{{ detectable_count }} auto-detect capable</span>
      </div>
    </footer>
  </main>

  <script>
    const algorithmDetails = JSON.parse('{{ algorithm_details_json | safe }}');
    const modeSelect = document.querySelector('select[name="mode"]');
    const statusMode = document.getElementById('status-mode');
    const chips = Array.from(document.querySelectorAll('.algo-chip'));
    const title = document.getElementById('algorithm-title');
    const category = document.getElementById('algorithm-category');
    const detection = document.getElementById('algorithm-detection');
    const description = document.getElementById('algorithm-description');

    function describeDetection(meta) {
      if (!meta) {
        return '—';
      }
      return meta.supports_detection ? 'Auto detection ready' : 'Manual only';
    }

    function setActiveChip(key) {
      chips.forEach((chip) => {
        const active = chip.dataset.key === key;
        chip.setAttribute('data-active', active);
      });
    }

    function applySelection(key) {
      const meta = algorithmDetails[key];
      if (key === 'auto') {
        statusMode.textContent = 'Auto Detect';
        title.textContent = 'Auto detection';
        category.textContent = 'Best-match inference';
        detection.textContent = 'Scans all detectable algorithms';
        description.textContent = 'Submit an observed sequence and the analyzer will interrogate each detectable generator until a match is confirmed.';
        setActiveChip('');
        return;
      }
      statusMode.textContent = meta ? meta.label : 'Manual';
      if (meta) {
        title.textContent = meta.label;
        category.textContent = meta.category;
        detection.textContent = describeDetection(meta);
        description.textContent = meta.description || 'No description available.';
      } else {
        title.textContent = 'Custom algorithm';
        category.textContent = '—';
        detection.textContent = 'Manual selection';
        description.textContent = 'Manual selections surface bespoke generator behaviour.';
      }
      setActiveChip(key);
    }

    if (modeSelect) {
      applySelection(modeSelect.value);
      modeSelect.addEventListener('change', (event) => {
        applySelection(event.target.value);
      });
    }

    chips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const key = chip.dataset.key;
        if (modeSelect) {
          modeSelect.value = key;
        }
        applySelection(key);
      });
    });
  </script>
</body>
</html>
"""
)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
