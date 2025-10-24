from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from flask import Flask, request

from rng_algorithms import (
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
  """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>RNG Analyzer</title>
  <style>
:root {
  color-scheme: dark;
  --bg: #0f172a;
  --bg-alt: #1e293b;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent-start: #6366f1;
  --accent-end: #8b5cf6;
  --error: #f87171;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at top, rgba(99, 102, 241, 0.18), transparent 55%),
    radial-gradient(circle at bottom, rgba(139, 92, 246, 0.15), transparent 55%),
    var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  padding: 3rem 1rem 4rem;
}
.app-shell {
  width: min(960px, 100%);
  display: grid;
  gap: 1.5rem;
}
.app-header {
  text-align: center;
  padding: 1.5rem;
  background: rgba(15, 23, 42, 0.7);
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.1);
  backdrop-filter: blur(8px);
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.45);
}
.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.35rem 0.8rem;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
  border: 1px solid rgba(99, 102, 241, 0.5);
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-bottom: 0.6rem;
}
.app-header h1 { margin: 0; font-size: clamp(2rem, 4vw, 2.7rem); }
.subtitle { margin-top: 0.4rem; color: var(--muted); }
.card {
  background: var(--bg-alt);
  border-radius: 16px;
  padding: 1.8rem;
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 14px 35px rgba(15, 23, 42, 0.5);
}
.card h2 { margin-top: 0; font-size: 1.3rem; }
label {
  display: grid;
  gap: 0.4rem;
  font-weight: 600;
  color: var(--muted);
}
textarea, input, select {
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 10px;
  padding: 0.7rem;
  color: var(--text);
  font-size: 1rem;
  font-family: inherit;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
textarea:focus, input:focus, select:focus {
  outline: none;
  border-color: rgba(99, 102, 241, 0.8);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
}
.options {
  display: grid;
  gap: 1rem;
  margin-top: 1rem;
}
.options-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}
.button-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}
button {
  cursor: pointer;
  border-radius: 12px;
  padding: 0.75rem 1.4rem;
  border: none;
  font-weight: 600;
  letter-spacing: 0.02em;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
button.primary {
  background: linear-gradient(135deg, var(--accent-start), var(--accent-end));
  color: white;
  box-shadow: 0 12px 25px rgba(99, 102, 241, 0.35);
}
button:not(.primary) {
  background: rgba(15, 23, 42, 0.7);
  color: var(--text);
  border: 1px solid rgba(148, 163, 184, 0.2);
}
button:hover { transform: translateY(-1px); }
button:active { transform: translateY(0); }
.results-section { margin-bottom: 1.3rem; }
.results-section h3 { margin-bottom: 0.4rem; }
.hint { margin: 0; color: var(--muted); font-size: 0.9rem; }
pre {
  background: rgba(15, 23, 42, 0.7);
  border-radius: 12px;
  padding: 0.9rem;
  overflow-x: auto;
  border: 1px solid rgba(148, 163, 184, 0.15);
  font-size: 0.95rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.hidden { display: none; }
#error-card { border: 1px solid rgba(248, 113, 113, 0.5); }
#error-output { color: var(--error); }
@media (max-width: 640px) {
  body { padding: 2rem 0.8rem 3rem; }
  .card { padding: 1.4rem; }
}
  </style>
</head>
<body>
  <main class=\"app-shell\">
    <header class=\"app-header\">
      <div class=\"badge\">DevSkits916</div>
      <h1>RNG Analyzer</h1>
      <p class=\"subtitle\">Detect or generate classic RNG families, from congruential to cryptographic and chaotic systems.</p>
    </header>

    <form method=\"post\" class=\"card\">
      <h2>Input</h2>
      <label for=\"sequence\">Observed sequence or seed material (comma, space, or newline separated; hex like 0x123 allowed)</label>
      <textarea id=\"sequence\" name=\"sequence\" rows=\"6\">{{ sequence }}</textarea>
      <section class=\"options\">
        <h2>Options</h2>
        <div class=\"options-grid\">
          <label>Forward predictions / count
            <input type=\"number\" name=\"forward\" min=\"0\" value=\"{{ forward }}\" />
          </label>
          <label>Backward predictions
            <input type=\"number\" name=\"backward\" min=\"0\" value=\"{{ backward }}\" />
          </label>
          <label>Algorithm mode
            <select name=\"mode\">
              <option value=\"auto\" {% if mode == 'auto' %}selected{% endif %}>Auto Detect / Infer</option>
              {% for category, items in algorithm_groups %}
                <optgroup label=\"{{ category }}\">
                  {% for entry in items %}
                    <option value=\"{{ entry.key }}\" {% if mode == entry.key %}selected{% endif %}>{{ entry.label }}</option>
                  {% endfor %}
                </optgroup>
              {% endfor %}
            </select>
          </label>
        </div>
        <div class=\"button-row\">
          <button type=\"submit\" class=\"primary\">Analyze / Predict</button>
          <button type=\"submit\" name=\"load_sample\" value=\"1\">Load Sample Configuration</button>
        </div>
      </section>
    </form>

    <section class=\"card\" id=\"results-card\">
      <h2>Results</h2>
      {% if result.algorithm %}<p class=\"hint\">Active algorithm: {{ result.algorithm }}</p>{% endif %}
      <div class=\"results-section\">
        <h3>Parameters</h3>
        <pre id=\"parameters-output\">{{ result.parameters }}</pre>
      </div>
      <div class=\"results-section\">
        <h3>Forward predictions</h3>
        <p class=\"hint\">Predicted or generated states after your last provided number.</p>
        <pre id=\"forward-output\">{{ result.forward }}</pre>
      </div>
      <div class=\"results-section\">
        <h3>Backward predictions</h3>
        <p class=\"hint\">States before your first provided number (where supported).</p>
        <pre id=\"backward-output\">{{ result.backward }}</pre>
      </div>
      <div class=\"results-section\">
        <h3>Generated sequence</h3>
        <pre id=\"generated-output\">{{ result.generated }}</pre>
      </div>
    </section>

    <section class=\"card {% if not result.error %}hidden{% endif %}\" id=\"error-card\">
      <h2>Error</h2>
      <pre id=\"error-output\">{{ result.error or '' }}</pre>
    </section>
  </main>
</body>
</html>
  """
)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
