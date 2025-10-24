from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from flask import Flask, request

app = Flask(__name__)


@dataclass
class Result:
  parameters: str = '—'
  forward: str = '—'
  backward: str = '—'
  generated: str = '—'
  error: Optional[str] = None


SAMPLE_DATA = {
  'lcg': {
    'label': 'LCG demo',
    'sequence': [
      1250496027, 1116302264, 1000676753, 1668674806, 908095735,
      71666532, 896336333, 1736731266, 1314989459, 1535244752,
    ],
    'forward': 5,
    'backward': 5,
  },
  'additive': {
    'label': 'Additive demo',
    'sequence': [1000, 1037, 1074, 1111, 1148, 1185, 1222, 1259, 1296, 1333],
    'forward': 5,
    'backward': 5,
  },
  'geometric': {
    'label': 'Geometric demo',
    'sequence': [
      243,
      729,
      2187,
      6561,
      19683,
      59049,
      177147,
      531441,
      1594323,
      4782969,
    ],
    'forward': 5,
    'backward': 5,
  },
  'secondOrder': {
    'label': 'Second-order linear demo',
    'sequence': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
    'forward': 5,
    'backward': 5,
  },
  'xorshift': {'label': 'Xorshift32 demo', 'seed': 0x12345678, 'count': 10},
  'mt19937': {'label': 'MT19937 demo', 'seed': 5489, 'count': 10},
}


def gcd(a: int, b: int) -> int:
  a = abs(a)
  b = abs(b)
  while b:
    a, b = b, a % b
  return a


def egcd(a: int, b: int) -> Tuple[int, int, int]:
  if b == 0:
    return a, 1, 0
  g, x1, y1 = egcd(b, a % b)
  x = y1
  y = x1 - (a // b) * y1
  return g, x, y


def mod_inverse(a: int, m: int) -> Optional[int]:
  a = (a % m + m) % m
  g, x, _ = egcd(a, m)
  if g != 1:
    return None
  return (x % m + m) % m


def parse_sequence(raw: str) -> List[int]:
  if not raw:
    return []
  clean = raw.replace('\n', ' ').replace(',', ' ')
  tokens = [token.strip() for token in clean.split()]
  result = []
  for token in tokens:
    if not token:
      continue
    base = 16 if token.lower().startswith('0x') else 10
    try:
      value = int(token, base=base)
    except ValueError:
      continue
    result.append(value)
  return result


def infer_modulus(seq: List[int]) -> int:
  values = []
  for i in range(len(seq) - 3):
    x0, x1, x2, x3 = seq[i : i + 4]
    t = (x3 - x2) * (x1 - x0) - (x2 - x1) * (x2 - x1)
    if t:
      values.append(abs(t))
  if not values:
    raise ValueError('Unable to infer modulus: no non-zero determinant differences.')
  modulus = values[0]
  for value in values[1:]:
    modulus = gcd(modulus, value)
  return modulus


def infer_lcg(seq: List[int]) -> Tuple[int, int, int]:
  if len(seq) < 4:
    raise ValueError('Need at least four values to infer an LCG.')
  modulus = infer_modulus(seq)
  if modulus <= 0:
    raise ValueError('Inferred modulus is not positive.')
  multiplier = increment = None
  for i in range(len(seq) - 2):
    x0, x1, x2 = seq[i : i + 3]
    delta = (x1 - x0) % modulus
    if delta == 0:
      continue
    inv = mod_inverse(delta, modulus)
    if inv is None:
      continue
    multiplier = ((x2 - x1) % modulus) * inv % modulus
    increment = (x1 - multiplier * x0) % modulus
    break
  if multiplier is None or increment is None:
    raise ValueError('Unable to determine multiplier and increment.')
  for i in range(len(seq) - 1):
    predicted = (multiplier * seq[i] + increment) % modulus
    if predicted != seq[i + 1] % modulus:
      raise ValueError('Inferred parameters do not reproduce the provided sequence.')
  return multiplier, increment, modulus


def predict_lcg(seq: List[int], params: Tuple[int, int, int], forward: int, backward: int) -> Tuple[List[int], List[int]]:
  a, c, m = params
  current = seq[-1]
  forward_values = []
  for _ in range(forward):
    current = (a * current + c) % m
    forward_values.append(current)
  backward_values = []
  if backward:
    inv_a = mod_inverse(a, m)
    if inv_a is None:
      raise ValueError('Cannot reverse this generator: multiplier has no modular inverse.')
    current = seq[0]
    for _ in range(backward):
      current = (inv_a * ((current - c) % m)) % m
      backward_values.append(current)
  return forward_values, backward_values


def infer_additive(seq: List[int]) -> int:
  if len(seq) < 2:
    raise ValueError('Need at least two values to infer additive step.')
  step = seq[1] - seq[0]
  for i in range(1, len(seq) - 1):
    if seq[i + 1] - seq[i] != step:
      raise ValueError('Sequence is not consistent with a single additive step.')
  return step


def predict_additive(seq: List[int], step: int, forward: int, backward: int) -> Tuple[List[int], List[int]]:
  current = seq[-1]
  forward_values = []
  for _ in range(forward):
    current += step
    forward_values.append(current)
  current = seq[0]
  backward_values = []
  for _ in range(backward):
    current -= step
    backward_values.append(current)
  return forward_values, backward_values


def infer_geometric(seq: List[int]) -> int:
  if len(seq) < 2:
    raise ValueError('Need at least two values to infer geometric ratio.')
  ratio: Optional[int] = None
  for i in range(len(seq) - 1):
    current, nxt = seq[i], seq[i + 1]
    if current == 0:
      if nxt != 0:
        raise ValueError('Encountered zero followed by non-zero value, cannot infer ratio.')
      continue
    if nxt % current != 0:
      raise ValueError('Successive values are not integer multiples, not a geometric progression.')
    candidate = nxt // current
    if ratio is None:
      ratio = candidate
    elif candidate != ratio:
      raise ValueError('Sequence does not share a constant ratio.')
  if ratio is None:
    ratio = 0
  for i in range(len(seq) - 1):
    if seq[i] * ratio != seq[i + 1]:
      raise ValueError('Ratio does not reproduce the provided sequence.')
  return ratio


def predict_geometric(seq: List[int], ratio: int, forward: int, backward: int) -> Tuple[List[int], List[int]]:
  current = seq[-1]
  forward_values = []
  for _ in range(forward):
    current *= ratio
    forward_values.append(current)
  backward_values: List[int] = []
  if backward:
    if ratio == 0:
      raise ValueError('Cannot reverse a geometric progression with ratio 0.')
    current = seq[0]
    for _ in range(backward):
      if current % ratio != 0:
        raise ValueError('Cannot reverse geometric progression: values are not divisible by ratio.')
      current //= ratio
      backward_values.append(current)
  return forward_values, backward_values


def infer_second_order(seq: List[int]) -> Tuple[int, int]:
  if len(seq) < 4:
    raise ValueError('Need at least four values to infer second-order recurrence.')
  p: Optional[int] = None
  q: Optional[int] = None
  for i in range(len(seq) - 3):
    s0, s1, s2, s3 = seq[i : i + 4]
    det = s1 * s1 - s0 * s2
    if det == 0:
      continue
    numerator_p = s2 * s1 - s0 * s3
    if numerator_p % det != 0:
      continue
    candidate_p = numerator_p // det
    numerator_q = s1 * s3 - s2 * s2
    if numerator_q % det != 0:
      continue
    candidate_q = numerator_q // det
    p = candidate_p
    q = candidate_q
    break
  if p is None or q is None:
    raise ValueError('Unable to infer a stable second-order recurrence.')
  if q == 0:
    raise ValueError('Second-order recurrence has q = 0, treat as geometric instead.')
  for i in range(len(seq) - 2):
    expected = p * seq[i + 1] + q * seq[i]
    if expected != seq[i + 2]:
      raise ValueError('Inferred coefficients do not reproduce the provided sequence.')
  return p, q


def predict_second_order(
  seq: List[int], params: Tuple[int, int], forward: int, backward: int
) -> Tuple[List[int], List[int]]:
  p, q = params
  prev, current = seq[-2], seq[-1]
  forward_values: List[int] = []
  for _ in range(forward):
    nxt = p * current + q * prev
    forward_values.append(nxt)
    prev, current = current, nxt
  backward_values: List[int] = []
  if backward:
    after, base = seq[1], seq[0]
    for _ in range(backward):
      numerator = after - p * base
      if numerator % q != 0:
        raise ValueError('Cannot reverse second-order recurrence: coefficients do not divide cleanly.')
      previous = numerator // q
      backward_values.append(previous)
      after, base = base, previous
  return forward_values, backward_values


def xorshift32(seed: int, count: int) -> List[int]:
  state = seed & 0xFFFFFFFF
  output = []
  for _ in range(count):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= (state >> 17) & 0xFFFFFFFF
    state ^= (state << 5) & 0xFFFFFFFF
    state &= 0xFFFFFFFF
    output.append(state)
  return output


class MT19937:
  def __init__(self, seed: int) -> None:
    self.index = 624
    self.mt = [0] * 624
    self.mt[0] = seed & 0xFFFFFFFF
    for i in range(1, 624):
      prev = self.mt[i - 1]
      self.mt[i] = (0x6C078965 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF

  def extract_number(self) -> int:
    if self.index >= 624:
      self.twist()
    y = self.mt[self.index]
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    self.index += 1
    return y & 0xFFFFFFFF

  def twist(self) -> None:
    upper_mask = 0x80000000
    lower_mask = 0x7FFFFFFF
    for i in range(624):
      y = (self.mt[i] & upper_mask) + (self.mt[(i + 1) % 624] & lower_mask)
      self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
      if y % 2:
        self.mt[i] ^= 0x9908B0DF
    self.index = 0


def mt19937_sequence(seed: int, count: int) -> List[int]:
  mt = MT19937(seed & 0xFFFFFFFF)
  return [mt.extract_number() for _ in range(count)]


class TemplateWrapper:
  def __init__(self, text: str) -> None:
    self.text = text

  def render(self, **context):
    from jinja2 import Template

    template = Template(self.text)
    return template.render(**context)


def render_page(form_data: dict, result: Result) -> str:
  return TEMPLATE.render(
    sequence=form_data.get('sequence', ''),
    forward=form_data.get('forward', 5),
    backward=form_data.get('backward', 5),
    mode=form_data.get('mode', 'auto'),
    result=result,
  )


@app.route('/', methods=['GET', 'POST'])
def index():
  form_data = {
    'sequence': ', '.join(map(str, SAMPLE_DATA['lcg']['sequence'])),
    'forward': SAMPLE_DATA['lcg']['forward'],
    'backward': SAMPLE_DATA['lcg']['backward'],
    'mode': 'auto',
  }
  result = Result()

  if request.method == 'POST':
    form_data['sequence'] = request.form.get('sequence', '')
    form_data['forward'] = request.form.get('forward', form_data['forward'])
    form_data['backward'] = request.form.get('backward', form_data['backward'])
    form_data['mode'] = request.form.get('mode', form_data['mode'])

    if request.form.get('load_sample'):
      mode = form_data['mode']
      if mode in {'lcg', 'auto'}:
        sample = SAMPLE_DATA['lcg']
        form_data['sequence'] = ', '.join(map(str, sample['sequence']))
        form_data['forward'] = sample['forward']
        form_data['backward'] = sample['backward']
      elif mode == 'additive':
        sample = SAMPLE_DATA['additive']
        form_data['sequence'] = ', '.join(map(str, sample['sequence']))
        form_data['forward'] = sample['forward']
        form_data['backward'] = sample['backward']
      elif mode == 'geometric':
        sample = SAMPLE_DATA['geometric']
        form_data['sequence'] = ', '.join(map(str, sample['sequence']))
        form_data['forward'] = sample['forward']
        form_data['backward'] = sample['backward']
      elif mode == 'secondOrder':
        sample = SAMPLE_DATA['secondOrder']
        form_data['sequence'] = ', '.join(map(str, sample['sequence']))
        form_data['forward'] = sample['forward']
        form_data['backward'] = sample['backward']
      else:
        sample = SAMPLE_DATA[mode]
        form_data['sequence'] = (
          f"0x{sample['seed']:x}" if mode == 'xorshift' else str(sample['seed'])
        )
        form_data['forward'] = sample['count']
        form_data['backward'] = 0
      return render_page(form_data, result)

    try:
      forward = max(0, int(form_data['forward'] or 0))
      backward = max(0, int(form_data['backward'] or 0))
    except ValueError:
      result.error = 'Forward and backward counts must be integers.'
      return render_page(form_data, result)

    mode = form_data['mode']
    sequence = parse_sequence(form_data['sequence'])

    try:
      if mode in {'xorshift', 'mt19937'}:
        seed = sequence[0] if sequence else SAMPLE_DATA[mode]['seed']
        count = forward or SAMPLE_DATA[mode]['count']
        generated = (
          xorshift32(seed, count)
          if mode == 'xorshift'
          else mt19937_sequence(seed, count)
        )
        result.parameters = f'Seed: {seed & 0xFFFFFFFF}'
        result.generated = ', '.join(str(n) for n in generated)
        result.forward = '—'
        result.backward = '—'
      else:
        if not sequence:
          raise ValueError('Please provide at least one number.')
        if mode == 'lcg':
          params = infer_lcg(sequence)
          forward_vals, backward_vals = predict_lcg(sequence, params, forward, backward)
          result.parameters = f'a = {params[0]}, c = {params[1]}, m = {params[2]}'
        elif mode == 'additive':
          step = infer_additive(sequence)
          forward_vals, backward_vals = predict_additive(sequence, step, forward, backward)
          result.parameters = f'k = {step}'
        elif mode == 'geometric':
          ratio = infer_geometric(sequence)
          forward_vals, backward_vals = predict_geometric(sequence, ratio, forward, backward)
          result.parameters = f'r = {ratio}'
        elif mode == 'secondOrder':
          params = infer_second_order(sequence)
          forward_vals, backward_vals = predict_second_order(sequence, params, forward, backward)
          result.parameters = f'p = {params[0]}, q = {params[1]}'
        else:
          notes = []
          try:
            params = infer_lcg(sequence)
            forward_vals, backward_vals = predict_lcg(sequence, params, forward, backward)
            result.parameters = f'Detected LCG: a = {params[0]}, c = {params[1]}, m = {params[2]}'
          except ValueError as first_error:
            notes.append(f'LCG: {first_error}')
            try:
              step = infer_additive(sequence)
              forward_vals, backward_vals = predict_additive(sequence, step, forward, backward)
              suffix = f" (notes: {' | '.join(notes)})" if notes else ''
              result.parameters = f'Detected additive sequence: k = {step}{suffix}'
            except ValueError as second_error:
              notes.append(f'Additive: {second_error}')
              try:
                ratio = infer_geometric(sequence)
                forward_vals, backward_vals = predict_geometric(sequence, ratio, forward, backward)
                suffix = f" (notes: {' | '.join(notes)})" if notes else ''
                result.parameters = f'Detected geometric progression: r = {ratio}{suffix}'
              except ValueError as third_error:
                notes.append(f'Geometric: {third_error}')
                params = infer_second_order(sequence)
                forward_vals, backward_vals = predict_second_order(sequence, params, forward, backward)
                suffix = f" (notes: {' | '.join(notes)})" if notes else ''
                result.parameters = (
                  f'Detected second-order recurrence: p = {params[0]}, q = {params[1]}{suffix}'
                )
        result.forward = ', '.join(str(n) for n in forward_vals) if forward_vals else '—'
        result.backward = ', '.join(str(n) for n in backward_vals) if backward_vals else '—'
    except Exception as exc:  # noqa: BLE001
      result.error = str(exc)

  return render_page(form_data, result)


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
  background: rgba(15, 23, 42, 0.75);
  color: var(--text);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  padding: 0.75rem;
  font-size: 1rem;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
textarea:focus, input:focus, select:focus {
  outline: none;
  border-color: rgba(99, 102, 241, 0.6);
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
}
textarea { min-height: 120px; resize: vertical; }
.options-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  margin-bottom: 1.5rem;
}
.button-row { display: flex; flex-wrap: wrap; gap: 0.8rem; }
button {
  border: none;
  border-radius: 12px;
  padding: 0.9rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease;
  background: rgba(148, 163, 184, 0.15);
  color: var(--text);
}
button.primary {
  background: linear-gradient(135deg, var(--accent-start), var(--accent-end));
  box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
}
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.45);
}
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
      <p class=\"subtitle\">Infer simple RNG parameters and explore sample generators.</p>
    </header>

    <form method=\"post\" class=\"card\">
      <h2>Input</h2>
      <label for=\"sequence\">Observed sequence (comma, space, or newline separated; hex like 0x123 allowed)</label>
      <textarea id=\"sequence\" name=\"sequence\" rows=\"6\">{{ sequence }}</textarea>
      <section class=\"options\">
        <h2>Options</h2>
        <div class=\"options-grid\">
          <label>Forward predictions
            <input type=\"number\" name=\"forward\" min=\"0\" value=\"{{ forward }}\" />
          </label>
          <label>Backward predictions
            <input type=\"number\" name=\"backward\" min=\"0\" value=\"{{ backward }}\" />
          </label>
          <label>Algorithm mode
            <select name=\"mode\">
              <option value=\"auto\" {% if mode == 'auto' %}selected{% endif %}>Auto Detect / Infer</option>
              <option value=\"lcg\" {% if mode == 'lcg' %}selected{% endif %}>LCG (manual analyze)</option>
              <option value=\"additive\" {% if mode == 'additive' %}selected{% endif %}>Additive (manual analyze)</option>
              <option value=\"geometric\" {% if mode == 'geometric' %}selected{% endif %}>Geometric progression (manual analyze)</option>
              <option value=\"secondOrder\" {% if mode == 'secondOrder' %}selected{% endif %}>Second-order linear recurrence (manual)</option>
              <option value=\"xorshift\" {% if mode == 'xorshift' %}selected{% endif %}>Xorshift demo</option>
              <option value=\"mt19937\" {% if mode == 'mt19937' %}selected{% endif %}>MT19937 demo</option>
            </select>
          </label>
        </div>
        <div class=\"button-row\">
          <button type=\"submit\" class=\"primary\">Analyze / Predict</button>
          <button type=\"submit\" name=\"load_sample\" value=\"1\">Load Sample Sequence for This Algorithm</button>
        </div>
      </section>
    </form>

    <section class=\"card\" id=\"results-card\">
      <h2>Results</h2>
      <div class=\"results-section\">
        <h3>Parameters</h3>
        <pre id=\"parameters-output\">{{ result.parameters }}</pre>
      </div>
      <div class=\"results-section\">
        <h3>Forward predictions</h3>
        <p class=\"hint\">Predicted states after your last provided number.</p>
        <pre id=\"forward-output\">{{ result.forward }}</pre>
      </div>
      <div class=\"results-section\">
        <h3>Backward predictions</h3>
        <p class=\"hint\">States before your first provided number (starting with immediately previous).</p>
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
