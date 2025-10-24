"""Collection of random number generator implementations and detectors."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


Number = Union[int, float]


def _ensure_int(value: Fraction) -> int:
  if value.denominator != 1:
    raise ValueError('Inferred parameter is not integral; unable to detect algorithm.')
  return int(value.numerator)


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


def parse_seed(sequence: List[Number], fallback: Iterable[int]) -> List[int]:
  if sequence:
    return [int(value) for value in sequence]
  return list(fallback)


def _format_prediction(values: Iterable[Number]) -> List[Number]:
  return [value for value in values]


@dataclass
class AnalysisResult:
  parameters: str
  forward: List[Number]
  backward: List[Number]
  generated: List[Number]


@dataclass
class AlgorithmHandler:
  key: str
  label: str
  category: str
  supports_detection: bool
  sample: Dict[str, Union[int, float, List[Number], Tuple[int, ...]]]
  analyzer: Optional[Callable[[List[Number], int, int], AnalysisResult]] = None
  generator: Optional[Callable[[List[Number], int], AnalysisResult]] = None
  description: str = ''


def _lcg_predict(seq: List[int], params: Tuple[int, int, int], forward: int, backward: int) -> Tuple[List[int], List[int]]:
  a, c, m = params
  current = seq[-1]
  forward_values: List[int] = []
  for _ in range(forward):
    current = (a * current + c) % m
    forward_values.append(current)
  backward_values: List[int] = []
  if backward:
    inv_a = mod_inverse(a, m)
    if inv_a is None:
      raise ValueError('Cannot reverse this generator: multiplier has no modular inverse.')
    current = seq[0]
    for _ in range(backward):
      current = (inv_a * ((current - c) % m)) % m
      backward_values.append(current)
  return forward_values, backward_values


def infer_lcg(seq: List[int]) -> Tuple[int, int, int]:
  if len(seq) < 4:
    raise ValueError('Need at least four values to infer an LCG.')
  modulus_values: List[int] = []
  for i in range(len(seq) - 3):
    x0, x1, x2, x3 = seq[i : i + 4]
    t = (x3 - x2) * (x1 - x0) - (x2 - x1) * (x2 - x1)
    if t:
      modulus_values.append(abs(t))
  if not modulus_values:
    raise ValueError('Unable to infer modulus: no non-zero determinant differences.')
  modulus = modulus_values[0]
  for value in modulus_values[1:]:
    modulus = gcd(modulus, value)
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


def analyze_lcg(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  int_sequence = [int(value) for value in sequence]
  params = infer_lcg(int_sequence)
  forward_vals, backward_vals = _lcg_predict(int_sequence, params, forward, backward)
  return AnalysisResult(
    parameters=f'a = {params[0]}, c = {params[1]}, m = {params[2]}',
    forward=_format_prediction(forward_vals),
    backward=_format_prediction(backward_vals),
    generated=[],
  )


def infer_additive(seq: List[int]) -> int:
  if len(seq) < 2:
    raise ValueError('Need at least two values to infer additive step.')
  step = seq[1] - seq[0]
  for i in range(1, len(seq) - 1):
    if seq[i + 1] - seq[i] != step:
      raise ValueError('Sequence is not consistent with a single additive step.')
  return step


def analyze_additive(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  int_sequence = [int(value) for value in sequence]
  step = infer_additive(int_sequence)
  current = int_sequence[-1]
  forward_vals = []
  for _ in range(forward):
    current += step
    forward_vals.append(current)
  current = int_sequence[0]
  backward_vals = []
  for _ in range(backward):
    current -= step
    backward_vals.append(current)
  return AnalysisResult(
    parameters=f'k = {step}',
    forward=_format_prediction(forward_vals),
    backward=_format_prediction(backward_vals),
    generated=[],
  )


def _solve_linear_3x3(matrix: List[List[int]], rhs: List[int]) -> Tuple[Fraction, Fraction, Fraction]:
  det = (
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
  )
  if det == 0:
    raise ValueError('Unable to solve system for quadratic congruential generator.')

  def det_replace(col: int) -> int:
    replaced = [row[:] for row in matrix]
    for i in range(3):
      replaced[i][col] = rhs[i]
    return (
      replaced[0][0] * (replaced[1][1] * replaced[2][2] - replaced[1][2] * replaced[2][1])
      - replaced[0][1] * (replaced[1][0] * replaced[2][2] - replaced[1][2] * replaced[2][0])
      + replaced[0][2] * (replaced[1][0] * replaced[2][1] - replaced[1][1] * replaced[2][0])
    )

  det_a = det_replace(0)
  det_b = det_replace(1)
  det_c = det_replace(2)
  return Fraction(det_a, det), Fraction(det_b, det), Fraction(det_c, det)


def infer_qcg(sequence: List[int]) -> Tuple[int, int, int, int]:
  if len(sequence) < 4:
    raise ValueError('Need at least four values to infer a quadratic generator.')
  matrix = []
  rhs = []
  for i in range(3):
    matrix.append([sequence[i] ** 2, sequence[i], 1])
    rhs.append(sequence[i + 1])
  a_frac, b_frac, c_frac = _solve_linear_3x3(matrix, rhs)
  a = _ensure_int(a_frac)
  b = _ensure_int(b_frac)
  c = _ensure_int(c_frac)
  diffs: List[int] = []
  for idx in range(len(sequence) - 1):
    expected = a * sequence[idx] ** 2 + b * sequence[idx] + c
    diffs.append(expected - sequence[idx + 1])
  modulus = 0
  for diff in diffs:
    modulus = gcd(modulus, diff)
  modulus = abs(modulus)
  if modulus == 0:
    modulus = max(sequence) + 1
  for idx in range(len(sequence) - 1):
    expected = (a * sequence[idx] ** 2 + b * sequence[idx] + c) % modulus
    if expected != sequence[idx + 1] % modulus:
      raise ValueError('Provided numbers are not produced by a quadratic generator with stable parameters.')
  return a, b, c, modulus


def analyze_qcg(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  int_sequence = [int(value) for value in sequence]
  a, b, c, modulus = infer_qcg(int_sequence)
  current = int_sequence[-1]
  forward_vals: List[int] = []
  for _ in range(forward):
    current = (a * current ** 2 + b * current + c) % modulus
    forward_vals.append(current)
  # Quadratic generators are not trivially invertible without expensive factorisation.
  backward_vals: List[int] = []
  return AnalysisResult(
    parameters=f'a = {a}, b = {b}, c = {c}, m = {modulus}',
    forward=_format_prediction(forward_vals),
    backward=_format_prediction(backward_vals),
    generated=[],
  )


def _xorshift32_step(state: int) -> int:
  state &= 0xFFFFFFFF
  state ^= (state << 13) & 0xFFFFFFFF
  state ^= (state >> 17) & 0xFFFFFFFF
  state ^= (state << 5) & 0xFFFFFFFF
  return state & 0xFFFFFFFF


def _reverse_xorshift32(state: int) -> int:
  state &= 0xFFFFFFFF
  # Reverse shift << 5
  value = state
  for shift, mask in ((5, 0xFFFFFFFF),):
    for _ in range(5):
      value ^= (value << shift) & mask
  # Reverse shift >> 17
  value ^= value >> 17
  value ^= value >> 34
  # Reverse shift << 13
  for _ in range(5):
    value ^= (value << 13) & 0xFFFFFFFF
  return value & 0xFFFFFFFF


def analyze_xorshift32(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  if len(sequence) < 2:
    raise ValueError('Need at least two values to verify XORShift32.')
  states = [int(value) & 0xFFFFFFFF for value in sequence]
  for idx in range(len(states) - 1):
    if _xorshift32_step(states[idx]) != states[idx + 1]:
      raise ValueError('Provided numbers do not follow XORShift32 transition.')
  current = states[-1]
  forward_vals = []
  for _ in range(forward):
    current = _xorshift32_step(current)
    forward_vals.append(current)
  backward_vals = []
  current = states[0]
  for _ in range(backward):
    current = _reverse_xorshift32(current)
    backward_vals.append(current)
  return AnalysisResult(
    parameters='XORShift32 detected (a=13,17,5)',
    forward=_format_prediction(forward_vals),
    backward=_format_prediction(backward_vals),
    generated=[],
  )


def analyze_blum_blum_shub(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  if len(sequence) < 2:
    raise ValueError('Need at least two values to detect Blum–Blum–Shub.')
  states = [int(value) for value in sequence]
  modulus = 0
  for idx in range(len(states) - 1):
    diff = states[idx + 1] - states[idx] ** 2
    modulus = gcd(modulus, diff)
  modulus = abs(modulus)
  if modulus == 0:
    raise ValueError('Unable to infer modulus for Blum–Blum–Shub.')
  for idx in range(len(states) - 1):
    expected = pow(states[idx], 2, modulus)
    if expected != states[idx + 1] % modulus:
      raise ValueError('Sequence is not consistent with Blum–Blum–Shub recurrence.')
  current = states[-1]
  forward_vals = []
  for _ in range(forward):
    current = pow(current, 2, modulus)
    forward_vals.append(current)
  return AnalysisResult(
    parameters=f'M = {modulus}',
    forward=_format_prediction(forward_vals),
    backward=[],
    generated=[],
  )


def analyze_lagged_fibonacci(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  states = [int(value) for value in sequence]
  if len(states) < 6:
    raise ValueError('Need at least six values to infer lagged Fibonacci parameters.')
  max_lag = min(12, len(states) - 1)
  best_params: Optional[Tuple[int, int, int]] = None
  for k in range(2, max_lag + 1):
    for j in range(1, k):
      diffs: List[int] = []
      for idx in range(k, len(states)):
        expected = states[idx - j] + states[idx - k]
        diffs.append(states[idx] - expected)
      modulus = 0
      for diff in diffs:
        modulus = gcd(modulus, diff)
      modulus = abs(modulus)
      if modulus == 0:
        if all(states[idx] == states[idx - j] + states[idx - k] for idx in range(k, len(states))):
          best_params = (j, k, 0)
          break
        continue
      if all((states[idx - j] + states[idx - k]) % modulus == states[idx] % modulus for idx in range(k, len(states))):
        best_params = (j, k, modulus)
        break
    if best_params:
      break
  if not best_params:
    raise ValueError('Unable to infer lagged Fibonacci parameters.')
  j, k, modulus = best_params
  working = states[:]
  forward_vals = []
  for _ in range(forward):
    idx = len(working)
    value = working[idx - j] + working[idx - k]
    if modulus:
      value %= modulus
    working.append(value)
    forward_vals.append(value)
  return AnalysisResult(
    parameters=(
      f'Lagged Fibonacci (additive) with j={j}, k={k}'
      + (f', m={modulus}' if modulus else ', no modulus wrap detected')
    ),
    forward=_format_prediction(forward_vals),
    backward=[],
    generated=[],
  )


def analyze_logistic_map(sequence: List[Number], forward: int, backward: int) -> AnalysisResult:
  if len(sequence) < 3:
    raise ValueError('Need at least three values to detect logistic map.')
  values = [float(x) for x in sequence]
  ratios: List[float] = []
  for idx in range(len(values) - 1):
    denom = values[idx] * (1.0 - values[idx])
    if abs(denom) < 1e-12:
      raise ValueError('Encountered zero denominator when estimating logistic map parameter.')
    ratios.append(values[idx + 1] / denom)
  r = sum(ratios) / len(ratios)
  for idx in range(len(values) - 1):
    expected = r * values[idx] * (1.0 - values[idx])
    if abs(expected - values[idx + 1]) > 1e-6:
      raise ValueError('Sequence is not consistent with a logistic map.')
  working = values[:]
  forward_vals: List[float] = []
  for _ in range(forward):
    nxt = r * working[-1] * (1.0 - working[-1])
    forward_vals.append(nxt)
    working.append(nxt)
  backward_vals: List[float] = []
  current = values[0]
  for _ in range(backward):
    discriminant = 0.25 - current / r
    if discriminant < 0:
      break
    prev = 0.5 + sqrt(discriminant)
    backward_vals.append(prev)
    current = prev
  return AnalysisResult(
    parameters=f'r = {r:.8f}',
    forward=_format_prediction(forward_vals),
    backward=_format_prediction(backward_vals),
    generated=[],
  )


def generate_lcg(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [])
  if len(params) < 4:
    # Seed, multiplier, increment, modulus
    seed, a, c, m = 1, 1103515245, 12345, 2 ** 31
  else:
    seed, a, c, m = params[:4]
  values = []
  current = seed % m
  for _ in range(forward):
    current = (a * current + c) % m
    values.append(current)
  return AnalysisResult(
    parameters=f'seed={seed}, a={a}, c={c}, m={m}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_qcg(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [])
  if len(params) < 5:
    seed, a, b, c, m = 1, 1, 1, 1, 2 ** 31 - 1
  else:
    seed, a, b, c, m = params[:5]
  values = []
  current = seed % m
  for _ in range(forward):
    current = (a * current ** 2 + b * current + c) % m
    values.append(current)
  return AnalysisResult(
    parameters=f'seed={seed}, a={a}, b={b}, c={c}, m={m}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_multiply_with_carry(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [])
  if len(params) < 3:
    seed, carry, multiplier = 123456789, 987654321, 4294957665
  else:
    seed, carry, multiplier = params[:3]
  modulus = 2 ** 32
  values = []
  state = seed % modulus
  carry = carry % modulus
  for _ in range(forward):
    total = multiplier * state + carry
    state = total % modulus
    carry = total // modulus
    values.append(state)
  return AnalysisResult(
    parameters=f'seed={seed}, carry={carry}, a={multiplier}, m=2^32',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_xorshift32(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [0x12345678])
  state = params[0] & 0xFFFFFFFF
  values = []
  for _ in range(forward):
    state = _xorshift32_step(state)
    values.append(state)
  return AnalysisResult(
    parameters=f'seed={params[0] & 0xFFFFFFFF}',
    forward=values,
    backward=[],
    generated=values,
  )


def _xorshift128_step(state: List[int]) -> int:
  x, y, z, w = state
  t = x ^ (x << 11) & 0xFFFFFFFF
  x, y, z = y, z, w
  w = (w ^ (w >> 19) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
  state[:] = [x, y, z, w]
  return w


def generate_xorshift128(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [123456789, 362436069, 521288629, 88675123])
  state = [value & 0xFFFFFFFF for value in params[:4]]
  values = []
  for _ in range(forward):
    values.append(_xorshift128_step(state))
  return AnalysisResult(
    parameters=f'state={[hex(v) for v in state]}',
    forward=values,
    backward=[],
    generated=values,
  )


def _xorwow_step(state: List[int]) -> int:
  x, y, z, w, v, d = state
  t = x ^ (x >> 2)
  x, y, z, w, v = y, z, w, v, v ^ (v << 4) ^ (t ^ (t << 1))
  result = (v + w + d) & 0xFFFFFFFF
  state[:] = [x, y, z, w, v, (d + 362437) & 0xFFFFFFFF]
  return result


def generate_xorwow(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [123456789, 362436069, 521288629, 88675123, 5783321, 6615241])
  state = [value & 0xFFFFFFFF for value in params[:6]]
  values = []
  for _ in range(forward):
    values.append(_xorwow_step(state))
  return AnalysisResult(
    parameters=f'state={[hex(v) for v in state]}',
    forward=values,
    backward=[],
    generated=values,
  )


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


def generate_mt(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [5489])
  mt = MT19937(params[0])
  values = [mt.extract_number() for _ in range(forward)]
  return AnalysisResult(
    parameters=f'seed={params[0]}',
    forward=values,
    backward=[],
    generated=values,
  )


def _well_generator(state: List[int], mat_a: int, m1: int, m2: int, m3: int) -> int:
  z0 = state[-1]
  z1 = state[m1] ^ (state[m1] << 16) ^ state[m2] ^ (state[m2] << 15)
  z2 = state[m3] ^ (state[m3] >> 11)
  new_v = z1 ^ z2
  state.append(new_v)
  state.pop(0)
  state[-1] ^= -((new_v & 1)) & mat_a
  return state[-1] & 0xFFFFFFFF


def _make_well(seed_list: List[int], size: int) -> List[int]:
  if len(seed_list) < size:
    seed_list = (seed_list * (size // len(seed_list) + 1))[:size]
  return [value & 0xFFFFFFFF for value in seed_list[:size]]


def generate_well(sequence: List[Number], forward: int, variant: str) -> AnalysisResult:
  params = parse_seed(sequence, [0x12345 + i for i in range(16)])
  if variant == 'well512':
    state = _make_well(params, 16)
    m1, m2, m3, mat_a = 13, 9, 5, 0xDA442D24
  elif variant == 'well1024':
    state = _make_well(params, 32)
    m1, m2, m3, mat_a = 3, 24, 10, 0x8EBFD028
  else:
    state = _make_well(params, 624)
    m1, m2, m3, mat_a = 70, 179, 449, 0xE46E1700
  values = []
  for _ in range(forward):
    values.append(_well_generator(state, mat_a, -m1, -m2, -m3))
  return AnalysisResult(
    parameters=f'WELL variant={variant}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_blum_blum_shub(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [119 * 127, 3])
  if len(params) == 1:
    modulus = params[0]
    state = 3
  else:
    modulus, state = params[:2]
  values = []
  current = state % modulus
  for _ in range(forward):
    current = pow(current, 2, modulus)
    values.append(current)
  return AnalysisResult(
    parameters=f'M={modulus}, seed={state}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_lagged_fibonacci(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [0, 1, 1, 2, 3, 5, 8, 13])
  modulus = 2 ** 32
  j, k = 5, 8
  state = [int(value) % modulus for value in params[:k]]
  values = []
  for _ in range(forward):
    value = (state[-j] + state[-k]) % modulus
    state.append(value)
    state.pop(0)
    values.append(value)
  return AnalysisResult(
    parameters=f'j={j}, k={k}, modulus=2^32',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_middle_square(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [675248])
  seed = params[0]
  width = len(str(seed))
  values = []
  current = seed
  for _ in range(forward):
    squared = current * current
    padded = f'{squared:0{2 * width}d}'
    start = (len(padded) - width) // 2
    current = int(padded[start : start + width])
    values.append(current)
  return AnalysisResult(
    parameters=f'seed={seed}, width={width}',
    forward=values,
    backward=[],
    generated=values,
  )


def _aes_ctr_stream(key: bytes, iv: bytes, count: int) -> List[int]:
  try:
    from Crypto.Cipher import AES
  except Exception as exc:  # pragma: no cover - dependency optional at runtime
    raise RuntimeError('pycryptodome is required for AES-CTR generation.') from exc
  cipher = AES.new(key, AES.MODE_CTR, nonce=b'', initial_value=int.from_bytes(iv, 'big'))
  stream = cipher.encrypt(b'\x00' * (count * 4))
  result = []
  for i in range(count):
    chunk = stream[4 * i : 4 * (i + 1)]
    result.append(int.from_bytes(chunk, 'big'))
  return result


def generate_aes_ctr(sequence: List[Number], forward: int) -> AnalysisResult:
  seed_bytes = bytes(int(x) & 0xFF for x in sequence)
  if len(seed_bytes) < 16:
    seed_bytes = (seed_bytes + b'RNGAnalyzerAESKey!!')[:16]
  key = seed_bytes[:16]
  iv = (b'initialvectorCTR' + key)[:16]
  values = _aes_ctr_stream(key, iv, max(1, forward))
  return AnalysisResult(
    parameters='AES-CTR DRBG',
    forward=values,
    backward=[],
    generated=values,
  )


def _chacha20_block(key: bytes, nonce: bytes, counter: int) -> bytes:
  def rotate(v: int, c: int) -> int:
    return ((v << c) & 0xFFFFFFFF) | ((v & 0xFFFFFFFF) >> (32 - c))

  constants = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
  key_words = [int.from_bytes(key[i : i + 4], 'little') for i in range(0, 32, 4)]
  nonce_words = [int.from_bytes(nonce[i : i + 4], 'little') for i in range(0, 12, 4)]
  state = constants + key_words + [counter] + nonce_words

  def quarter_round(a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = rotate(state[d], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = rotate(state[b], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = rotate(state[d], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = rotate(state[b], 7)

  working = state[:]
  for _ in range(10):
    quarter_round(0, 4, 8, 12)
    quarter_round(1, 5, 9, 13)
    quarter_round(2, 6, 10, 14)
    quarter_round(3, 7, 11, 15)
    quarter_round(0, 5, 10, 15)
    quarter_round(1, 6, 11, 12)
    quarter_round(2, 7, 8, 13)
    quarter_round(3, 4, 9, 14)

  result = []
  for original, transformed in zip(state, working):
    value = (original + transformed) & 0xFFFFFFFF
    result.append(value)
  return b''.join(value.to_bytes(4, 'little') for value in result)


def generate_chacha20(sequence: List[Number], forward: int) -> AnalysisResult:
  seed_bytes = bytes(int(x) & 0xFF for x in sequence)
  if len(seed_bytes) < 32:
    seed_bytes = (seed_bytes + b'ChaCha20RNGAnalyzerSeed........')[:32]
  key = seed_bytes[:32]
  nonce = b'RNGChaCha20'[:12]
  values = []
  for counter in range(max(1, forward)):
    block = _chacha20_block(key, nonce, counter)
    values.append(int.from_bytes(block[:8], 'little'))
  return AnalysisResult(
    parameters='ChaCha20 (reduced stream sample)',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_sha_drbg(sequence: List[Number], forward: int) -> AnalysisResult:
  import hashlib

  seed_bytes = bytes(int(x) & 0xFF for x in sequence) or b'SHADRGBSeed'
  key = b'RNGAnalyzerKey'
  v = b'RNGAnalyzerV  '
  values = []
  for _ in range(max(1, forward)):
    h = hashlib.sha256(key + v + seed_bytes).digest()
    values.append(int.from_bytes(h[:8], 'big'))
    v = hashlib.sha256(v + seed_bytes).digest()
  return AnalysisResult(
    parameters='SHA256-based DRBG (simplified)',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_fortuna(sequence: List[Number], forward: int) -> AnalysisResult:
  import hashlib

  pools = [hashlib.sha256(bytes([i])).digest() for i in range(32)]
  seed = bytes(int(x) & 0xFF for x in sequence)
  key = hashlib.sha256(seed or b'FortunaSeed').digest()
  values = []
  counter = 0
  for _ in range(max(1, forward)):
    digest = hashlib.sha256(key + pools[counter % len(pools)] + counter.to_bytes(4, 'big')).digest()
    values.append(int.from_bytes(digest[:8], 'big'))
    counter += 1
  return AnalysisResult(
    parameters='Fortuna-inspired generator (simplified)',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_pcg32(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [42, 54])
  state = params[0] & 0xFFFFFFFFFFFFFFFF
  inc = ((params[1] & 0xFFFFFFFFFFFFFFFF) << 1) | 1
  multiplier = 6364136223846793005
  values = []
  for _ in range(forward):
    state = (state * multiplier + inc) & 0xFFFFFFFFFFFFFFFF
    xorshifted = ((state >> 18) ^ state) >> 27
    rot = state >> 59
    value = (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
    values.append(value & 0xFFFFFFFF)
  return AnalysisResult(
    parameters=f'state={params[0]}, increment={params[1]}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_splitmix64(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [0x123456789ABCDEF0])
  state = params[0] & 0xFFFFFFFFFFFFFFFF
  values = []
  for _ in range(forward):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    values.append(z)
  return AnalysisResult(
    parameters=f'seed={params[0]}',
    forward=values,
    backward=[],
    generated=values,
  )


def _rotl(x: int, k: int) -> int:
  return ((x << k) & 0xFFFFFFFFFFFFFFFF) | (x >> (64 - k))


def generate_xoroshiro128plus(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(sequence, [0x0123456789ABCDEF, 0xF0E1D2C3B4A59687])
  s0, s1 = [value & 0xFFFFFFFFFFFFFFFF for value in params[:2]]
  values = []
  for _ in range(forward):
    result = (s0 + s1) & 0xFFFFFFFFFFFFFFFF
    s1 ^= s0
    s0 = _rotl(s0, 55) ^ s1 ^ (s1 << 14 & 0xFFFFFFFFFFFFFFFF)
    s1 = _rotl(s1, 36)
    values.append(result)
  return AnalysisResult(
    parameters=f'state0={params[0]}, state1={params[1]}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_xoshiro256(sequence: List[Number], forward: int) -> AnalysisResult:
  params = parse_seed(
    sequence,
    [
      0x0123456789ABCDEF,
      0xF0E1D2C3B4A59687,
      0x0F1E2D3C4B5A6978,
      0x89ABCDEF01234567,
    ],
  )
  s = [value & 0xFFFFFFFFFFFFFFFF for value in params[:4]]
  values = []
  for _ in range(forward):
    result = (s[0] + s[3]) & 0xFFFFFFFFFFFFFFFF
    t = (s[1] << 17) & 0xFFFFFFFFFFFFFFFF
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    values.append(result)
  return AnalysisResult(
    parameters=f'state={[hex(v) for v in s]}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_halton(sequence: List[Number], forward: int) -> AnalysisResult:
  bases = parse_seed(sequence, [2, 3])
  count = max(1, forward)

  def halton_single(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    while index > 0:
      f /= base
      r += f * (index % base)
      index //= base
    return r

  values = []
  for idx in range(1, count + 1):
    point = [halton_single(idx, base) for base in bases[:2]]
    values.append(point)
  flattened = [coord for point in values for coord in point]
  return AnalysisResult(
    parameters=f'Halton sequence bases={bases[:2]}',
    forward=flattened,
    backward=[],
    generated=flattened,
  )


def generate_tent_map(sequence: List[Number], forward: int) -> AnalysisResult:
  params = sequence or [0.123, 1.999]
  state = float(params[0])
  r = float(params[1])
  values = []
  for _ in range(max(1, forward)):
    state = r * min(state, 1 - state)
    values.append(state)
  return AnalysisResult(
    parameters=f'seed={params[0]}, r={r}',
    forward=values,
    backward=[],
    generated=values,
  )


def generate_henon_map(sequence: List[Number], forward: int) -> AnalysisResult:
  params = sequence or [0.1, 0.3, 1.4, 0.3]
  x, y = float(params[0]), float(params[1])
  a = float(params[2])
  b = float(params[3])
  values = []
  for _ in range(max(1, forward)):
    x_next = 1 - a * x * x + y
    y = b * x
    x = x_next
    values.append((x, y))
  flattened = [coord for pair in values for coord in pair]
  return AnalysisResult(
    parameters=f'x0={params[0]}, y0={params[1]}, a={a}, b={b}',
    forward=flattened,
    backward=[],
    generated=flattened,
  )


def generate_logistic(sequence: List[Number], forward: int) -> AnalysisResult:
  params = sequence or [0.5, 3.9]
  x = float(params[0])
  r = float(params[1])
  values = []
  for _ in range(max(1, forward)):
    x = r * x * (1.0 - x)
    values.append(x)
  return AnalysisResult(
    parameters=f'seed={params[0]}, r={r}',
    forward=values,
    backward=[],
    generated=values,
  )


def _sobol_1d(index: int) -> float:
  result = 0.0
  factor = 0.5
  while index:
    if index & 1:
      result += factor
    index >>= 1
    factor *= 0.5
  return result


def generate_sobol(sequence: List[Number], forward: int) -> AnalysisResult:
  count = max(1, forward)
  values = [_sobol_1d(i) for i in range(1, count + 1)]
  return AnalysisResult(
    parameters='Sobol (1D, base polynomial x + 1)',
    forward=values,
    backward=[],
    generated=values,
  )


def _niederreiter_base2(index: int) -> float:
  result = 0.0
  factor = 0.5
  while index:
    if index & 1:
      result += factor
    index >>= 1
    factor *= 0.5
  return result


def generate_niederreiter(sequence: List[Number], forward: int) -> AnalysisResult:
  count = max(1, forward)
  values = [_niederreiter_base2(i ^ (i >> 1)) for i in range(1, count + 1)]
  return AnalysisResult(
    parameters='Niederreiter (base-2, Gray code)',
    forward=values,
    backward=[],
    generated=values,
  )


ALGORITHMS: List[AlgorithmHandler] = [
  AlgorithmHandler(
    key='lcg',
    label='Linear Congruential Generator',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_lcg,
    generator=generate_lcg,
    sample={
      'sequence': [
        1250496027,
        1116302264,
        1000676753,
        1668674806,
        908095735,
        915748896,
      ],
      'forward': 5,
      'backward': 5,
    },
  ),
  AlgorithmHandler(
    key='additive',
    label='Additive progression',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_additive,
    generator=None,
    sample={'sequence': [1000, 1037, 1074, 1111, 1148], 'forward': 5, 'backward': 5},
  ),
  AlgorithmHandler(
    key='qcg',
    label='Quadratic Congruential Generator',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_qcg,
    generator=generate_qcg,
    sample={
      'sequence': [7, 124, 31129, 1938122674, 7512639004760188579],
      'forward': 3,
      'backward': 0,
    },
  ),
  AlgorithmHandler(
    key='xorshift32',
    label='XORShift32',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_xorshift32,
    generator=generate_xorshift32,
    sample={
      'sequence': [2274908837, 358294691, 1210119364, 2176035992],
      'forward': 5,
      'backward': 5,
    },
  ),
  AlgorithmHandler(
    key='bbs',
    label='Blum–Blum–Shub',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_blum_blum_shub,
    generator=generate_blum_blum_shub,
    sample={'sequence': [3, 9, 81, 53361, 284100513], 'forward': 5, 'backward': 0},
  ),
  AlgorithmHandler(
    key='lagged-fibonacci',
    label='Lagged Fibonacci (additive)',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_lagged_fibonacci,
    generator=generate_lagged_fibonacci,
    sample={'sequence': [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], 'forward': 5, 'backward': 0},
  ),
  AlgorithmHandler(
    key='logistic-map',
    label='Logistic map',
    category='Detectable generators',
    supports_detection=True,
    analyzer=analyze_logistic_map,
    generator=generate_logistic,
    sample={
      'sequence': [0.4, 0.9359999999999999, 0.2336256000000002, 0.6982742481960964],
      'forward': 5,
      'backward': 2,
    },
  ),
  AlgorithmHandler(
    key='mwc',
    label='Multiply-With-Carry',
    category='Manual generators',
    supports_detection=False,
    generator=generate_multiply_with_carry,
    sample={'seed': 123456789, 'carry': 362436, 'multiplier': 36969, 'count': 10},
  ),
  AlgorithmHandler(
    key='xorshift128',
    label='XORShift128',
    category='Manual generators',
    supports_detection=False,
    generator=generate_xorshift128,
    sample={'seed': [123456789, 362436069, 521288629, 88675123], 'count': 10},
  ),
  AlgorithmHandler(
    key='xorwow',
    label='XORWOW',
    category='Manual generators',
    supports_detection=False,
    generator=generate_xorwow,
    sample={'seed': [123456789, 362436069, 521288629, 88675123, 5783321, 6615241], 'count': 10},
  ),
  AlgorithmHandler(
    key='mt19937',
    label='Mersenne Twister (MT19937)',
    category='Manual generators',
    supports_detection=False,
    generator=generate_mt,
    sample={'seed': 5489, 'count': 10},
  ),
  AlgorithmHandler(
    key='well512',
    label='WELL512',
    category='Manual generators',
    supports_detection=False,
    generator=lambda seq, n: generate_well(seq, n, 'well512'),
    sample={'seed': [0x12345 + i for i in range(16)], 'count': 10},
  ),
  AlgorithmHandler(
    key='well1024',
    label='WELL1024',
    category='Manual generators',
    supports_detection=False,
    generator=lambda seq, n: generate_well(seq, n, 'well1024'),
    sample={'seed': [0x23456 + i for i in range(32)], 'count': 10},
  ),
  AlgorithmHandler(
    key='well19937',
    label='WELL19937',
    category='Manual generators',
    supports_detection=False,
    generator=lambda seq, n: generate_well(seq, n, 'well19937'),
    sample={'seed': [0x34567 + i for i in range(624)], 'count': 10},
  ),
  AlgorithmHandler(
    key='middle-square',
    label='Middle square',
    category='Manual generators',
    supports_detection=False,
    generator=generate_middle_square,
    sample={'seed': 675248, 'count': 8},
  ),
  AlgorithmHandler(
    key='aes-ctr',
    label='AES-CTR DRBG',
    category='Cryptographic generators',
    supports_detection=False,
    generator=generate_aes_ctr,
    sample={'seed': [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF], 'count': 8},
  ),
  AlgorithmHandler(
    key='chacha20',
    label='ChaCha20',
    category='Cryptographic generators',
    supports_detection=False,
    generator=generate_chacha20,
    sample={'seed': [i for i in range(16)], 'count': 8},
  ),
  AlgorithmHandler(
    key='sha-drbg',
    label='SHA-DRBG',
    category='Cryptographic generators',
    supports_detection=False,
    generator=generate_sha_drbg,
    sample={'seed': [0xAA, 0xBB, 0xCC], 'count': 8},
  ),
  AlgorithmHandler(
    key='fortuna',
    label='Fortuna',
    category='Cryptographic generators',
    supports_detection=False,
    generator=generate_fortuna,
    sample={'seed': [0x12, 0x34, 0x56, 0x78], 'count': 8},
  ),
  AlgorithmHandler(
    key='pcg32',
    label='PCG32',
    category='Manual generators',
    supports_detection=False,
    generator=generate_pcg32,
    sample={'seed': [42, 54], 'count': 10},
  ),
  AlgorithmHandler(
    key='splitmix64',
    label='SplitMix64',
    category='Manual generators',
    supports_detection=False,
    generator=generate_splitmix64,
    sample={'seed': [0x123456789ABCDEF0], 'count': 10},
  ),
  AlgorithmHandler(
    key='xoroshiro128plus',
    label='Xoroshiro128+',
    category='Manual generators',
    supports_detection=False,
    generator=generate_xoroshiro128plus,
    sample={'seed': [0x0123456789ABCDEF, 0xF0E1D2C3B4A59687], 'count': 10},
  ),
  AlgorithmHandler(
    key='xoshiro256',
    label='Xoshiro256**',
    category='Manual generators',
    supports_detection=False,
    generator=generate_xoshiro256,
    sample={'seed': [0x0123456789ABCDEF, 0xF0E1D2C3B4A59687, 0x0F1E2D3C4B5A6978, 0x89ABCDEF01234567], 'count': 10},
  ),
  AlgorithmHandler(
    key='halton',
    label='Halton sequence',
    category='Quasi-random sequences',
    supports_detection=False,
    generator=generate_halton,
    sample={'seed': [2, 3], 'count': 8},
  ),
  AlgorithmHandler(
    key='sobol',
    label='Sobol sequence',
    category='Quasi-random sequences',
    supports_detection=False,
    generator=generate_sobol,
    sample={'count': 8},
  ),
  AlgorithmHandler(
    key='niederreiter',
    label='Niederreiter sequence',
    category='Quasi-random sequences',
    supports_detection=False,
    generator=generate_niederreiter,
    sample={'count': 8},
  ),
  AlgorithmHandler(
    key='tent-map',
    label='Tent map',
    category='Chaotic maps',
    supports_detection=False,
    generator=generate_tent_map,
    sample={'seed': [0.123, 1.999], 'count': 10},
  ),
  AlgorithmHandler(
    key='henon-map',
    label='Henon map',
    category='Chaotic maps',
    supports_detection=False,
    generator=generate_henon_map,
    sample={'seed': [0.1, 0.3, 1.4, 0.3], 'count': 10},
  ),
]


ALGORITHM_MAP: Dict[str, AlgorithmHandler] = {algo.key: algo for algo in ALGORITHMS}
DETECTABLE_ALGORITHMS: List[AlgorithmHandler] = [
  algo for algo in ALGORITHMS if algo.supports_detection and algo.analyzer is not None
]

