const SAMPLE_DATA = {
  lcg: {
    label: 'LCG demo',
    sequence: [
      1250496027, 1116302264, 1000676753, 1668674806, 908095735,
      71666532, 896336333, 1736731266, 1314989459, 1535244752
    ],
    forward: 5,
    backward: 5
  },
  additive: {
    label: 'Additive demo',
    sequence: [1000, 1037, 1074, 1111, 1148, 1185, 1222, 1259, 1296, 1333],
    forward: 5,
    backward: 5
  },
  geometric: {
    label: 'Geometric demo',
    sequence: [5, 15, 45, 135, 405, 1215, 3645, 10935, 32805, 98415],
    forward: 5,
    backward: 5
  },
  secondOrder: {
    label: 'Second-order linear demo',
    sequence: [2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
    forward: 5,
    backward: 5
  },
  xorshift: {
    label: 'Xorshift32 demo',
    seed: 0x12345678,
    count: 10
  },
  mt19937: {
    label: 'MT19937 demo',
    seed: 5489,
    count: 10
  }
};

const sequenceInput = document.getElementById('sequence');
const forwardInput = document.getElementById('forward-count');
const backwardInput = document.getElementById('backward-count');
const algorithmSelect = document.getElementById('algorithm-mode');
const analyzeBtn = document.getElementById('analyze-btn');
const loadSampleBtn = document.getElementById('load-sample-btn');

const parametersOutput = document.getElementById('parameters-output');
const forwardOutput = document.getElementById('forward-output');
const backwardOutput = document.getElementById('backward-output');
const generatedOutput = document.getElementById('generated-output');
const errorCard = document.getElementById('error-card');
const errorOutput = document.getElementById('error-output');

function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function egcd(a, b) {
  if (b === 0) return { g: a, x: 1, y: 0 };
  const { g, x, y } = egcd(b, a % b);
  return { g, x: y, y: x - Math.floor(a / b) * y };
}

function modInverse(a, m) {
  a = ((a % m) + m) % m;
  const { g, x } = egcd(a, m);
  if (g !== 1) return null;
  return ((x % m) + m) % m;
}

function parseSequence(input) {
  if (!input) return [];
  const tokens = input
    .split(/[^0-9xa-fA-F+-]+/)
    .map((token) => token.trim())
    .filter(Boolean);
  return tokens.map((token) => {
    if (/^[-+]?0x[0-9a-fA-F]+$/.test(token)) {
      return Number.parseInt(token, 16);
    }
    return Number.parseInt(token, 10);
  }).filter((num) => Number.isFinite(num));
}

function inferModulus(seq) {
  const values = [];
  for (let i = 0; i < seq.length - 3; i++) {
    const x0 = seq[i];
    const x1 = seq[i + 1];
    const x2 = seq[i + 2];
    const x3 = seq[i + 3];
    const t = (x3 - x2) * (x1 - x0) - (x2 - x1) * (x2 - x1);
    if (t !== 0) {
      values.push(Math.abs(t));
    }
  }
  if (!values.length) {
    throw new Error('Unable to infer modulus: no suitable determinant differences.');
  }
  return values.reduce((acc, val) => gcd(acc, val));
}

function inferLCG(seq) {
  if (seq.length < 4) {
    throw new Error('Need at least four values to infer an LCG.');
  }
  let m = inferModulus(seq);
  if (!m || m <= 0) {
    throw new Error('Failed to infer a valid modulus.');
  }
  let a = null;
  let c = null;
  for (let i = 0; i < seq.length - 2; i++) {
    const x0 = seq[i];
    const x1 = seq[i + 1];
    const x2 = seq[i + 2];
    let delta = ((x1 - x0) % m + m) % m;
    if (delta === 0) continue;
    const inv = modInverse(delta, m);
    if (inv === null) continue;
    const candidateA = ((x2 - x1) % m + m) % m;
    a = (candidateA * inv) % m;
    c = ((x1 - (a * x0) % m) % m + m) % m;
    break;
  }
  if (a === null || c === null) {
    throw new Error('Unable to determine multiplier and increment.');
  }
  for (let i = 0; i < seq.length - 1; i++) {
    const next = (a * seq[i] + c) % m;
    if (((next - seq[i + 1]) % m + m) % m !== 0) {
      throw new Error('Inferred parameters do not reproduce the provided sequence.');
    }
  }
  return { a, c, m };
}

function predictLCG(seq, params, forwardCount, backwardCount) {
  const { a, c, m } = params;
  const forward = [];
  let current = seq[seq.length - 1];
  for (let i = 0; i < forwardCount; i++) {
    current = (a * current + c) % m;
    forward.push(current);
  }

  const backward = [];
  if (backwardCount > 0) {
    const invA = modInverse(a, m);
    if (invA === null) {
      throw new Error('Cannot reverse this generator because multiplier has no modular inverse.');
    }
    current = seq[0];
    for (let i = 0; i < backwardCount; i++) {
      current = (invA * ((current - c) % m + m)) % m;
      backward.push(current);
    }
  }
  return { forward, backward };
}

function inferAdditive(seq) {
  if (seq.length < 2) {
    throw new Error('Need at least two values to infer additive step.');
  }
  const diff = seq[1] - seq[0];
  for (let i = 1; i < seq.length - 1; i++) {
    if (seq[i + 1] - seq[i] !== diff) {
      throw new Error('Sequence is not consistent with a constant difference.');
    }
  }
  return { k: diff };
}

function predictAdditive(seq, params, forwardCount, backwardCount) {
  const { k } = params;
  const forward = [];
  let current = seq[seq.length - 1];
  for (let i = 0; i < forwardCount; i++) {
    current += k;
    forward.push(current);
  }

  const backward = [];
  current = seq[0];
  for (let i = 0; i < backwardCount; i++) {
    current -= k;
    backward.push(current);
  }
  return { forward, backward };
}

function inferGeometric(seq) {
  if (seq.length < 2) {
    throw new Error('Need at least two values to infer geometric ratio.');
  }
  let ratio = null;
  for (let i = 0; i < seq.length - 1; i++) {
    const current = seq[i];
    const next = seq[i + 1];
    if (current === 0) {
      if (next !== 0) {
        throw new Error('Encountered zero followed by non-zero value, cannot infer ratio.');
      }
      continue;
    }
    if (next % current !== 0) {
      throw new Error('Successive values are not integer multiples, not a geometric progression.');
    }
    const candidate = next / current;
    if (ratio === null) {
      ratio = candidate;
    } else if (candidate !== ratio) {
      throw new Error('Sequence does not share a constant ratio.');
    }
  }
  if (ratio === null) {
    ratio = 0;
  }
  for (let i = 0; i < seq.length - 1; i++) {
    const expected = seq[i] * ratio;
    if (expected !== seq[i + 1]) {
      throw new Error('Ratio does not reproduce the provided sequence.');
    }
  }
  return { r: ratio };
}

function predictGeometric(seq, params, forwardCount, backwardCount) {
  const { r } = params;
  const forward = [];
  let current = seq[seq.length - 1];
  for (let i = 0; i < forwardCount; i++) {
    current *= r;
    forward.push(current);
  }

  const backward = [];
  if (backwardCount > 0) {
    if (r === 0) {
      throw new Error('Cannot reverse a geometric progression with ratio 0.');
    }
    current = seq[0];
    for (let i = 0; i < backwardCount; i++) {
      if (current % r !== 0) {
        throw new Error('Cannot reverse geometric progression: values are not divisible by ratio.');
      }
      current /= r;
      backward.push(current);
    }
  }
  return { forward, backward };
}

function inferSecondOrder(seq) {
  if (seq.length < 4) {
    throw new Error('Need at least four values to infer second-order recurrence.');
  }
  let p = null;
  let q = null;
  for (let i = 0; i < seq.length - 3; i++) {
    const s0 = seq[i];
    const s1 = seq[i + 1];
    const s2 = seq[i + 2];
    const s3 = seq[i + 3];
    const det = s1 * s1 - s0 * s2;
    if (det === 0) {
      continue;
    }
    const numeratorP = s2 * s1 - s0 * s3;
    if (numeratorP % det !== 0) {
      continue;
    }
    const candidateP = numeratorP / det;
    const numeratorQ = s1 * s3 - s2 * s2;
    if (numeratorQ % det !== 0) {
      continue;
    }
    const candidateQ = numeratorQ / det;
    p = candidateP;
    q = candidateQ;
    break;
  }
  if (p === null || q === null) {
    throw new Error('Unable to infer a stable second-order recurrence.');
  }
  if (q === 0) {
    throw new Error('Second-order recurrence has zero q, treat as geometric instead.');
  }
  for (let i = 0; i < seq.length - 2; i++) {
    const expected = p * seq[i + 1] + q * seq[i];
    if (expected !== seq[i + 2]) {
      throw new Error('Inferred coefficients do not reproduce the provided sequence.');
    }
  }
  return { p, q };
}

function predictSecondOrder(seq, params, forwardCount, backwardCount) {
  const { p, q } = params;
  const forward = [];
  let prev = seq[seq.length - 2];
  let current = seq[seq.length - 1];
  for (let i = 0; i < forwardCount; i++) {
    const next = p * current + q * prev;
    forward.push(next);
    prev = current;
    current = next;
  }

  const backward = [];
  if (backwardCount > 0) {
    let after = seq[1];
    let base = seq[0];
    for (let i = 0; i < backwardCount; i++) {
      const numerator = after - p * base;
      if (numerator % q !== 0) {
        throw new Error('Cannot reverse second-order recurrence: coefficients do not divide cleanly.');
      }
      const previous = numerator / q;
      backward.push(previous);
      after = base;
      base = previous;
    }
  }
  return { forward, backward };
}

function xorshift32(seed, count) {
  let state = seed >>> 0;
  const out = [];
  for (let i = 0; i < count; i++) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state >>>= 0;
    state ^= state << 5;
    state >>>= 0;
    out.push(state >>> 0);
  }
  return out;
}

class MT19937 {
  constructor(seed) {
    this.index = 624;
    this.mt = new Array(624);
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] >>> 0;
      this.mt[i] = (0x6c078965 * (prev ^ (prev >>> 30)) + i) >>> 0;
    }
  }

  extractNumber() {
    if (this.index >= 624) {
      this.twist();
    }
    let y = this.mt[this.index];
    y ^= (y >>> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >>> 18);
    this.index++;
    return y >>> 0;
  }

  twist() {
    const upperMask = 0x80000000;
    const lowerMask = 0x7fffffff;
    for (let i = 0; i < 624; i++) {
      const y = ((this.mt[i] & upperMask) + (this.mt[(i + 1) % 624] & lowerMask)) >>> 0;
      this.mt[i] = this.mt[(i + 397) % 624] ^ (y >>> 1);
      if (y % 2 !== 0) {
        this.mt[i] ^= 0x9908b0df;
      }
    }
    this.index = 0;
  }
}

function mt19937Sequence(seed, count) {
  const mt = new MT19937(seed >>> 0);
  const result = [];
  for (let i = 0; i < count; i++) {
    result.push(mt.extractNumber());
  }
  return result;
}

function resetOutputs() {
  parametersOutput.textContent = '—';
  forwardOutput.textContent = '—';
  backwardOutput.textContent = '—';
  generatedOutput.textContent = '—';
  errorOutput.textContent = '';
  errorCard.classList.add('hidden');
}

function showError(message) {
  errorOutput.textContent = message;
  errorCard.classList.remove('hidden');
}

function displayResults({ parameters, forward, backward, generated }) {
  resetOutputs();
  if (parameters) {
    parametersOutput.textContent = parameters;
  }
  if (forward) {
    forwardOutput.textContent = forward.length ? forward.join(', ') : '—';
  }
  if (backward) {
    backwardOutput.textContent = backward.length ? backward.join(', ') : '—';
  }
  if (generated) {
    generatedOutput.textContent = generated.length ? generated.join(', ') : '—';
  }
}

function autoDetect(seq) {
  const attempts = [];
  try {
    const params = inferLCG(seq);
    return { type: 'lcg', params, notes: [] };
  } catch (error) {
    attempts.push(`LCG: ${error.message}`);
  }

  try {
    const params = inferAdditive(seq);
    return { type: 'additive', params, notes: attempts };
  } catch (error) {
    attempts.push(`Additive: ${error.message}`);
  }

  try {
    const params = inferGeometric(seq);
    return { type: 'geometric', params, notes: attempts };
  } catch (error) {
    attempts.push(`Geometric: ${error.message}`);
  }

  try {
    const params = inferSecondOrder(seq);
    return { type: 'secondOrder', params, notes: attempts };
  } catch (error) {
    attempts.push(`Second-order: ${error.message}`);
  }

  throw new Error(`No supported model matched this sequence. ${attempts.join(' | ')}`);
}

function handleAnalysis() {
  resetOutputs();
  try {
    const seq = parseSequence(sequenceInput.value);
    const forwardCount = Math.max(0, Number.parseInt(forwardInput.value, 10) || 0);
    const backwardCount = Math.max(0, Number.parseInt(backwardInput.value, 10) || 0);
    const mode = algorithmSelect.value;

    if (mode === 'xorshift' || mode === 'mt19937') {
      const seed = seq.length ? seq[0] : SAMPLE_DATA[mode].seed;
      const count = forwardCount > 0 ? forwardCount : SAMPLE_DATA[mode].count;
      if (mode === 'xorshift') {
        const generated = xorshift32(seed >>> 0, count || 10);
        displayResults({
          parameters: `Seed: ${seed >>> 0}`,
          generated
        });
      } else {
        const generated = mt19937Sequence(seed >>> 0, count || 10);
        displayResults({
          parameters: `Seed: ${seed >>> 0}`,
          generated
        });
      }
      return;
    }

    if (!seq.length) {
      throw new Error('Please provide at least one number.');
    }

    if (mode === 'lcg') {
      const params = inferLCG(seq);
      const { forward, backward } = predictLCG(seq, params, forwardCount, backwardCount);
      displayResults({
        parameters: `a = ${params.a}, c = ${params.c}, m = ${params.m}`,
        forward,
        backward
      });
      return;
    }

    if (mode === 'additive') {
      const params = inferAdditive(seq);
      const { forward, backward } = predictAdditive(seq, params, forwardCount, backwardCount);
      displayResults({
        parameters: `k = ${params.k}`,
        forward,
        backward
      });
      return;
    }

    if (mode === 'geometric') {
      const params = inferGeometric(seq);
      const { forward, backward } = predictGeometric(seq, params, forwardCount, backwardCount);
      displayResults({
        parameters: `r = ${params.r}`,
        forward,
        backward
      });
      return;
    }

    if (mode === 'secondOrder') {
      const params = inferSecondOrder(seq);
      const { forward, backward } = predictSecondOrder(seq, params, forwardCount, backwardCount);
      displayResults({
        parameters: `p = ${params.p}, q = ${params.q}`,
        forward,
        backward
      });
      return;
    }

    if (mode === 'auto') {
      const { type, params, notes } = autoDetect(seq);
      const noteText = notes && notes.length ? ` (notes: ${notes.join(' | ')})` : '';
      if (type === 'lcg') {
        const { forward, backward } = predictLCG(seq, params, forwardCount, backwardCount);
        displayResults({
          parameters: `Detected LCG: a = ${params.a}, c = ${params.c}, m = ${params.m}`,
          forward,
          backward
        });
      } else if (type === 'additive') {
        const { forward, backward } = predictAdditive(seq, params, forwardCount, backwardCount);
        displayResults({
          parameters: `Detected additive sequence: k = ${params.k}${noteText}`,
          forward,
          backward
        });
      } else if (type === 'geometric') {
        const { forward, backward } = predictGeometric(seq, params, forwardCount, backwardCount);
        displayResults({
          parameters: `Detected geometric progression: r = ${params.r}${noteText}`,
          forward,
          backward
        });
      } else {
        const { forward, backward } = predictSecondOrder(seq, params, forwardCount, backwardCount);
        displayResults({
          parameters: `Detected second-order recurrence: p = ${params.p}, q = ${params.q}${noteText}`,
          forward,
          backward
        });
      }
      return;
    }
  } catch (error) {
    showError(error.message || 'Unexpected error.');
  }
}

function loadSample() {
  resetOutputs();
  const mode = algorithmSelect.value;
  if (mode === 'lcg' || mode === 'auto') {
    const sample = SAMPLE_DATA.lcg;
    sequenceInput.value = sample.sequence.join(', ');
    forwardInput.value = sample.forward;
    backwardInput.value = sample.backward;
  } else if (mode === 'additive') {
    const sample = SAMPLE_DATA.additive;
    sequenceInput.value = sample.sequence.join(', ');
    forwardInput.value = sample.forward;
    backwardInput.value = sample.backward;
  } else if (mode === 'geometric') {
    const sample = SAMPLE_DATA.geometric;
    sequenceInput.value = sample.sequence.join(', ');
    forwardInput.value = sample.forward;
    backwardInput.value = sample.backward;
  } else if (mode === 'secondOrder') {
    const sample = SAMPLE_DATA.secondOrder;
    sequenceInput.value = sample.sequence.join(', ');
    forwardInput.value = sample.forward;
    backwardInput.value = sample.backward;
  } else if (mode === 'xorshift' || mode === 'mt19937') {
    sequenceInput.value = mode === 'xorshift'
      ? `0x${SAMPLE_DATA.xorshift.seed.toString(16)}`
      : `${SAMPLE_DATA.mt19937.seed}`;
    forwardInput.value = SAMPLE_DATA[mode].count;
    backwardInput.value = 0;
  }
}

analyzeBtn.addEventListener('click', handleAnalysis);
loadSampleBtn.addEventListener('click', loadSample);

loadSample();
