# RNG Analyzer
https://devskits916.github.io/Pequito-sploit/
RNG Analyzer is an educational tool that helps you study simple random number generators (RNGs). Paste an observed integer sequence and the app will try to infer the algorithm that produced it, then predict both future and past values. The project supports two deployment modes so you can host a purely static version or a full-stack Flask deployment.

## Features
- Reverse-engineer linear congruential generators (LCG) from at least four observed values.
- Detect simple additive sequences and extrapolate forward/backward.
- Identify integer geometric progressions and extrapolate in both directions when ratios divide cleanly.
- Infer second-order linear recurrences (e.g., Fibonacci-like sequences) with integer coefficients.
- Demonstrate xorshift32 and MT19937 generators from seeds so you can compare styles of RNG output.
- Predict forward and backward states for supported algorithms.
- Built-in sample sequences for every algorithm so you can explore without providing data.
- Auto-detection pipeline that tries LCG, additive, geometric, then second-order recurrences with helpful error notes.

## Project Structure
```
.
├── README.md
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── server/
    ├── requirements.txt
    └── sequence_predictor_web.py
```

## Deployment Modes

### Static Front-end (GitHub Pages)
1. Serve the contents of `frontend/` as-is. No build step or backend is required.
2. Open `frontend/index.html` directly in a browser, or publish the folder to GitHub Pages.
3. All inference and prediction logic runs in the browser using plain JavaScript.

### Full-stack Flask App (Render)
1. Install dependencies:
   ```bash
   cd server
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python3 sequence_predictor_web.py
   ```
3. Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the interface.
4. Deploy to Render by pointing a Python web service at `server/sequence_predictor_web.py`. The app binds to `0.0.0.0:5000` by default.

Both interfaces share the same layout and logic, giving users a consistent experience whether running locally, on GitHub Pages, or on Render.
