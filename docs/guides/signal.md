# Signal Processing Guide

Endgame provides 45 signal processing transforms covering filtering, spectral
analysis, wavelet decomposition, entropy/complexity measures, and spatial
filtering for EEG and BCI applications. Every transform follows the sklearn
interface (`fit`, `transform`) and composes naturally into `Pipeline` objects.

**Import convention:** `import endgame as eg`

---

## Filtering

Five filter classes cover the most common preprocessing needs for physiological,
sensor, and audio signals.

| Class | Use case |
|---|---|
| `ButterworthFilter` | IIR bandpass / lowpass / highpass |
| `FIRFilter` | Linear-phase FIR with window design |
| `SavgolFilter` | Savitzky-Golay smoothing (polynomial) |
| `NotchFilter` | Narrow-band powerline rejection |
| `FilterBank` | Apply multiple filters in parallel |

```python
from endgame.signal import ButterworthFilter, FilterBank, NotchFilter

# Bandpass EEG signal to the alpha band (8-13 Hz), sampled at 256 Hz
bp = ButterworthFilter(btype='bandpass', low=8.0, high=13.0, fs=256, order=4)
bp.fit(X_train)            # learns nothing; validates parameters
X_filtered = bp.transform(X_train)   # shape: (n_samples, n_channels, n_times)

# Remove 50 Hz powerline interference before bandpassing
notch = NotchFilter(freq=50.0, fs=256, quality=30)
X_clean = notch.fit_transform(X_train)

# Apply a bank of filters and concatenate features
bank = FilterBank(
    filters=[
        ButterworthFilter(btype='bandpass', low=4,  high=8,  fs=256),   # theta
        ButterworthFilter(btype='bandpass', low=8,  high=13, fs=256),   # alpha
        ButterworthFilter(btype='bandpass', low=13, high=30, fs=256),   # beta
    ]
)
X_bank = bank.fit_transform(X_train)  # shape: (n_samples, n_filters * n_channels, n_times)
```

Use inside a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from endgame.signal import ButterworthFilter, BandPowerExtractor
from endgame.models import LGBMWrapper

pipe = Pipeline([
    ('filter', ButterworthFilter(btype='bandpass', low=1.0, high=40.0, fs=256)),
    ('features', BandPowerExtractor(fs=256, bands='standard')),
    ('clf', LGBMWrapper()),
])
pipe.fit(X_train, y_train)
```

---

## Spectral Analysis

Spectral transforms extract frequency-domain features from raw signals.

```python
from endgame.signal import FFTTransformer, WelchPSD, MultitaperPSD, BandPowerExtractor

# Fast Fourier Transform — returns magnitude spectrum per channel
fft = FFTTransformer(n_fft=256, log_scale=True)
X_spec = fft.fit_transform(X)    # shape: (n_samples, n_channels, n_fft // 2 + 1)

# Welch power spectral density estimate (overlapping windows, averaged)
welch = WelchPSD(fs=256, nperseg=128, noverlap=64)
freqs, psd = welch.fit_transform(X)   # psd shape: (n_samples, n_channels, n_freqs)

# Multitaper PSD — lower variance than Welch via DPSS tapers
mt = MultitaperPSD(fs=256, bandwidth=2.0, n_tapers=None)  # n_tapers auto
mt_psd = mt.fit_transform(X)

# Extract scalar band-power features ready for a classifier
extractor = BandPowerExtractor(
    fs=256,
    bands={
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 45),
    },
    method='welch',   # 'fft' | 'welch' | 'multitaper'
    log_power=True,
)
X_bp = extractor.fit_transform(X)  # shape: (n_samples, n_channels * n_bands)
```

---

## Wavelet Decomposition

Three wavelet transforms cover continuous, discrete, and packet decompositions.

```python
from endgame.signal import CWTTransformer, DWTTransformer, WaveletPacketTransformer

# Continuous Wavelet Transform — time-frequency scalogram
cwt = CWTTransformer(wavelet='morl', scales=range(1, 64))
X_cwt = cwt.fit_transform(X)   # shape: (n_samples, n_channels, n_scales, n_times)

# Discrete Wavelet Transform — multi-resolution decomposition
dwt = DWTTransformer(wavelet='db4', level=5, feature='energy')
X_dwt = dwt.fit_transform(X)   # shape: (n_samples, n_channels * (level + 1))

# Wavelet Packet Transform — full binary tree of subbands
wpt = WaveletPacketTransformer(wavelet='sym5', level=3, feature='energy')
X_wpt = wpt.fit_transform(X)
```

---

## Entropy and Complexity Measures

Entropy and fractal-dimension features characterise signal regularity and are
widely used in EEG, HRV, and time-series classification.

### Entropy

```python
from endgame.signal import (
    PermutationEntropy,
    SampleEntropy,
    ApproximateEntropy,
)

# Permutation entropy — fast, robust to noise
pe = PermutationEntropy(order=3, delay=1, normalize=True)
X_pe = pe.fit_transform(X)   # shape: (n_samples, n_channels)

# Sample entropy — regularity of template matches
se = SampleEntropy(m=2, r=0.2)   # r is fraction of std
X_se = se.fit_transform(X)

# Approximate entropy — faster approximation to SampEn
ae = ApproximateEntropy(m=2, r=0.2)
X_ae = ae.fit_transform(X)
```

### Complexity

```python
from endgame.signal import (
    HiguchiFD,
    HurstExponent,
    DFA,
    LempelZivComplexity,
)

# Higuchi fractal dimension
hfd = HiguchiFD(k_max=10)
X_hfd = hfd.fit_transform(X)

# Hurst exponent — long-range dependence
hurst = HurstExponent(method='rs')  # 'rs' | 'dfa'
X_h = hurst.fit_transform(X)

# Detrended Fluctuation Analysis
dfa = DFA(scales=None, order=1)   # scales auto-selected
X_dfa = dfa.fit_transform(X)

# Lempel-Ziv complexity — sequence compression ratio
lz = LempelZivComplexity(normalize=True)
X_lz = lz.fit_transform(X)
```

---

## Spatial Filtering (EEG / BCI)

Spatial filters reduce channel dimensionality and maximise class discriminability
for multi-channel data.

```python
from endgame.signal import CSP, TangentSpace, FilterBankCSP

# Common Spatial Patterns — maximises variance ratio between two classes
csp = CSP(n_components=4, reg=None, log=True)
csp.fit(X_train, y_train)   # X shape: (n_trials, n_channels, n_times)
X_csp = csp.transform(X_test)  # shape: (n_trials, n_components)

# Tangent Space projection — Riemannian geometry on SPD matrices
ts = TangentSpace(metric='riemann')
ts.fit(X_train)
X_ts = ts.transform(X_test)

# Filter Bank CSP — apply CSP per frequency band then concatenate
fbcsp = FilterBankCSP(
    fs=256,
    bands={'alpha': (8, 13), 'beta': (13, 30)},
    n_components=4,
)
fbcsp.fit(X_train, y_train)
X_fb = fbcsp.transform(X_test)
```

---

## Full EEG Pipeline Example

```python
from sklearn.pipeline import Pipeline
from endgame.signal import (
    NotchFilter, FilterBank, ButterworthFilter,
    BandPowerExtractor, CSP,
)
from endgame.models import LGBMWrapper

pipe = Pipeline([
    ('notch',    NotchFilter(freq=50.0, fs=256)),
    ('bank',     FilterBank([
                     ButterworthFilter(btype='bandpass', low=8,  high=13, fs=256),
                     ButterworthFilter(btype='bandpass', low=13, high=30, fs=256),
                 ])),
    ('csp',      CSP(n_components=6)),
    ('clf',      LGBMWrapper(preset='endgame')),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

---

## API Reference

Full parameter documentation for every class listed above is available in the
auto-generated API reference at `docs/api/signal.rst` or by calling
`help(ClassName)` at the Python prompt.
