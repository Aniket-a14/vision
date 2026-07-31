# Multi-Modal Defect Prediction for Casting Production
### A complete build guide — laptop-only, ~4 weeks, CPU

---

## 0. What this system actually is

A quality-control system that predicts casting defects in submersible pump impellers by fusing **two information sources that arrive at different moments in the production cycle**:

| Source | When available | What it tells you |
|---|---|---|
| Process parameters (pour temp, injection pressure, speed, alloy chemistry, tool wear) | **Before and during** the shot | Whether this part is *likely* to come out bad — and *why* |
| Inline camera image | **After** the part is cast | Whether the part *is* bad |

The project's thesis: **fusing both beats either alone under realistic factory imaging conditions**, and the process-parameter branch is what makes the system *prescriptive* rather than merely descriptive — it can tell an operator which knob to turn.

That framing matters. If you present this as "we classified images and also had some numbers," it's a weak capstone. Presented as above, it's a genuine predictive-quality-control argument.

---

## 1. The architecture decision (and why)

**Chosen: late fusion with pre-extracted image features, single pipeline.**

```
                    ┌─────────────────────────────────────┐
  Impeller image →  │ Frozen ResNet-18 (ImageNet weights)  │ → 512-d embedding
                    │ NO training, single forward pass      │        │
                    └─────────────────────────────────────┘        │
                                                                    ▼
                                                              PCA → 64-d
                                                                    │
  Process params ─────────────────────────────────────────────┐    │
  (9 tabular features)                                        ▼    ▼
                                                        ┌──────────────────┐
                                                        │  concatenate     │
                                                        │  (73 features)   │
                                                        └────────┬─────────┘
                                                                 ▼
                                                        ┌──────────────────┐
                                                        │ XGBoost classifier│
                                                        └────────┬─────────┘
                                                                 ▼
                                              p(defect) → cost-optimal threshold → decision
                                                                 │
                                                                 ▼
                                              SHAP → root cause → parameter recommendation
```

**Why this and not end-to-end deep fusion:**

1. **CPU-feasible.** You extract image features *once* (~5–10 min for all 7,348 images), cache them to disk, and every experiment afterward trains in seconds. End-to-end CNN fine-tuning on a laptop CPU is hours per run, and you need dozens of runs.
2. **Iteration speed is the real currency in a 4-week project.** Cached features mean you can run the full ablation study 50 times in an afternoon.
3. **SHAP works cleanly on tree models.** Your root-cause analysis section — arguably the highest-value part — depends on this.
4. Your own project brief explicitly permits "or use pre-extracted features." This is the intended path.

**Optional stretch:** a small PyTorch MLP fusion head as a second fusion variant, to show you compared fusion strategies. Include only if Week 3 goes smoothly.

---

## 2. The dataset problem, and the honest solution

### 2.1 The constraint you must know about

**No public dataset contains both process parameters and images for the same physical part.** Image defect datasets (casting, MVTec AD, NEU steel) have no process data. Process datasets (SECOM, Bosch, AI4I 2020) have no images. Do not spend a week searching for one; it does not exist in accessible form.

### 2.2 Base dataset — images

**Casting Product Image Data for Quality Inspection** (Dabhi, 2020, Kaggle)
`kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product`

- 7,348 grayscale top-view images of submersible pump impellers, 300×300 px
- Pre-split: **train** 3,758 defective / 2,875 ok · **test** 453 defective / 262 ok
- Captured under controlled stable lighting
- Defect types present: blow holes, pinholes, burrs, shrinkage, mould and pouring defects

This is the right choice: real industrial parts, real defects, clean labels, small enough for CPU, and casting is a process whose physics you can defend in a viva.

### 2.3 Process parameters — physics-based simulation

You will generate process parameters using a **causal physical model of aluminium die casting**, then sample the defect label from that model, then attach a real image matching the sampled label.

**This must be declared explicitly in your report.** Write, in the methodology section:

> Process parameters are synthesised from a physically-grounded causal model of Al-Si die casting, since no public dataset provides paired process telemetry and part imagery. Labels are sampled from the physics model's defect probability and matched to real defect images from the casting dataset. The system therefore constitutes a *validated simulation testbed* for multi-modal fusion; the vision branch operates on genuine industrial imagery, while the process branch operates on simulated but physically-consistent telemetry.

Examiners respect declared limitations. They do not respect undeclared ones.

### 2.4 The single mistake that would destroy the project

**Do not generate process parameters from the label.**

If you write `if label == defective: temperature = high`, you have leaked the answer into the features. Your model will report 99.8% accuracy and it will mean absolutely nothing. This is the most common failure mode in synthetic-data projects and an examiner will find it in thirty seconds.

**Correct causal order — always:**

```
parameters  →  defect probability (physics + irreducible noise)  →  sampled label  →  matched image
```

The irreducible noise term is what caps your achievable accuracy at a realistic level. Target a process-only ROC-AUC of **0.80–0.88**. If you're getting 0.97 from parameters alone, your noise term is too small and the result is not credible — real foundries cannot predict defects that well from process data alone.

---

## 3. Environment setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn xgboost shap opencv-python matplotlib seaborn tqdm pyarrow
```

The CPU-only torch wheel is ~200 MB instead of ~2.5 GB. Use it.

**Project structure:**

```
defect-prediction/
├── data/
│   ├── raw/casting_data/{train,test}/{def_front,ok_front}/
│   ├── processed/          # parquet + cached .npy features
│   └── exports/            # CSVs for Power BI
├── src/
│   ├── config.py
│   ├── simulate_process.py
│   ├── extract_features.py
│   ├── train_models.py
│   ├── explain.py
│   └── economics.py
├── notebooks/
├── figures/
└── report/
```

---

## 4. Phase 1 — The process parameter simulator

`src/simulate_process.py`

### 4.1 Parameter definitions

Nine features, all physically meaningful for Al-Si die casting:

| Feature | Unit | Distribution | Controllable? |
|---|---|---|---|
| `pour_temp_c` | °C | N(700, 18) | Yes |
| `die_temp_c` | °C | N(220, 20) | Yes |
| `inj_pressure_bar` | bar | N(850, 70) | Yes |
| `inj_speed_ms` | m/s | N(3.0, 0.45) | Yes |
| `hold_time_s` | s | N(6.0, 1.0) | Yes |
| `cooling_time_s` | s | N(12.0, 2.2) | Yes |
| `si_content_pct` | % | N(9.5, 0.7) | No (material) |
| `fe_content_pct` | % | Gamma(4, 0.2) | No (impurity) |
| `tool_wear_shots` | count | U(0, 50000) | No (maintenance state) |

### 4.2 The causal defect model

Each term encodes a real casting failure mechanism:

- **Pour temperature — quadratic (U-shaped).** Too hot → hydrogen absorption → gas porosity. Too cold → misrun and cold shut. Optimum ≈ 700 °C.
- **Injection pressure — one-sided.** Low pressure fails to feed shrinkage → internal voids.
- **Injection speed — quadratic.** Too slow → cold shut. Too fast → turbulent fill → air entrapment.
- **Die temperature — one-sided.** Cold die → premature freezing at the skin.
- **Cooling time — one-sided.** Insufficient cooling → shrinkage and distortion on ejection.
- **Fe content — one-sided.** Fe above ~0.9 % forms brittle β-Al₅FeSi platelets.
- **Si content — one-sided.** Low Si → poor fluidity → incomplete fill.
- **Tool wear — monotone superlinear.** Worn dies → flash, dimensional drift.
- **Temperature × pressure interaction.** Hot melt *and* low pressure is far worse than either alone — the classic gas-porosity compounding case.

```python
import numpy as np
import pandas as pd
from scipy.special import expit

RANDOM_SEED = 42
NOISE_SD = 0.90          # irreducible process noise — tune this to hit AUC 0.80-0.88

CONTROLLABLE = ["pour_temp_c", "die_temp_c", "inj_pressure_bar",
                "inj_speed_ms", "hold_time_s", "cooling_time_s"]
UNCONTROLLABLE = ["si_content_pct", "fe_content_pct", "tool_wear_shots"]
FEATURES = CONTROLLABLE + UNCONTROLLABLE


def sample_process_params(n, rng):
    return pd.DataFrame({
        "pour_temp_c":      rng.normal(700, 18, n),
        "die_temp_c":       rng.normal(220, 20, n),
        "inj_pressure_bar": rng.normal(850, 70, n),
        "inj_speed_ms":     rng.normal(3.0, 0.45, n),
        "hold_time_s":      rng.normal(6.0, 1.0, n),
        "cooling_time_s":   rng.normal(12.0, 2.2, n),
        "si_content_pct":   rng.normal(9.5, 0.7, n),
        "fe_content_pct":   rng.gamma(4.0, 0.2, n),
        "tool_wear_shots":  rng.uniform(0, 50000, n),
    })


def _relu(x):
    return np.maximum(x, 0.0)


def defect_logit(df, rng, noise_sd=NOISE_SD):
    """Physically-motivated defect propensity. NEVER uses the label."""
    t  = (df["pour_temp_c"]      - 700) / 18      # standardised deviations
    dt = (df["die_temp_c"]       - 220) / 20
    p  = (df["inj_pressure_bar"] - 850) / 70
    v  = (df["inj_speed_ms"]     - 3.0) / 0.45
    h  = (df["hold_time_s"]      - 6.0) / 1.0
    c  = (df["cooling_time_s"]   - 12.0) / 2.2

    z = (
        0.95 * t**2                                   # gas porosity / misrun
      + 0.80 * _relu(-p)                              # shrinkage from underfeeding
      + 0.50 * v**2                                   # cold shut / turbulence
      + 0.40 * _relu(-dt)                             # cold die
      + 0.35 * _relu(-c)                              # insufficient cooling
      + 0.25 * _relu(-h)                              # short hold
      + 0.55 * _relu((df["fe_content_pct"] - 0.9) / 0.15)   # Fe intermetallics
      + 0.30 * _relu(-(df["si_content_pct"] - 8.5) / 0.7)   # low fluidity
      + 0.70 * (df["tool_wear_shots"] / 50000) ** 1.5       # die wear
      + 0.50 * t * _relu(-p)                          # hot + low pressure interaction
      + rng.normal(0, noise_sd, len(df))              # irreducible noise
    )
    return z.to_numpy()


def calibrate_intercept(z, target_prevalence):
    """Bisection for b0 such that mean(sigmoid(z + b0)) == target."""
    lo, hi = -25.0, 25.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if expit(z + mid).mean() < target_prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_split(image_paths, image_labels, rng, oversample=5):
    """
    image_paths  : list[str]
    image_labels : np.ndarray of 0/1  (1 = defective)
    Returns a DataFrame with one row per image, parameters generated causally.
    """
    n_img = len(image_labels)
    target_prev = image_labels.mean()

    cand = sample_process_params(n_img * oversample, rng)
    z = defect_logit(cand, rng)
    b0 = calibrate_intercept(z, target_prev)
    cand["true_defect_prob"] = expit(z + b0)
    cand["y_sim"] = rng.binomial(1, cand["true_defect_prob"])

    pool_def = cand.index[cand.y_sim == 1].to_numpy()
    pool_ok  = cand.index[cand.y_sim == 0].to_numpy()
    rng.shuffle(pool_def); rng.shuffle(pool_ok)

    need_def, need_ok = int(image_labels.sum()), int((1 - image_labels).sum())
    assert len(pool_def) >= need_def and len(pool_ok) >= need_ok, \
        "Increase `oversample`."

    it_def, it_ok = iter(pool_def), iter(pool_ok)
    chosen = [next(it_def) if lab == 1 else next(it_ok) for lab in image_labels]

    out = cand.loc[chosen].reset_index(drop=True)
    out["image_path"] = image_paths
    out["label"] = image_labels
    assert (out.y_sim.to_numpy() == image_labels).all()
    return out.drop(columns=["y_sim"])
```

Run it for train and test **with different RNG streams**, save to `data/processed/{train,test}_process.parquet`.

**Sanity checks before moving on:**
- Prevalence matches the image split (train ≈ 0.567, test ≈ 0.634)
- No single parameter correlates with the label above ~0.35 (`df.corr()['label']`)
- Plot `pour_temp_c` histograms split by label — you should see the U-shape (defects at *both* tails), not a clean separation

---

## 5. Phase 2 — Inline camera simulation

This is what makes your fusion result meaningful.

The Kaggle images were shot under a controlled lighting rig. A CNN gets ~99% on them, which leaves fusion nothing to contribute. **Real inline cameras on a moving conveyor do not produce lab-quality images** — there's motion blur, lighting drift, sensor noise, and lower effective resolution. Published work on this exact dataset shows classification accuracy dropping to the mid-80s under realistic noise.

So you evaluate under **two imaging regimes** and report both:

- **Regime A (Lab):** original images — the optimistic baseline
- **Regime B (Inline):** degraded images — the realistic deployment condition

```python
import cv2
import numpy as np

def degrade_inline(img_gray, rng, severity=1.0):
    """Simulate a conveyor-mounted inline inspection camera."""
    img = img_gray.astype(np.float32)

    # 1. Directional motion blur (part moving under the camera)
    k = 9
    kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k)); kern /= kern.sum()
    img = cv2.filter2D(img, -1, kern)

    # 2. Lighting drift (no controlled rig on the line)
    img = rng.uniform(0.65, 1.35) * img + rng.uniform(-30, 30)

    # 3. Sensor noise (short exposure, industrial CMOS)
    img += rng.normal(0, 12 * severity, img.shape)

    # 4. Effective resolution loss
    img = np.clip(img, 0, 255).astype(np.uint8)
    small = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (300, 300), interpolation=cv2.INTER_LINEAR)
```

**Critical rule: apply degradation identically to train and test.** If you train on clean and test on degraded, you're measuring domain shift, not fusion benefit — a different (and messier) experiment.

Save 6 before/after example pairs to `figures/` immediately. This becomes a figure in your report.

---

## 6. Phase 3 — Image feature extraction

`src/extract_features.py`. Runs once per regime, results cached.

```python
import numpy as np, torch, cv2
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_extractor():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()          # -> 512-d global average pooled embedding
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_image(path, rng=None, degrade=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if degrade:
        img = degrade_inline(img, rng)
    return np.repeat(img[:, :, None], 3, axis=2)   # grayscale -> 3ch for ImageNet stats

@torch.no_grad()
def extract(paths, degrade=False, seed=0, batch_size=32):
    rng = np.random.default_rng(seed)
    model = build_extractor()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    feats = []
    for i in tqdm(range(0, len(paths), batch_size)):
        batch = [_tf(load_image(p, rng, degrade)) for p in paths[i:i + batch_size]]
        feats.append(model(torch.stack(batch)).numpy())
    return np.vstack(feats).astype(np.float32)
```

```python
np.save("data/processed/train_feats_lab.npy",    extract(train_paths, degrade=False))
np.save("data/processed/test_feats_lab.npy",     extract(test_paths,  degrade=False))
np.save("data/processed/train_feats_inline.npy", extract(train_paths, degrade=True, seed=1))
np.save("data/processed/test_feats_inline.npy",  extract(test_paths,  degrade=True, seed=2))
```

Expect roughly 5–12 minutes per regime on a typical laptop CPU. Four files, one coffee break, and you never run it again.

> **If you have any GPU access at all** (Colab free tier counts), the one thing worth spending it on is fine-tuning the ResNet-18 backbone on the inline regime — it lifts the vision baseline and makes the comparison fairer. Everything else stays on your laptop.

---

## 7. Phase 4 — Models and the ablation study

`src/train_models.py`

The ablation **is** the experiment. Three models × two imaging regimes = six results.

```python
import numpy as np, xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, recall_score, precision_score,
                             confusion_matrix, brier_score_loss)

N_PCA = 64

def prepare(train_feats, test_feats, train_tab, test_tab):
    scaler = StandardScaler().fit(train_feats)
    pca = PCA(n_components=N_PCA, random_state=42).fit(scaler.transform(train_feats))
    tr_img = pca.transform(scaler.transform(train_feats))
    te_img = pca.transform(scaler.transform(test_feats))
    img_cols = [f"img_pc{i:02d}" for i in range(N_PCA)]
    return tr_img, te_img, img_cols, pca

def fit_xgb(X_tr, y_tr, X_te, y_te):
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.5, eval_metric="logloss",
        early_stopping_rounds=50, n_jobs=-1, random_state=42,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return clf

def evaluate(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc":   roc_auc_score(y_true, proba),
        "pr_auc":    average_precision_score(y_true, proba),
        "f1":        f1_score(y_true, pred),
        "recall":    recall_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "brier":     brier_score_loss(y_true, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
```

Three model configurations:

1. **Vision-only** — XGBoost on the 64 image PCs
2. **Process-only** — XGBoost on the 9 tabular features
3. **Fusion** — XGBoost on all 73 features

Run each under Lab and Inline. Also fit a `LogisticRegression` on the raw 512-d embeddings as a vision sanity check — if it wildly beats the PCA version, raise `N_PCA`.

### Expected result shape

These are *anticipated ranges*, not promises — report whatever you actually measure:

| Model | Lab imaging ROC-AUC | Inline imaging ROC-AUC |
|---|---|---|
| Vision-only | 0.98 – 0.995 | 0.86 – 0.91 |
| Process-only | 0.82 – 0.88 | 0.82 – 0.88 (unchanged) |
| **Fusion** | 0.98 – 0.996 | **0.92 – 0.95** |

**The headline finding:** under lab imaging, fusion adds little — vision already saturates. Under realistic inline imaging, fusion recovers a large share of the lost performance because the process signal is *uncorrelated with image quality*. This is a real, defensible, non-obvious conclusion, and it's exactly the kind of thing that earns a strong grade.

Plot ROC-AUC vs. degradation severity (0.0 → 2.0 in steps) for all three models. That single chart is your best figure.

---

## 8. Phase 5 — Root cause analysis and parameter recommendation

`src/explain.py`

### 8.1 Global drivers

```python
import shap
explainer = shap.TreeExplainer(fusion_model)
sv = explainer.shap_values(X_test)

shap.summary_plot(sv[:, tab_idx], X_test[:, tab_idx], feature_names=FEATURES)
```

Restrict the summary plot to the **tabular** columns. Image principal components are uninterpretable and clutter the figure. Report image contribution as a single aggregate: `|SHAP|` summed over all PCs versus summed over process features — one clean bar chart showing modality contribution.

Verify that SHAP recovers your simulator's physics. Pour temperature should show high SHAP at both extremes, tool wear should be monotone increasing. **If it doesn't, your model is not learning the causal structure** and you should say so rather than hide it.

### 8.2 Prescriptive recommendations

This is the section that converts a classifier into a *control system*, and almost no student does it.

For any part flagged as high-risk, search the controllable parameters for the smallest adjustment that drops predicted risk below threshold:

```python
STEPS = {
    "pour_temp_c":      np.arange(-30, 31, 5),
    "die_temp_c":       np.arange(-30, 31, 5),
    "inj_pressure_bar": np.arange(-120, 121, 20),
    "inj_speed_ms":     np.arange(-0.8, 0.81, 0.2),
    "hold_time_s":      np.arange(-2, 2.1, 0.5),
    "cooling_time_s":   np.arange(-4, 4.1, 1.0),
}

def recommend(model, row, feat_names, threshold, max_changes=2):
    """Cheapest single- or double-parameter counterfactual that clears threshold."""
    base = model.predict_proba(row.reshape(1, -1))[0, 1]
    candidates = []
    for name, deltas in STEPS.items():
        j = feat_names.index(name)
        for d in deltas:
            if d == 0:
                continue
            trial = row.copy(); trial[j] += d
            p = model.predict_proba(trial.reshape(1, -1))[0, 1]
            if p < threshold:
                cost = abs(d) / (deltas.max() - deltas.min())   # normalised effort
                candidates.append((cost, name, float(d), base, float(p)))
    candidates.sort()
    return candidates[:5]
```

Output reads like an operator instruction:

> Part #4417 — predicted defect risk **0.78**
> Recommended: reduce pour temperature by **15 °C** → risk drops to **0.21**
> Primary drivers: pour temperature (SHAP +1.42), injection pressure (SHAP +0.61)

**Validation you must include:** feed the recommended parameters back into the *ground-truth simulator* (`defect_logit`) and check that true defect probability actually falls. This is the only way to distinguish a real causal recommendation from a model artefact — and being able to do this validation is a genuine advantage of the simulated design. Report the fraction of recommendations that verify, and be honest if it's not 100%.

---

## 9. Phase 6 — Cost of Poor Quality

`src/economics.py`

### 9.1 Correct the prevalence first

Your dataset is ~57% defective. No foundry operates at 57% scrap. Applying your confusion matrix directly to production volumes would be economically meaningless.

Apply a **prior shift correction** to recalibrate probabilities to a realistic base rate (2–4%):

```python
def adjust_prior(p, prev_train, prev_real):
    r = (prev_real / prev_train) / ((1 - prev_real) / (1 - prev_train))
    return (p * r) / (p * r + (1 - p))
```

Then reweight your test-set confusion matrix, or resample the test set to the target prevalence. State the assumed base rate and cite a source or justify it.

Doing this correctly is a distinguishing detail. Most projects skip it entirely.

### 9.2 Cost model

| Parameter | Symbol | Example value | Notes |
|---|---|---|---|
| Annual volume | V | 500,000 parts | State your assumption |
| True defect rate | π | 3.0 % | Post-correction |
| Scrap cost / part | C_s | $18 | Material + melt + machine time |
| Rework cost / part | C_r | $6 | Where salvageable |
| **Escape cost** | C_e | $250 | Warranty, field failure, customer charge-back |
| False alarm cost | C_fa | $4 | Unnecessary teardown / manual re-inspection |
| Manual inspection recall | — | 85 % | Human baseline |

```
Expected annual cost = V · [ π·(1-Recall)·C_e  +  π·Recall·C_r  +  (1-π)·FPR·C_fa ]
Annual saving = Cost(manual baseline) − Cost(model)
```

### 9.3 Cost-optimal threshold

Do **not** use 0.5. Escapes cost ~60× more than false alarms, so the optimal threshold is far lower:

```python
def optimal_threshold(y_true, proba, c_fn, c_fp, grid=np.linspace(0.01, 0.99, 197)):
    costs = []
    for t in grid:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        costs.append(fn * c_fn + fp * c_fp)
    return float(grid[int(np.argmin(costs))]), np.array(costs)
```

Plot expected cost vs. threshold and mark the minimum. Then run **sensitivity analysis**: sweep C_e from $50 to $1000 and show how the optimal threshold and net saving move. That demonstrates you understand these are assumptions, not facts — which is exactly the maturity an examiner looks for.

Report savings for all three models. The fusion model's advantage in *dollars* under inline imaging is your closing argument.

---

## 10. Phase 7 — Power BI dashboard

Export one wide CSV per model/regime to `data/exports/`:

```
part_id, split, true_label, pred_label, defect_prob, defect_prob_adjusted,
pour_temp_c, die_temp_c, inj_pressure_bar, inj_speed_ms, hold_time_s,
cooling_time_s, si_content_pct, fe_content_pct, tool_wear_shots,
shap_pour_temp, shap_pressure, shap_speed, shap_tool_wear, ...,
top_recommendation, recommended_delta, risk_after_change,
model_name, imaging_regime, estimated_cost_usd, timestamp
```

Synthesise a `timestamp` column spanning a simulated production shift — it makes the time-series visuals work.

**Four pages:**

1. **Executive** — KPI cards (defect rate, model recall, annual COPQ, net saving), trend line, model comparison bar
2. **Process control** — SPC-style charts per parameter with control limits, defect rate overlaid; scatter of pour temp vs. pressure coloured by predicted risk (the interaction becomes visible)
3. **Root cause** — mean |SHAP| ranking, dependence plots, defect rate by tool-wear decile
4. **Part inspector** — searchable table, slicers on regime and risk band, per-part recommendation panel

Add a threshold slider parameter driving a live COPQ measure. It's the single most impressive interaction you can build and takes about twenty minutes.

*No Power BI licence?* Streamlit gives you the same four pages in Python, deploys free, and is arguably a better fit for a CV.

---

## 11. Four-week schedule

**Week 1 — Data foundation**
Download dataset · verify counts (3758/2875/453/262) · build simulator · sanity checks · degradation function · save before/after figures · EDA notebook
→ *Gate: no single feature correlates >0.35 with label; U-shape visible in pour temp.*

**Week 2 — Features and baselines**
Extract all four feature caches · vision-only and process-only baselines both regimes · locked evaluation protocol · calibration curves
→ *Gate: process-only AUC in 0.80–0.88. If above, increase `NOISE_SD` and regenerate.*

**Week 3 — Fusion and interpretation**
Fusion model · full 3×2 ablation · degradation sweep chart · SHAP analysis · recommendation engine · simulator-based validation of recommendations
→ *Gate: fusion beats vision-only under inline imaging by a margin you can defend.*

**Week 4 — Economics, dashboard, report**
Prior correction · cost-optimal thresholds · sensitivity analysis · Power BI build · report and slides
→ *Gate: every figure in the report is regenerable from a script.*

Freeze scope at end of Week 3. Week 4 is for writing, not for new experiments.

---

## 12. Report structure

1. **Introduction** — reactive inspection vs. predictive quality control; the gap
2. **Literature review** — vision-based defect detection; process-parameter SPC; multi-modal fusion in manufacturing; the paired-data gap that motivates your design
3. **Methodology** — dataset; causal simulator with the physics justification table; the inline-imaging argument; fusion architecture; **explicit limitations subsection**
4. **Experiments** — protocol, metrics, threshold selection
5. **Results** — 3×2 ablation table; degradation sweep; SHAP; recommendation validation rate
6. **Economics** — prior correction, COPQ, sensitivity
7. **Dashboard**
8. **Discussion** — when does fusion help, and why; deployment implications
9. **Limitations and future work** — simulated telemetry; single part geometry; binary rather than multi-class defect typing; path to validation on real paired data
10. **Conclusion**

Write section 9 properly. A candid limitations section signals you understand your own work; a defensive one signals the opposite.

---

## 13. Pitfalls, ranked by how much damage they do

1. **Label leakage into simulated parameters.** Project-ending. Check by confirming no feature correlates above ~0.35 with the label.
2. **Reporting only lab-imaging results.** Fusion looks pointless and your thesis collapses. The inline regime is the entire argument.
3. **Skipping the prevalence correction.** Your COPQ figures become fiction.
4. **Threshold fixed at 0.5.** Wrong by construction when FN costs 60× FP.
5. **Fitting the scaler or PCA on the full dataset.** Fit on train only, transform test. Silent, subtle, and an examiner may catch it.
6. **Degrading test images but not train images.** Measures domain shift, not fusion benefit.
7. **Reporting accuracy as the headline metric.** With imbalanced defect data, use PR-AUC and recall on the defect class.
8. **Attempting end-to-end CNN training on a laptop CPU.** You will lose a week to a run that doesn't converge.
9. **Presenting simulated data as real.** Declare it clearly, in the methodology, in your own words.
10. **Building the dashboard before the models are frozen.** You will rebuild it twice.

---

## 14. Immediate first steps

1. Create the venv, install dependencies, verify `import torch, xgboost, shap, cv2` all succeed
2. Download the Kaggle dataset; confirm the four folder counts match Section 2.2 exactly
3. Write `simulate_process.py`, run it, plot `pour_temp_c` histograms split by label — confirm the U-shape
4. Run `extract_features.py` on the *lab* regime only and time it; that number sizes the rest of your compute planning

Steps 1–4 are one solid evening. Once the feature cache exists, everything downstream is fast.

---

*Reference data on the casting dataset (7,348 images; train 3,758 def / 2,875 ok; test 453 def / 262 ok; 300×300 grayscale, controlled lighting) is from the Kaggle dataset documentation and published work using it. Cost figures throughout are illustrative — substitute real values if your institution has industry contacts, and state your sources.*
