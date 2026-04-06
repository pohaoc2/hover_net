# Stage 4 — Nutrient Proxy Maps: Equations & Assumptions

## Overview

Stage 4 derives three spatially-resolved proxy maps from CyCIF multiplex channel
intensities: **vasculature**, **oxygen**, and **glucose**. No direct hypoxia markers
(CA9, HIF-1α, GLUT1) are present in the panel; all nutrient maps are computed from
vascular geometry (CD31, SMA) and proliferative demand (Ki67, CD68).

---

## 1. Vasculature Mask

**Channels:** CD31 (required), SMA (optional refinement)

**Method:**

1. Percentile-normalize CD31 intensity → Otsu threshold → binary mask `V(x)`
2. Optional SMA refinement (pericyte support):

```
V_refined(x) = V_CD31(x)  ∪  ( V_SMA(x)  ∩  dilate(V_CD31(x), r) )

  r = --sma-adjacency-px  (default 2 px)
```

Only SMA pixels immediately adjacent to CD31 are included, reducing false
positives from cancer-associated fibroblasts (CAFs).

**Assumption:** CD31⁺ pixels delineate vessel lumens; SMA⁺ pixels adjacent to
CD31 indicate pericyte-covered (functionally perfused) vessels.

**No direct literature equation** — standard IHC binarization practice.

---

## 2. Oxygen Map

### 2a. Distance model (default: `--oxygen-model distance`)

**Channels:** CD31 vessel mask

**Equation:**

```
O2_hat(x) = 1 − clip( d(x, V) / d_max_O2,  0,  1 )

  d(x, V)    = Euclidean distance from pixel x to nearest vessel pixel
  d_max_O2   = 160 µm  (--oxygen-max-dist-um, default 160)

  O2_hat = 1  →  well-oxygenated (near vessel)
  O2_hat = 0  →  hypoxic (≥ 160 µm from any vessel)
```

**Source — clamp value (160 µm):**

> Grimes DR, Fletcher AG, Partridge M. *Oxygen consumption dynamics in steady-state
> tumour models.* **Royal Society Open Science** 1:140080, 2014.
> DOI: [10.1098/rsos.140080](https://royalsocietypublishing.org/doi/10.1098/rsos.140080)
>
> Derives analytical Krogh cylinder solutions; reports lethal O₂ radius ~160 µm
> in colorectal tumours under physiological consumption rates.

**Source — physical clamp normalization (vs. percentile):**

> Zaidi M, Fu F, Cojocari D, McKee TD, Wouters BG. *Quantitative Visualization of
> Hypoxia and Proliferation Gradients Within Histological Tissue Sections.*
> **Frontiers in Bioengineering and Biotechnology** 7:397, 2019.
> PMC: [PMC6906162](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906162/) |
> Code: [github.com/STTARR/Vessel-Distance-Analysis](https://github.com/STTARR/Vessel-Distance-Analysis)
>
> Computes per-nucleus distance to nearest CD31 vessel; validates correlation
> with pimonidazole-based hypoxia in CRC, pancreatic, and ovarian xenografts.

**Key assumption:** Oxygen follows a monotone gradient from vessel wall outward
(Krogh cylinder geometry). Pixels beyond 160 µm are all treated as equally hypoxic.

**Known limitation:** 2D sections miss out-of-plane vessels, systematically
underestimating oxygenation — see Grimes et al. *J R Soc Interface* 2016
([PMC4843681](https://pmc.ncbi.nlm.nih.gov/articles/PMC4843681/)).

---

### 2b. WSI-scale WKB model (`--oxygen-model wsi-pde`)

**Channels:** CD31 (vessel source), Ki67 (proliferative demand), CD68 (immune demand)

**Method:** Closed-form WKB (geometric optics) approximation of the steady-state
1D diffusion-consumption equation `D d²u/dx² − k·u = 0`. Solved once on the
**full WSI** at coarse resolution; per-patch results are cropped from the global field.

```
Algorithm:
  1. Build vessel mask V(x) from CD31 via Otsu threshold (coarse resolution)
  2. dist(x) = distance_transform_edt(~V(x))   — Euclidean distance from vessel
  3. k(x)    = k_base  +  w_Ki67 · K_hat(x)  +  w_CD68 · M_hat(x)
  4. L(x)    = krogh_um / (mpp_coarse · √(k(x)/k_base))
  5. u(x)    = exp(−dist(x) / L(x))

Variables:
  u(x)       ∈ [0,1]   normalised nutrient supply (output)
  krogh_um              e-folding decay length L in µm  (--oxygen-pde-krogh-um, default 200)
                        = √(D/k); distinct from --oxygen-krogh-um (160 µm, distance-model clamp)
                        Note: 160 µm is the Krogh *radius* (c=0, zero-order cylindrical PDE);
                        the first-order e-folding length L = √(D_O2/k_O2)
                        = √(2000/0.011–0.033) ≈ 200–400 µm (Secomb 1995, Grimes 2014)
  mpp_coarse            µm/px at coarse resolution = mpp × ds
  k_base                basal consumption  (--oxygen-consumption-base, default 0.1)
  K_hat(x)              percentile-normalised Ki67
  M_hat(x)              percentile-normalised CD68
  w_Ki67                (--oxygen-consumption-demand-weight, default 0.3)
  w_CD68                (--cd68-consumption-weight, default 0.1)
```

For constant k, step 5 is the **exact** solution to the 1D Cartesian diffusion-
consumption PDE. For spatially varying k(x), adjusting L(x) locally is the WKB
approximation — exact when k varies slowly relative to L.

**Note on `--wsi-pde-max-iters` / `--wsi-pde-tol`:** These CLI arguments are accepted
for compatibility but are **unused** — the WKB solver is non-iterative (O(N),
runs in ~25 s for a 30 M-pixel WSI). There are no Jacobi iterations.

**Source — k(x) formulation:**

> Kumar P, Lacroix M, Dupré P et al. *Deciphering oxygen distribution and hypoxia
> profiles in the tumor microenvironment: a data-driven mechanistic modeling approach.*
> **Physics in Medicine and Biology** 69:125023, 2024.
> DOI: [10.1088/1361-6560/ad524a](https://doi.org/10.1088/1361-6560/ad524a) |
> Dataset: [zenodo.org/records/10796880](https://zenodo.org/records/10796880)
>
> Proposes data-driven fitting of k(x) using CA9 hypoxia staining as ground truth.
> The paper does **not** publish fixed scalar weights; they are inferred from CA9 data.

**Parameter status:**
- k(x) additive form → from Kumar 2024
- `--oxygen-pde-krogh-um = 200 µm` (L = √(D_O2/k_O2) ≈ 200–400 µm from literature;
  200 µm is a conservative mid-range estimate). Note: the Grimes 2014 Krogh *radius*
  (160 µm) is from the zero-order cylindrical PDE — not equivalent to the first-order
  e-folding length used here.
- `k_base = 0.1`, `w_Ki67 = 0.3`, `w_CD68 = 0.1` → **heuristic defaults**;
  CA9/CAIX channel absent in this panel, so data-driven fitting cannot be performed.
  To fit: plot CA9 intensity vs. model residual as a function of distance from vessel,
  then optimize weights to minimise error.

**Key assumption:** Nutrient supply is vessel-sourced and decays exponentially with
distance. High Ki67/CD68 shortens the effective supply radius. The output is a
relative proxy, not an absolute concentration.

---

## 3. Glucose Map

### 3a. Distance model (`--glucose-model distance`)

**Channels:** CD31 vessel mask

**Equation:** Identical form to oxygen, with a wider diffusion clamp:

```
G_hat(x) = 1 − clip( d(x, V) / d_max_glc,  0,  1 )

  d_max_glc  = 450 µm  (--glucose-max-dist-um, default 450)
```

**Source — clamp value (450 µm):**

> Grimes DR et al. *Oxygen consumption dynamics in steady-state tumour models.*
> **Royal Society Open Science** 1:140080, 2014 (above); extended in
> *PLOS Computational Biology* 2015 ([PMID 26517813](https://pubmed.ncbi.nlm.nih.gov/26517813/)).
>
> Glucose critical supply distance ~450 µm in CRC, approximately 2.8× the O₂
> limit. Although glucose has a lower diffusion coefficient than O₂
> (D_glc ≈ 0.67×10⁻⁹ m²/s vs D_O2 ≈ 2.1×10⁻⁹ m²/s), the much higher plasma
> glucose concentration (~5 mM vs ~200 µM for O₂) produces a shallower gradient
> and a longer effective supply zone.

**Key assumption:** Glucose availability follows the same geometric proxy as
oxygen but decays over a physically wider radius.

---

### 3b. Demand model (`--glucose-model max`, default)

**Channels:** Ki67

**Equation:**

```
G_hat(x) = percentile_norm( Ki67(x) )

If PCNA present:
  G_hat(x) = max( percentile_norm(Ki67(x)),  percentile_norm(PCNA(x)) )
```

This maps **metabolic demand**, not supply. High values indicate proliferatively
active regions that require more glucose, not that glucose is available there.

**Biological basis:** Ki67 marks all active cell-cycle phases (G1–S–G2–M) and
is the standard clinical proliferation index. Ki67 alone is preferred over
Ki67+PCNA because PCNA is also elevated during DNA repair (not proliferation).

---

### 3c. WSI-scale WKB model (`--glucose-model wsi-pde`)

Same algorithm and consumption map as oxygen WKB (§2b), with independent parameters:
`--glucose-consumption-base`, `--glucose-consumption-demand-weight`,
`--glucose-pde-krogh-um` (default: 120 µm).

**Key difference from oxygen — why the models diverge:**

Both 160 µm (O₂) and 450 µm (glucose) in Grimes 2014 are **Krogh cylinder radii**
(r_k, where c→0) from the zero-order cylindrical PDE:

```
D · (1/r) · d/dr(r · du/dr) = k           (zero-order, cylindrical)
r_k = √(2 · D · C_vessel / k)
```

The code uses the first-order 1D approximation where the relevant quantity is the
**e-folding length** L = √(D/k), not r_k. These are different:

```
Model         Nutrient    Krogh radius r_k    e-folding L = √(D/k)    Why they differ
─────────────────────────────────────────────────────────────────────────────────────
Grimes 2014   O₂          160 µm              ~200–400 µm             r_k ≈ √(2·D·C/k);
                                                                       C_O2 ≈ 200 µM (low)
              Glucose     450 µm              ~120 µm                 C_glc ≈ 5 mM (high,
                                                                       drives large r_k)
```

In the normalized model (u=1 at vessel wall), C_vessel cancels out — the gradient
shape depends only on L = √(D/k). The high plasma glucose concentration that makes
r_k_glc = 450 µm is **not captured** once you normalize to [0, 1].

```
--oxygen-pde-krogh-um  = 200 µm  (L_O2  = √(2000 / 0.02)  ≈ 316 µm; 200 µm conservative)
--glucose-pde-krogh-um = 120 µm  (L_glc = √(75   / 0.005) ≈ 122 µm)
```

Glucose has a shorter e-folding length than oxygen despite having the larger Krogh
radius: D_glc (~75 µm²/s) is ~25× smaller than D_O2 (~2000 µm²/s), and k_glc is only
modestly smaller than k_O2, so L_glc < L_O2.

---

## 4. Ki67-vs-Distance Validation (`--validate-ki67-distance`)

**Purpose:** Validate that the CD31 distance proxy is biologically meaningful by
checking whether Ki67 peaks at intermediate vessel distances (Zaidi et al. 2019).

**Method:** Accumulate per-pixel Ki67 intensity in distance bins across all patches:

```
For each bin b:
  Ki67_mean(b) = sum( K_hat(x)  for x in bin(b) )  /  count( x in bin(b) )

  bin width = --validate-bin-um  (default 10 µm)
  bins cover 0 → d_max_O2
```

**Expected result (Zaidi 2019):** Ki67 peaks at ~50–100 µm from vessels
(sufficient O₂ for cell division), then drops beyond ~150 µm (hypoxic
quiescence). A flat or monotone curve indicates poor vessel mask quality.

**Output:** `{out}/validation/ki67_vs_distance.csv`
Columns: `distance_um`, `ki67_mean`, `pixel_count`

---

## Summary Table

| Component | Equation (text) | Parameter | Source |
|---|---|---|---|
| O₂ distance clamp | `d_max = 160 µm` | `--oxygen-krogh-um 160` | Grimes 2014 |
| Glucose distance clamp | `d_max = 450 µm` | `--glucose-krogh-um 450` | Grimes 2014/2015 |
| Physical normalization | `clip(d / d_max, 0, 1)` | — | Zaidi 2019 |
| WKB supply proxy | `u = exp(−dist / L(x))` | `--oxygen/glucose-model wsi-pde` | Kumar 2024 |
| O₂ WKB decay length | `L = 200 / mpp_coarse` = 154 px at ds=4 | `--oxygen-pde-krogh-um 200` | Secomb 1995, Grimes 2014 |
| Glucose WKB decay length | `L = 120 / mpp_coarse` = 92 px at ds=4 | `--glucose-pde-krogh-um 120` | Freyer 1988 |
| Consumption map | `k = k_base + w_K·K_hat + w_M·M_hat` | `--*-demand-weight`, `--cd68-*` | Kumar 2024 |
| Demand weight fitting | Requires CA9/CAIX ground truth | — | Kumar 2024 |
| SMA vessel refinement | `V_CD31 ∪ (V_SMA ∩ dilate(V_CD31))` | `--sma-adjacency-px` | — |
| Ki67 validation | mean Ki67 per distance bin | `--validate-bin-um` | Zaidi 2019 |
