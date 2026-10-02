# POETIC: Signal-Decoupled Online Utility Learning under Strategic and Privacy-Preserving Spatial Crowdsourcing

> **Anonymous Repository for Peer Review**  
> In compliance with the EDBT double-blind review and reproducibility guidelines, all author-identifying metadata has been sanitized. This repository contains the complete implementation, baseline models, adversarial attack simulations, and automated benchmark pipelines for **POETIC**.

---

## 📖 Overview

In dynamic spatial crowdsourcing (SC) environments, platforms face a fundamental **Privacy-Incentive-Learning trilemma**:
1. **Adversarial Feedback Coupling**: Strategic workers exploit online Multi-Armed Bandit (MAB) learning engines by alternating between loss-leading bids and low-effort execution.
2. **Privacy Vulnerability**: Quality feedback and worker bids leak sensitive location and economic preferences.
3. **Non-Stationarity**: Spatiotemporal worker availability and environmental demand shift dynamically over time.

**POETIC** is a robust, privacy-preserving crowdsourcing framework featuring:
- **$\Phi$-Decoupled Online Utility Learning**: Structurally severs the feedback manipulation channel by updating utility estimates exclusively through sanitized execution quality signals ($\mathcal{D}_t^{\text{direct}} = 0$).
- **Stability-Aware Reputation Engine (StabRep)**: Employs a variance-penalized reputation metric that imposes an inescapable deception cost on oscillatory arbitrage attacks.
- **Multi-Stage Secure Top-$k$ Selection Protocol**: Reduces cryptographic sorting overhead from naive $\mathcal{O}(N \log^2 N)$ to $\mathcal{O}(N \log^2 g)$ under dual-server Secure Multi-Party Computation (SMPC).
- **Incentive-Compatible Settlement**: Adheres strictly to Myerson's critical-value payment characterization under virtual quality scoring, guaranteeing single-stage Dominant Strategy Incentive Compatibility (DSIC).
- **Rigorous Differential Privacy Accounting**: Provides provable cumulative privacy guarantees ($\mathcal{E}_{\text{total}} \le 3.82$ for $\delta = 10^{-5}$ across $T=5000$ rounds) via Rényi Differential Privacy (RDP).

---

## 📁 Repository Structure

```text
.
├── README.md                      # Reproduction guide and framework overview
├── requirements.txt               # Python package dependencies
├── config.py                      # Global parameters and system configurations
│
├── core/                          # POETIC Core Framework
│   ├── poe_platform.py            # Dual-server platform orchestration & decoupled learning
│   ├── protocols.py               # SMPC secure sorting & critical-value payment protocols
│   ├── worker.py                  # Worker lifecycle, cost models & 3 adversarial attack heuristics
│   ├── metrics.py                 # Evaluation metrics (NSV, regret, CumEstErr, runtime)
│   └── data_loader.py             # Spatiotemporal trace loading and spatial grid mapping
│
├── baselines.py                   # Competitor implementations (PrivCO, Truthful-DP-MAB, OPPS-adapted, etc.)
│
├── experiments/                   # Automated Audited Reproduction Scripts
│   ├── run_official_table3.py     # Master script for Table 3 (Clean vs. 30% Adversarial)
│   ├── run_paper_experiments_10runs.py # 10-run evaluation for system-wide performance (Fig. 1)
│   ├── run_ablation_10runs.py     # Component ablation study (w/o Phi, w/o StabRep)
│   ├── run_stability_10runs.py    # Concept drift & learning stability trajectories (Fig. 4)
│   ├── run_efficiency_exp.py      # Cryptographic runtime & communication scalability
│   └── run_esterr_experiment.py   # Utility tracking error convergence analysis
│
├── visualization/                 # Plotting scripts for paper figures
│   ├── plot_all_icde.py           # Unified figure generation pipeline
│   └── visualization.py           # Core plotting utilities
│
└── data/                          # Dataset preprocessing & sample records
    ├── preprocess_tdrive.py       # Beijing T-Drive trajectory preprocessor
    └── (processed data files)     # Parquet/PKL files for spatial grids and context maps
```

---

## 🛠️ Environment Setup & Prerequisites

### 1. Requirements
- **OS**: Linux (tested on Ubuntu 22.04), macOS, or Windows 10/11
- **Python**: Version $\ge 3.9$ (Python 3.11 / 3.12 recommended)
- **RAM**: $\ge 16\text{ GB}$ recommended for 10-run multi-process benchmarks

### 2. Installation
Create and activate a virtual environment, then install dependencies:

```bash
# Clone the repository (or extract the anonymous archive)
cd POETIC-EDBT

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet created, the core packages can be installed via)*:
```bash
pip install numpy scipy pandas pyarrow matplotlib
```

---

## 🚀 Quick Start (One-Click Verification)

To quickly verify that the simulation pipeline, dual-server cryptographic primitives, and decoupled learning algorithms run correctly, execute a short verification run ($T=200$ rounds):

```bash
# Run a lightweight sanity test on NYC Taxi dataset
python run_official_table3.py --dataset nyc --quick
```

---

## 📊 Reproducing Paper Results

All audited figures and tables reported in the paper can be reproduced using the scripts detailed below.

### 1. Reproducing Table 3: Macro-Level Performance Benchmark
Computes Cumulative Utility, Cumulative Dynamic Regret, Cumulative Estimation Error, and Average Runtime across both clean and 30% contaminated environments:

```bash
# Run full benchmark on NYC Taxi dataset (T=5000 rounds)
python run_official_table3.py --dataset nyc

# Run full benchmark on Beijing T-Drive dataset (T=5000 rounds)
python run_official_table3.py --dataset tdrive
```
*Outputs: Printed LaTeX tables and structured result logs in `./results/`.*

### 2. Reproducing Component Ablations (Table 3 & Fig. 7)
Evaluates POETIC variants: `POETIC (Full)`, `w/o StabRep` (no variance penalty), `w/o Phi` (no signal decoupling), and `w/o Both`:

```bash
python run_ablation_10runs.py
```

### 3. Reproducing Stability and Concept Drift Trajectories (Fig. 4)
Simulates abrupt global concept drift at round $t = 2500$ across 10 independent repetitions:

```bash
python run_stability_10runs.py
```
*Outputs: Generates `results/stability_10runs_official.json` and renders `fig4_stability.pdf`.*

### 4. Reproducing Cryptographic Efficiency & Scalability Benchmark
Measures SMPC comparison throughput, communication overhead (MB), and wall-clock execution time under varying worker pool sizes $N \in [500, 5000]$:

```bash
python run_efficiency_exp.py
```

### 5. Plotting Figures
To regenerate all PDF figures matching the publication:

```bash
python plot_all_icde.py
```

---

## 🗃️ Datasets & Preprocessing

The experiments evaluate real-world spatiotemporal mobility distributions from two publicly accessible benchmarks:

1. **NYC Taxi Trip Records**:
   - **Source**: NYC Taxi and Limousine Commission ([TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)).
   - **Spatial Coverage**: Manhattan urban grid ($500\text{m} \times 500\text{m}$ cells).
   - **Characteristics**: Denser demand peaks, high spatiotemporal volatility, long-tail traffic patterns.

2. **Beijing T-Drive Trajectory Dataset**:
   - **Source**: Microsoft Research ([T-Drive Sample](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)).
   - **Spatial Coverage**: Beijing metropolitan area.
   - **Characteristics**: Large-scale urban trajectory network with smooth spatial transitions.

### Preprocessing Pipelines
The raw GPS coordinate traces are converted into discrete spatial mobility grids and dispatch matrices:
```bash
# Preprocess raw NYC Taxi CSV records
python data/process_nyc.py

# Preprocess raw T-Drive trajectory text files
python preprocess_tdrive.py
```
Precomputed coordinate context maps (`context_maps_nyc.pkl`, `processed_nyc_with_grid.parquet`, `processed_tdrive.parquet`) are included in `./data/` for rapid execution without downloading the raw multi-gigabyte CSVs.

---

## ⚙️ Core Configuration & Hyperparameters

Default system configurations are formalized in `config.py`:

| Parameter | Symbol | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Operational Horizon** | $T$ | `5000` | Total interaction rounds |
| **Worker Population** | $N$ | `2000` | Active candidate pool size |
| **Round Winner Quota** | $k$ | `20` | Selected workers per round |
| **Initial Learning Rate** | $\alpha_0$ | `0.15` | Step size for EMA utility tracking |
| **Exploration Ratio** | $\rho_{\text{exp}}$ | `0.05` | Frequency of elite pool exploration rounds |
| **Elite Pool Factor** | $F$ | `3` | Candidate pool multiplier for exploration ($|\mathcal{E}_t| = \lfloor kF \rfloor$) |
| **Subgroup Size / Quota** | $g / c$ | `50 / 10` | SMPC Stage 1 subgroup size and local promotion quota |
| **Reputation Learning Rate** | $\eta_R / \eta_v$ | `0.2 / 0.2` | Running rates for reputation mean and variance momentum |
| **Stability Penalty Factor** | $\lambda$ | `0.5` | Variance discount factor in StabRep score |
| **Reputation Bonus Factor** | $\beta$ | `0.5` | Reputation bonus scaling factor |
| **Single-Round DP Budget** | $\epsilon_{\text{step}}$ | `1.0` | Laplace perturbation budget for reported execution quality |
| **Cumulative Privacy Bound** | $\mathcal{E}_{\text{total}}$ | $\le 3.82$ | Total RDP-composed privacy bound ($\delta = 10^{-5}$) |
| **Default Adversary Ratio** | $r_{\text{adv}}$ | `30%` | Default fraction of forward-looking strategic workers |

---

## 🛡️ Strategic Adversary Attack Models

To ensure non-trivial robustness validation, the framework implements three formal attack heuristics in `worker.py`:
1. **Oscillatory Arbitrage (Pumping & Dumping)**: Workers periodically alternate between high execution quality ($q_H = 4.8$) with loss-leading bids ($b_i = c_i - \delta_b$) to accumulate reputation, and low quality ($q_L = 1.0$) with bid markups to extract economic rents.
2. **Monotonic Quality Decay**: Emulates post-selection free-riding by linearly degrading execution effort after initial selection while maintaining competitive bids.
3. **Adaptive Empirical Best-Response**: Adjusts bids dynamically via stochastic approximation $b_i(t+1) = b_i(t) + \eta_b(\mathbb{I}_{\{w_i \in \mathcal{S}_t\}} - \pi_{\text{target}})$ to exploit the platform's clearing boundaries.

---

## 🔒 Anonymity & Open-Access Compliance

This codebase strictly adheres to the EDBT double-blind reviewing guidelines:
- No author names, affiliations, or identifying commit histories are included.
- All experiments can be reproduced end-to-end on standard multi-core commodity servers.
- Upon publication, the repository will be migrated to an open-source permissive license.
