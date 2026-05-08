# Imitation Learning Project

## Setup

```bash
pip install stable-baselines3[extra] gymnasium matplotlib pandas numpy
```

For IQ-Learn:
```bash
git clone https://github.com/Div99/IQ-Learn iq_learn
pip install -r iq_learn/requirements.txt
```

For f-IRL:
```bash
git clone https://github.com/twni2016/f-IRL f_irl
pip install -r f_irl/requirements.txt
```

## Workflow

### 1. Train expert policies
```bash
python train_expert.py --env CartPole-v1
python train_expert.py --env Pendulum-v1
```
Saves to: `experts/`

### 2. Collect expert datasets (all K values)
```bash
python collect_demos.py --env CartPole-v1
python collect_demos.py --env Pendulum-v1
```
Saves to: `datasets/`

### 3. Run IL experiments
```bash
python run_experiments.py --dry-run   # preview all 72 runs
python run_experiments.py             # run everything
```
Saves to: `results/results.csv`

### 4. Generate figures
```bash
python plot_results.py
```
Saves to: `figures/` — both PDF (report) and individual env plots

## Project structure
```
il_project/
├── train_expert.py       # Phase 1: train expert policies
├── collect_demos.py      # Phase 2: generate expert datasets
├── run_experiments.py    # Phase 3: sweep all algo/env/K/seed combos
├── plot_results.py       # Generate publication-quality figures
├── experts/              # Saved expert policy checkpoints
├── datasets/             # Expert trajectory .npz files
├── results/              # results.csv from all runs
├── figures/              # PDF plots for report
├── iq_learn/             # IQ-Learn repo (clone separately)
└── f_irl/                # f-IRL repo (clone separately)
```

## Experimental design
- **Environments:** CartPole-v1, Pendulum-v1
- **Algorithms:** IQ-Learn, f-IRL, SOAR + f-IRL
- **K values:** 1, 5, 20, 100 trajectories
- **Seeds:** 3 per configuration (0, 1, 2)
- **Total runs:** 2 × 3 × 4 × 3 = 72
