# Applied Project 3 Imitation Learning

Comparing IQ-Learn, f-IRL, and SOAR+f-IRL on CartPole and Pendulum with different amounts of expert data (K = 1, 5, 20, 100 trajectories).

## Setup

```bash
pip install stable-baselines3[extra] gymnasium matplotlib pandas numpy torch
```

Clone the repos we use:
```bash
git clone https://github.com/Div99/IQ-Learn
git clone https://github.com/twni2016/f-IRL
pip install -r IQ-Learn/requirements.txt
pip install -r f-IRL/requirements.txt
```

## How to run

**1. Train expert policies**
```bash
python train_expert.py --env CartPole-v1
python train_expert.py --env Pendulum-v1
```

**2. Collect demonstrations**
```bash
python collect_demos.py --env CartPole-v1
python collect_demos.py --env Pendulum-v1
```

**3. Run experiments**
```bash
python train_iqlearn_standalone.py --env CartPole-v1 --all
python train_iqlearn_standalone.py --env Pendulum-v1 --all
python train_firl_standalone.py --env CartPole-v1 --all
python train_firl_standalone.py --env Pendulum-v1 --all
python train_soar_firl_standalone.py --env CartPole-v1 --all
python train_soar_firl_standalone.py --env Pendulum-v1 --all
```

`--all` runs all K values and 3 seeds. For a single run: `--K 20 --seed 0`

**4. Plot results**
```bash
python plot_results.py
```

## Code structure

- `train_expert.py` trains PPO (CartPole) and SAC (Pendulum) experts
- `collect_demos.py` rolls out expert and saves trajectory datasets
- `train_iqlearn_standalone.py` IQ-Learn, imports iq_loss from IQ-Learn repo
- `train_firl_standalone.py` f-IRL, imports discriminator from f-IRL repo
- `train_soar_firl_standalone.py` SOAR+f-IRL, same as f-IRL but with L=4 critics
- `plot_results.py` makes figures from results CSVs

## Notes on implementation

For IQ-Learn we import `iq_loss` from the repo and write everything else ourselves (networks, buffer, training loop) to avoid their hydra/wandb setup.

For f-IRL we use `SMMIRLDisc` from the repo for the discriminator.

SOAR is the same as f-IRL but with 4 critics instead of 1. The optimistic Q (Algorithm 5 from the paper) is in `optimistic_q()`.