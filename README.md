# Recurrent PPO in Jax

This is a reimplementation of Recurrent PPO and A2C algorithm adapted from CleanRL PPO+LSTM. Recurrent PPO is a variant of the Proximal Policy Optimization (PPO) algorithm that incorporates a Recurrent Neural Network (RNN) to model temporal dependencies in sequential decision-making tasks. 

**Currently Supported Models:** 
1. Multilayered LSTM 
2. Multilayered GRU
3. GTrXL (https://arxiv.org/abs/1910.06764)

**Default supported environments:** MiniGrid

## Usage

```
python trainer.py trainer=ppo trainer/seq_model=gtrxl
```

## Available Configurations
The following configuration groups are available:

- task: minigrid_onehot, minigrid_pixel
- trainer: a2c, ppo
- trainer/seq_model: gru, lstm, gtrxl
You can override any configuration in the following way:

```
foo.bar=value
```

Here's the default configuration:
```yaml
tags: null
project_name: default_project
seed: 1
steps: 10000000
log_interval: 10000
eval_episodes: 20
eval_interval: 500000
use_wandb: false
task:
  task: minigrid_onehot
  name: MiniGrid-MemoryS13Random-v0
  view_size: 3
  max_steps: null
trainer:
  agent: ppo
  d_actor: 128
  d_critic: 128
  num_envs: 8
  rollout_len: 256
  sequence_length: null
  gamma: 0.99
  gae_lambda: 0.95
  num_minibatches: 8
  update_epochs: 4
  norm_adv: true
  clip_coef: 0.1
  ent_coef:
    initial: 0.01
    final: null
    max_decay_steps: 1000
    power: 1
  vf_coef: 0.5
  max_grad_norm: 0.5
  optimizer:
    learning_rate:
      initial: 0.00025
      final: null
      max_decay_steps: 1000
      power: 1
  seq_model:
    name: lstm
    d_model: 128
    n_layers: 1
    reset_hidden_on_terminate: true


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
## License
The code is available under is released under the MIT License. See LICENSE for more information.

