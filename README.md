# Recurrent PPO in Jax

This is a reimplementation of Recurrent PPO and A2C algorithm adapted from CleanRL PPO+LSTM. Recurrent PPO is a variant of the Proximal Policy Optimization (PPO) algorithm that incorporates a Recurrent Neural Network (RNN) to model temporal dependencies in sequential decision-making tasks. 

**Currently Supported Models:** Multilayered-LSTM, Multilayered-GRU
**Default supported environments:** MiniGrid

## Usage

```
python trainer.py 
```

## Available Configurations
The following configuration groups are available:

- task: minigrid_onehot, minigrid_pixel
- trainer: a2c_gru, a2c_lstm, ppo_lstm

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
eval_episodes: 10
eval_interval: 500000
use_wandb: false
task:
  task: minigrid_pixel
  name: MiniGrid-DoorKey-5x5-v0
  view_size: 7
  max_steps: null
trainer:
  agent: ppo
  model: lstm
  d_model: 128
  n_layers: 1
  d_actor: 128
  d_critic: 128
  num_envs: 8
  rollout_len: 256
  anneal_lr: false
  gamma: 0.99
  gae_lambda: 0.95
  num_minibatches: 8
  update_epochs: 4
  norm_adv: true
  clip_coef: 0.1
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: null
  reset_on_terminate: true
  optimizer:
    learning_rate: 0.0002
    
#Powered by Hydra (https://hydra.cc)
#Use --hydra-help to view Hydra specific help
```
## License
The code is available under is released under the MIT License. See LICENSE for more information.

