import json
import numpy as np
import optax
import pandas as pd
import rlax
import wandb
import gymnasium as gym

from argparse import Namespace
from src.trainers.base_trainer import BaseTrainer
from collections import OrderedDict
from src.tasks.envs.minigrid_env import create_minigrid_env_onehot,create_minigrid_env_pixel
from src.agents.a2c import A2CAgent
from src.agents.ppo import PPOAgent
from src.model_fns import *
from src.tasks.envs.wrappers import *
from src.trainers.utils import *
from gymnasium.wrappers import AutoResetWrapper

def get_env_initializers(env_config):
    if env_config['task']=='minigrid_pixel':
        env_fn=lambda: create_minigrid_env_pixel(**env_config)
        repr_fn=atari_conv_repr_model()
    elif env_config['task']=='minigrid_onehot':
        env_fn=lambda: create_minigrid_env_onehot(**env_config)
        repr_fn=flatten_repr_model()
    return env_fn,repr_fn


class ControlTrainer(BaseTrainer):

    def __init__(self,**kwargs):
        """
            Initialize the MiniGridTrainer class.

            Args:
            - wandb_run: wandb run object.
            - trainer_config: A dictionary of training configuration parameters.
                - rollout_len: Length of a rollout.
                - num_actors: Number of actors to run in parallel.
                - gamma: Discount factor.
                - model: Model to use, either 'lstm' or 'repr_minigrid'.
                - d_model: Dimension of the model.
                - n_layers: Number of layers in the LSTM model.
                - d_actor: Dimension of the actor model.
                - d_critic: Dimension of the critic model.
                - lr: Learning rate.
                - max_grad_norm: Maximum gradient norm.
                - agent: Agent to use, either 'a2c' or 'ppo'.
                - lamb: Lambda value.
                - value_coef: Coefficient for the value loss.
                - entropy_coef: Coefficient for the entropy loss.
                - record_steps_interval: Steps interval for recording.
            - env_config: A dictionary of environment configuration parameters.
            - global_args: A dictionary of global configuration parameters.
            - key: Random key for Jax.
            - seed: Seed for random number generation.
        """

        env_fn,repr_fn=get_env_initializers(kwargs['env_config'])
        self.wandb_run=kwargs['wandb_run']
        self.trainer_config=kwargs['trainer_config']
        self.env_config=kwargs['env_config']
        self.global_config=kwargs['global_args']
        self.rollout_len=self.trainer_config['rollout_len']
        self.num_envs=self.trainer_config['num_envs']
        self.gamma=self.trainer_config['gamma']
        #We will create two environments, one for training (vectorized) and one for evaluation
        train_seeds=np.random.randint(0,9999,size=self.num_envs,dtype=int).tolist()
        eval_seeds=int(np.random.randint(0,9999,size=1,dtype=int))
        train_envs=gym.vector.SyncVectorEnv([lambda: EpisodeStatisticsWrapper(AutoResetWrapper(env_fn()))for seed in train_seeds])
        eval_env=RecordRollout(AutoResetWrapper(env_fn()))
        train_envs.reset(seed=train_seeds)
        eval_env.reset(seed=eval_seeds)
        

        params_key,self.random_key=jax.random.split(kwargs['key'])
        
        if self.trainer_config['model']=='lstm':
            model_fn=seq_model_lstm(self.trainer_config['d_model'],self.trainer_config['n_layers'],reset_on_terminate=self.trainer_config.get('reset_on_terminate',True))
        elif self.trainer_config['model']=='gru':
            model_fn=seq_model_gru(self.trainer_config['d_model'],self.trainer_config['n_layers'],reset_on_terminate=self.trainer_config.get('reset_on_terminate',True))
        actor_fn=actor_model_discete(self.trainer_config['d_actor'],eval_env.action_space.n)
        critic_fn=critic_model(self.trainer_config['d_critic'])
        #Setup optimizer
        
        if self.trainer_config['agent']=='a2c':
            self.optimizer=optax.chain(optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),  # Clip by the gradient by the global norm.
                                    optax.adam(**self.trainer_config.optimizer))  # Use Adam optimizer with learning rate.
            self.agent=A2CAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                rollout_len=self.rollout_len,
                                gamma=self.trainer_config['gamma'],lamb=self.trainer_config['lamb'],
                                value_loss_coef=self.trainer_config['value_coef'],
                                entropy_coef=self.trainer_config['entropy_coef'],
                                arg_max=self.trainer_config['arg_max'])
        elif self.trainer_config['agent']=='ppo':
            #Used from CleanRL PPO implementation
            batch_size = self.trainer_config['num_envs']*self.trainer_config['rollout_len'] 
            num_updates = self.global_config.steps // batch_size
            optimizer_config = dict(self.trainer_config.optimizer)
            learning_rate=optimizer_config.pop("learning_rate")
            def linear_schedule(count):
                # anneal learning rate linearly after one training iteration which contains
                # (args.num_minibatches * args.update_epochs) gradient updates
                frac = 1.0 - (count // (self.trainer_config['num_minibatches'] * 
                            self.trainer_config['update_epochs'])) / num_updates
                return learning_rate * frac

            self.optimizer=optax.chain(
                                optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),
                                optax.inject_hyperparams(optax.adam)(
                                    learning_rate=linear_schedule if self.trainer_config['anneal_lr'] 
                                    else learning_rate, eps=1e-5, **optimizer_config
                                ),
                            )
            self.agent=PPOAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                num_steps=self.rollout_len,
                                anneal_lr=self.trainer_config.get('anneal_lr', True),
                                gamma=self.trainer_config.get('gamma', 0.99),
                                gae_lambda=self.trainer_config.get('gae_lambda', 0.95),
                                num_minibatches=self.trainer_config.get('num_minibatches', 4),
                                update_epochs=self.trainer_config.get('update_epochs', 4),
                                norm_adv=self.trainer_config.get('norm_adv', True),
                                clip_coef=self.trainer_config.get('clip_coef', 0.1),
                                ent_coef=self.trainer_config.get('ent_coef', 0.01),
                                vf_coef=self.trainer_config.get('vf_coef', 0.5),
                                max_grad_norm=self.trainer_config.get('max_grad_norm', 0.5),
                                target_kl=self.trainer_config.get('target_kl', None))

        
        self.agent.reset(params_key,self.random_key)
        self.step_count=0
        self.episode_lengths=[]
        self.average_reward_per_episode=[]
        self.average_return_per_episode=[]
        self.losses=[]
        self.critic_losses=[]
        self.actor_losses=[]
        self.entropy_losses=[]
        self.result_data=[]
        self.reward_sum=0
        self.statistic_data=dict()
        self.B=self.num_envs*self.rollout_len
        self.log_interval=self.global_config.log_interval
        self.next_log_step=self.log_interval
        self.average_return_per_episode=[]
        if 'eval_interval' in self.global_config:
            self.eval_interval=self.global_config['eval_interval']
            self.next_eval_step=self.eval_interval
        else:
            self.eval_interval=None
        

    def step(self, **kwargs):
        self.random_key=jax.random.split(self.random_key)[0]
        (loss,(value_loss,entropy_loss,actor_loss,rewards),infos)=self.agent.step(self.random_key)
        #Extract info data across all actors and steps
        #Get the leaves of the infos tree where the final_info key is present
        
        leaves=[info for info in infos if '_final_info' in info]

        # Increase the step counter
        self.step_count+=(self.B)
        #Iterate over the leaves and extract the final_info data
        for leaf in leaves:
             for env_info in leaf['final_info'][leaf['_final_info']]:  
                 for key,value in env_info.items():
                     if isinstance(value, (int, float,)):
                         if key not in self.statistic_data:
                             self.statistic_data[key]=[]
                         self.statistic_data[key].append(value)

                 ep_rewards=jnp.array(env_info['rewards'],dtype=jnp.float32)
                 _,average_return_per_episode=average_reward_and_return_in_episode(ep_rewards,self.gamma)
                 self.average_return_per_episode.append(average_return_per_episode)

        # Log the data
        self.losses.append(loss)
        self.critic_losses.append(value_loss)
        self.actor_losses.append(actor_loss)
        self.entropy_losses.append(entropy_loss)
        self.reward_sum+=rewards.sum()
        if self.step_count>=self.next_log_step:
            #Calculate the mean of the elements in statistic_data and log them, finally clear the statistic_data
            metrics={}
            for key in self.statistic_data.keys():
                agg_value=np.mean(self.statistic_data[key])
                metrics={**metrics,key:agg_value}
                self.statistic_data[key]=[]
            self.next_log_step+=self.log_interval
            critic_loss=np.mean(self.critic_losses)
            actor_loss=np.mean(self.actor_losses)
            entropy_loss=np.mean(self.entropy_losses)
            loss=np.mean(self.losses)
            reward_mean=float(self.reward_sum/self.log_interval)
            return_mean=np.mean(self.average_return_per_episode)
            self.reward_sum=0
            self.critic_losses=[]
            self.actor_losses=[]
            self.entropy_losses=[]
            self.losses=[]
            self.average_return_per_episode=[]
            metrics={'step':self.step_count,'loss':loss,'critic_loss':critic_loss,
                                    'actor_loss':actor_loss,'entropy_loss':entropy_loss,'mean_reward':reward_mean,
                                    'return_per_episode':return_mean,
                                    **metrics
                                    }
            self.result_data.append(metrics)
        else:
            metrics=None
        if self.eval_interval is not None and self.step_count>=self.next_eval_step:
            self.next_eval_step+=self.eval_interval
            avg_episode_len,avg_episode_return,rollouts=self.agent.evaluate(self.random_key,self.global_config['eval_episodes'])
            rollouts=np.concatenate(rollouts,axis=0)
            if metrics is None:
                metrics={}
            metrics['step']=self.step_count
            metrics['eval_avg_episode_len']=float(avg_episode_len)
            metrics['eval_avg_episode_return']=float(avg_episode_return)
            metrics['rollouts']=wandb.Video(rollouts, fps=15, format="gif")
        return loss,metrics,self.step_count
    

    def get_summary_table(self):
        return pd.DataFrame(self.result_data).to_json(default_handler=str)
    
