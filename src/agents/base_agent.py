import jax
import jax.numpy as jnp
import optax
import rlax
import tqdm
import numpy as np

from src.models.actor_critic import *
from typing import Callable
from src.utils import tree_index
from src.models.actor_critic import ActorCriticModel
from argparse import Namespace




class BaseAgent:
    def __init__(self,train_envs,eval_env,rollout_len,repr_model_fn:Callable,seq_model_fn:Callable,
                        actor_fn:Callable,critic_fn:Callable,use_gumbel_sampling=False) -> None:
        self.env=train_envs
        self.eval_env=eval_env
        self.rollout_len=rollout_len
        self.seq_fn,self.seq_init=seq_model_fn
        self.use_gumbel_sampling=use_gumbel_sampling
        self.ac_model=nn.vmap(ActorCriticModel,
                              variable_axes={'params': None},
                                split_rngs={'params': False})(repr_model_fn,self.seq_fn,actor_fn,critic_fn)
        @jax.jit
        def actor_critic_fn(random_key,params,inputs,terminations,last_memory):
            """

            Args:
                random_key (_type_): _description_
                params (_type_): _description_
                inputs (_type_): shape (BXTXrepr_dim)
                last_memory (_type_): _description_

            Returns:
                _type_: _description_
            """
            act_logits,values,memory=self.ac_model.apply(params,inputs,terminations,last_memory,rngs={'random':random_key})
            return act_logits,values,memory
        
        self.actor_critic_fn=actor_critic_fn
    
    

    def reset(self,params_key,random_key):
        #Reset the Agent and initilize the parameters
        self.tick=0
        self.o_tick,_=self.env.reset()
        self.r_tick=jnp.zeros(self.env.num_envs)
        self.term_tick=jnp.zeros(self.env.num_envs,dtype=jnp.int8)
        self.h_tickminus1=jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x,axis=0),self.env.num_envs,axis=0),self.seq_init())
        self._params=self.ac_model.init({'params':params_key,'random':random_key},jnp.expand_dims(self.o_tick,1),jnp.expand_dims(self.term_tick,1),
                                       self.h_tickminus1)
        def params_sum(params):
            return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape),params)))
        print("Number of params: ",params_sum(self.params))
        
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value
    
    def unroll_actors(self,random_key):
        """
        
        Starts with O_{tick},H_{tick-1} as the start state and 
            updates the state to O_(tick+rollout_len+1), H_(tick+rollout_len)
            Inheriting classes should optimize for timesteps tick to tick+rollout_len
        Returns:
            observations : list
            A list of observations O_(tick) - O_(tick+rollout_len+1)
        actions : list
            A list of actions from A_(tick) - A_(tick+rollout_len)
        rewards : list
            A list of observations R_(tick) - R_(tick+rollout_len+1)
        terminations : list
            A list of shape \gamma_(tick) - \gamma_(tick+rollout_len+1)
        critic_preds : list
            A list of critic predictions V_(tick) - V_(tick+rollout_len+1)
        actor_preds : list
            A list of actor predictions A_(tick) - A_(tick+rollout_len) and shape BXTXnum_actions
        
        
        """
        #Unrolls the actor for rollout_len steps, takes rollout_len actions
        h_tickminus1=self.h_tickminus1
        o_tick=self.o_tick
        r_tick=self.r_tick
        term_tick=self.term_tick
        actions=[]
        observations=[]
        rewards=[]
        critic_preds=[]
        actor_preds=[]
        terminations=[]
        infos=[]
        for t in range(1,self.rollout_len+1):
            #Add observation and reward and timestep tick
            observations.append(o_tick.copy())
            rewards.append(r_tick.copy())
            terminations.append(term_tick.copy())
            random_key,model_key=jax.random.split(random_key)
            act_logits,v_tick,htick=self.actor_critic_fn(model_key,self.params,jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),
                                                         h_tickminus1)
            if self.use_gumbel_sampling:
                # sample action: Gumbel-softmax trick
                # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
                u = jax.random.uniform(random_key, shape=act_logits.shape)
                acts_tick=jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
            else:
                acts_tick=jax.random.categorical(random_key,act_logits).squeeze(axis=-1)
            #Take a step in the environment
            o_tickplus1,r_tickplus1,term_tickplus1,trunc_tickplus1,info=self.env.step(acts_tick)
            term_tickplus1=jnp.logical_or(term_tickplus1,trunc_tickplus1)
            #Add action at timestep tick 
            critic_preds.append(v_tick.copy())
            actor_preds.append(act_logits.copy())
            actions.append(acts_tick.copy())
            infos.append(info)
            o_tick=o_tickplus1
            r_tick=r_tickplus1
            h_tickminus1=htick
            term_tick=term_tickplus1
            self.tick+=1
        #add the last observation and reward
        observations.append(o_tick)
        rewards.append(r_tick)
        terminations.append(term_tick)
        #get the value for timestep (tick+rollout_len+1), we need this to do bootstrapping
        random_key,model_key=jax.random.split(random_key)
        _,v_tick,_=self.actor_critic_fn(model_key,self.params,jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),h_tickminus1)
        critic_preds.append(v_tick)
        #Update to timestep
        self.o_tick=o_tick.copy()
        self.r_tick=r_tick.copy()
        self.h_tickminus1=h_tickminus1.copy()
        self.term_tick=term_tick.copy()
        #Shape is num_actorsXrollout_lenX*...
        return Namespace(**{
            'observations':jnp.stack(observations,1),
            'actions':jnp.stack(actions,1),
            'rewards':jnp.stack(rewards,1),
            'terminations':jnp.stack(terminations,1),
            'infos':infos,
            'critic_preds':jnp.squeeze(jnp.stack(critic_preds,1),axis=-1), #During unrolling phase, we don't need the time dimension as it is one, instead we need the rollout_len dimension
            'actor_preds':jnp.stack(actor_preds,1)
        })
        

    def evaluate(self,random_key,eval_episodes):
        #Evaluate the agent for evaluation_steps
        #Create a single zero hidden state
        #Get the hidden state from the first actor
        o_tick,_=self.eval_env.reset()
        episode_lens=[]
        episode_avgreturns=[]
        rollouts=[]
        for i in tqdm.tqdm(range(eval_episodes)):
            done=False
            rewards=[]
            #Initialize zero hidden state at the start of each episode, shape is infered from the hidden state of the first environment
            h_tickminus1=jax.tree_map(lambda x:jnp.expand_dims(jnp.zeros(x[0].shape),0) ,self.h_tickminus1)
            while not done:
                #Take a step in the environment
                random_key,model_key=jax.random.split(random_key)
                act_logits,v_tick,htick=self.actor_critic_fn(model_key,self.params,jnp.expand_dims(o_tick,axis=(0,1)),jnp.ones((1,1)),h_tickminus1)
                if hasattr(self,'arg_max') and self.arg_max:
                    acts_tick=jnp.argmax(act_logits,axis=-1)
                else:
                    if self.use_gumbel_sampling:
                         # sample action: Gumbel-softmax trick
                        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
                        u = jax.random.uniform(random_key, shape=act_logits.shape)
                        acts_tick=jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
                    else:
                        acts_tick=jax.random.categorical(random_key,act_logits).squeeze(axis=-1)
                o_tick,r_tick,term,trunc,info=self.eval_env.step(acts_tick)
                done=term or trunc
                rewards.append(r_tick)
                h_tickminus1=htick
            #Get the rollout frames
            rollouts.append(info['frames'])
            episode_lens.append(len(rewards))
            rewards=jnp.array(rewards,dtype=jnp.float32)
            avg_return=rlax.discounted_returns(rewards,self.gamma*jnp.ones_like(rewards),jnp.zeros_like(rewards)).mean()
            episode_avgreturns.append(avg_return)
        avg_episode_len=jnp.array(episode_lens).mean()
        avg_episode_return=jnp.array(episode_avgreturns).mean()
        return avg_episode_len,avg_episode_return,rollouts