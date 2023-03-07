import jax.numpy as jnp
import jax.nn as nn
import numpy as np
import jax

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal



class LSTM(nn.Module):
    d_model:int
    reset_on_terminate:bool=True

    @nn.compact
    def __call__(self,inputs,terminations,last_state):
        #inputs; TXinput_dim
        #carry: 
        #last_hidden: d_model
        reset_on_terminate=self.reset_on_terminate
        class LSTMout(nn.Module):
            @nn.compact    
            def __call__(self,carry,inputs):
                inputs,terminations=inputs
                if reset_on_terminate:
                    carry=jax.tree_map(lambda x: x*(1.0-terminations),carry) #Stop gradient if termination
                (new_c, new_h), new_h=nn.OptimizedLSTMCell(kernel_init=orthogonal(jnp.sqrt(2)),
                            recurrent_kernel_init=orthogonal(jnp.sqrt(2)),bias_init=constant(0.0))(carry,inputs)
                return (new_c, new_h), ((new_c, new_h),new_h)
        model=nn.scan(LSTMout,variable_broadcast="params",
                   split_rngs={"params": False},)
        carry,(new_states,y_t)=model()(last_state,(inputs,terminations))

        return y_t,new_states
    
    def initialize_state(self):
        return (jnp.zeros((self.d_model,)),jnp.zeros((self.d_model,)))



class LSTMMultiLayer(nn.Module):
    d_model:int
    n_layers:int
    reset_on_terminate:bool=True
    @nn.compact
    def __call__(self, inputs, terminations, last_states):
        """
        inputs: TXinput_dim
        terminations: T
        """
        new_memory=[None]*self.n_layers
        for i in range(self.n_layers):
            if i == 0:
                y_t, new_memory[i] = LSTM(self.d_model,self.reset_on_terminate)(inputs, terminations,last_states[i])
            else:
                y_t, new_memory[i] = LSTM(self.d_model,self.reset_on_terminate)(y_t, terminations,last_states[i])
        return y_t, new_memory
    
    @staticmethod
    def initialize_state(d_model,n_layers):
        return [(jnp.zeros((d_model,)),jnp.zeros((d_model,))) for _ in range(n_layers)]


class GRU(nn.Module):
    d_model:int
    reset_on_terminate:bool=True

    @nn.compact
    def __call__(self,inputs,terminations,last_state):
        #inputs; TXinput_dim
        #carry: 
        #last_hidden: d_model
        reset_on_terminate=self.reset_on_terminate
        class GRUout(nn.Module):
            @nn.compact    
            def __call__(self,carry,inputs):
                inputs,terminations=inputs
                if reset_on_terminate:
                    carry=jax.tree_map(lambda x: x*(1.0-terminations),carry) #Stop gradient if termination
                new_c, new_h=nn.GRUCell(kernel_init=orthogonal(jnp.sqrt(2)),
                            recurrent_kernel_init=orthogonal(jnp.sqrt(2)),bias_init=constant(0.0))(carry,inputs)
                return new_c, (new_c,new_h)
        model=nn.scan(GRUout,variable_broadcast="params",
                   split_rngs={"params": False},)
        carry,(new_states,y_t)=model()(last_state,(inputs,terminations))

        return y_t,new_states
    
    def initialize_state(self):
        return jnp.zeros((self.d_model,))


class GRUMultiLayer(nn.Module):
    d_model:int
    n_layers:int
    reset_on_terminate:bool=True
    @nn.compact
    def __call__(self, inputs, terminations, last_states):
        """
        inputs: TXinput_dim
        terminations: T
        """
        new_memory=[None]*self.n_layers
        for i in range(self.n_layers):
            if i == 0:
                y_t, new_memory[i] = GRU(self.d_model,self.reset_on_terminate)(inputs, terminations,last_states[i])
            else:
                y_t, new_memory[i] = GRU(self.d_model,self.reset_on_terminate)(y_t, terminations,last_states[i])
        return y_t, new_memory
    
    @staticmethod
    def initialize_state(d_model,n_layers):
        return [jnp.zeros((d_model,)) for _ in range(n_layers)]