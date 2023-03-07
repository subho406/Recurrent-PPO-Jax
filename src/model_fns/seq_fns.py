import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from src.utils import tree_index
from src.models.rnns.rnn import LSTMMultiLayer,GRUMultiLayer
from flax.linen.initializers import constant, orthogonal

def seq_model_lstm(d_model,n_layers,reset_on_terminate):
    def thurn():
        return LSTMMultiLayer(d_model,n_layers,reset_on_terminate=reset_on_terminate)
    def initialize():
        return LSTMMultiLayer.initialize_state(d_model,n_layers)
    return thurn,initialize

def seq_model_gru(d_model,n_layers,reset_on_terminate):
    def thurn():
        return GRUMultiLayer(d_model,n_layers,reset_on_terminate=reset_on_terminate)
    def initialize():
        return GRUMultiLayer.initialize_state(d_model,n_layers)
    return thurn,initialize
