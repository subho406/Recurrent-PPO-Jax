import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from src.utils import tree_index
from src.models.rnns.rnn import LSTMMultiLayer,GRUMultiLayer
from flax.linen.initializers import constant, orthogonal
from src.models.transformers.gtrxl import GTrXL

def seq_model_lstm(**kwargs):
    def thurn():
        return LSTMMultiLayer(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'],reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return LSTMMultiLayer.initialize_state(kwargs['d_model'],kwargs['n_layers'])
    return thurn,initialize

def seq_model_gru(**kwargs):
    def thurn():
        return GRUMultiLayer(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'],
                             reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return GRUMultiLayer.initialize_state(kwargs['d_model'],kwargs['n_layers'])
    return thurn,initialize

def seq_model_gtrxl(**kwargs):
    def thurn():
        return GTrXL(head_dim=kwargs['head_dim'],embedding_dim=kwargs['embedding_dim'],head_num=kwargs['head_num'],
                     mlp_num=kwargs['mlp_num'],layer_num=kwargs['layer_num'],memory_len=kwargs['memory_len'],
                     dropout_ratio=kwargs['dropout_ratio'],gru_gating=True,gru_bias=kwargs['gru_bias'],train=kwargs.get('train',True),
                     reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return GTrXL.initialize_state(memory_len=kwargs['memory_len'],embedding_dim=kwargs['embedding_dim'],
                                      layer_num=kwargs['layer_num'])
    return thurn,initialize