# utils.py
import theano
from config import *
import numpy as np


def generate_data(n=batch_size, msg_len=msg_len, key_len=key_len):
    x = (np.random.randint(0, 2, size=(n, msg_len))*2-1).astype(theano.config.floatX)
    y = (np.random.randint(0, 2, size=(n, key_len))*2-1).astype(theano.config.floatX)
    return x,y