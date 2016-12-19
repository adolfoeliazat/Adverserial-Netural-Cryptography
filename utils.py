# utils.py
import theano

import numpy as np


def gen_data(n=batch_size, msg_len=msg_len, key_len=key_len):
    x = (np.random.randint(0, 2, size=(n, msg_len))*2-1).astype(theano.config.floatX)
    y = (np.random.randint(0, 2, size=(n, key_len))*2-1).astype(theano.config.floatX)
    return x,y

def assess(pred_fn, n=batch_size, msg_len=msg_len, key_len=key_len):
    msg_in_val, key_val = gen_data(n, msg_len, key_len)
    return np.round(np.abs(msg_in_val[0:n] - pred_fn(msg_in_val[0:n], key_val[0:n])), 0)


def err_over_samples(err_fn, n=batch_size):
    msg_in_val, key_val = gen_data(n)
    return err_fn(msg_in_val[0:n], key_val[0:n])
