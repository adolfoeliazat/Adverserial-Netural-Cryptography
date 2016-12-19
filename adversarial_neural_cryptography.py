import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from utils import LeNetConvPoolLayer, HiddenLayer, collect_params, generate_data
from lasagne.updates import adam, sgd

from config import *

rng = np.random.RandomState(29797)

class AdverserialConvSetup():

    def __init__(self, reshaped_input, name='unnamed'):
        
        self.name = name
        #poolsize = (1,1) to make the LeNet layer a normal Conv Layer
        self.conv_layer1 = LeNetConvPoolLayer(rng,
                                              input = reshaped_input,
                                              filter_shape=(2, 1, 4, 1), 
                                              image_shape=(None, 1, None, 1),
                                              stride=(1,1),
                                              poolsize = (1, 1),
                                              border_mode=(2,0),
                                              activation = T.nnet.relu)
        
        self.conv_layer2 = LeNetConvPoolLayer(rng,
                                              input = self.conv_layer1.output, 
                                              filter_shape=(4, 2, 2, 1),
                                              image_shape=(None, 2, None, 1),
                                              stride=(2,1),
                                              poolsize = (1, 1),
                                              border_mode=(0,0),
                                              activation = T.nnet.relu)
        
        self.conv_layer3 = LeNetConvPoolLayer(rng,
                                              input = self.conv_layer2.output,
                                              filter_shape=(4, 4, 1, 1),
                                              image_shape=(None, 4, None, 1),
                                              stride=(1,1),
                                              poolsize = (1, 1),
                                              border_mode=(0,0),
                                              activation = T.nnet.relu)
        
        self.conv_layer4 = LeNetConvPoolLayer(rng,
                                              input = self.conv_layer3.output,
                                              filter_shape=(1, 4, 1, 1),
                                              image_shape=(None, 4, None, 1),
                                              stride=(1,1),
                                              poolsize = (1, 1),
                                              border_mode=(0,0),
                                              activation = T.tanh)

        
        self.output = self.conv_layer4.output
        
        self.params = self.conv_layer1.params + self.conv_layer2.params + self.conv_layer3.params + self.conv_layer4.params        
            
msg_in = T.matrix('msg_in')
key = T.matrix('key')

#Alice's Network
alice_in = T.concatenate([msg_in, key], axis=1)

alice_hid = HiddenLayer(rng,
                        input = alice_in,
                        n_in = msg_len + key_len,
                        n_out = msg_len + key_len,
                        activation = T.nnet.relu)

alice_conv_in = alice_hid.output.reshape((batch_size, 1, msg_len + key_len, 1))

alice_conv = AdverserialConvSetup(alice_conv_in, 'alice')

alice_comm = alice_conv.output.reshape((batch_size, msg_len))

#Bob's Network
bob_in = T.concatenate([alice_comm, key], axis=1)

bob_hid = HiddenLayer(rng,
                      input = bob_in, 
                      n_in = comm_len + key_len,
                      n_out = comm_len + key_len,
                      activation = T.nnet.relu)

bob_conv_in = bob_hid.output.reshape((batch_size, 1, comm_len + key_len, 1))

bob_conv = AdverserialConvSetup(bob_conv_in, 'bob')

bob_msg = bob_conv.output.reshape((batch_size, msg_len))

#Eve's Network
eve_hid1 = HiddenLayer(rng,
                       input = alice_comm, 
                       n_in = comm_len,
                       n_out = comm_len + key_len,
                       activation = T.nnet.relu)
                          
eve_hid2 = HiddenLayer(rng,
                       input = eve_hid1.output, 
                       n_in = comm_len + key_len,
                       n_out = comm_len + key_len,
                       activation = T.nnet.relu)

eve_conv_in = eve_hid2.output.reshape((batch_size, 1, comm_len + key_len, 1))

eve_conv = AdverserialConvSetup(eve_conv_in, 'eve')

eve_msg = eve_conv.output.reshape((batch_size, msg_len))



decrypt_err_eve = T.mean(T.abs_(msg_in - eve_msg))

decrypt_err_bob = T.mean(T.abs_(msg_in - bob_msg))

loss_bob = decrypt_err_bob + (1. - decrypt_err_eve) ** 2.


params_bob  = collect_params([bob_conv, bob_hid, alice_conv, alice_hid])
updates_bob  = adam(loss_bob, params_bob, learning_rate)
error_bob   = theano.function(inputs=[msg_in, key],
                                    outputs=decrypt_err_bob)
train_bob = theano.function(inputs=[msg_in, key],
                                    outputs=loss_bob,
                                    updates=updates_bob)
pred_bob  = theano.function(inputs=[msg_in, key], outputs=bob_msg)

params_eve   = collect_params([eve_hid1, eve_hid2, eve_conv])
updates_eve  = adam(decrypt_err_eve, params_eve, learning_rate)
error_eve   = theano.function(inputs=[msg_in, key],
                                  outputs=decrypt_err_eve)
train_eve = theano.function(inputs=[msg_in, key],
                                  outputs=decrypt_err_eve,
                                  updates=updates_eve)
pred_eve  = theano.function(inputs=[msg_in, key], outputs=eve_msg)

def train(bob_or_eve, results, max_iters, print_every, es=0., es_limit=100):
    count = 0
    for i in range(max_iters):
        
        msg_in_val, key_val = generate_data()
        
        if bob_or_eve == 'bob':
            loss = train_bob(msg_in_val, key_val)
            results = np.hstack((results, error_bob(msg_in_val, key_val).sum()))
        elif bob_or_eve == 'eve':
            loss = train_eve(msg_in_val, key_val)
            results = np.hstack((results, error_eve(msg_in_val, key_val).sum()))
        
        if i % print_every == 0:
            print 'training loss:', loss
        
        if es and loss < es:
            count += 1
            if count > es_limit:
                break
    return np.hstack((results, np.repeat(results[-1], max_iters - i - 1)))

results_bob, results_eve = [], []


# Perform adversarial training
def run_adverserial_training():
    results_bob = [] 
    results_eve = []
    for i in range(adversarial_iterations):

        print 'training bob and alice, run:', i+1
        results_bob = train('bob', results_bob, n_iterations, n_iterations*printing_factor, es=early_stopping_criterion)
        print 'training eve, run:', i+1
        results_eve = train('eve', results_eve, n_iterations, n_iterations*printing_factor, es=early_stopping_criterion)

    return results_bob, results_eve

# Plotting the results of each experiment
if __name__=='__main__':
    results_bob, results_eve = run_adverserial_training()
    plt.plot([np.min(results_bob[i:i+n_iterations])*100 for i in np.arange(0, len(results_bob), n_iterations)])
    plt.plot([np.min(results_eve[i:i+n_iterations])*100 for i in np.arange(0, len(results_eve), n_iterations)])
    plt.legend(['bob', 'eve'])
    plt.xlabel('Number of Adversarial Runs')
    plt.ylabel('Decryption Error')
    plt.show()