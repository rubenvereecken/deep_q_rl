import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import cuda_convnet

conv_layer = cuda_convnet.Conv2DCCLayer

batch_size = 4
num_frames = 1
input_width, input_height = (84, 84)

def build_network():
    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height)
    )

    l_conv = conv_layer(
        l_in,
        num_filters=16,
        filter_size=(8,8),
        stride=(4,4),
    )
    return l_conv

l_out = build_network()

rewards = T.col('rewards')
actions = T.icol('actions')
terminals = T.icol('terminals')

rewards_shared = theano.shared(
    np.zeros((batch_size, 1), dtype=theano.config.floatX),
    broadcastable=(False, True), name='rewards')

actions_shared = theano.shared(
    np.zeros((batch_size, 1), dtype='int32'),
    broadcastable=(False, True), name='actions')

givens = {
    rewards: rewards_shared,
    actions: actions_shared,
}

q_vals = lasagne.layers.get_output(l_out)
diff = q_vals[T.arange(rewards.shape[0]), actions.reshape((-1,))].reshape((-1, 1))
loss = T.sum(diff)

params = lasagne.layers.helper.get_all_params(l_out)
updates = lasagne.updates.rmsprop(loss, params, .99, .99, 1e-6)
train = theano.function([], [loss], updates=updates, givens=givens)
