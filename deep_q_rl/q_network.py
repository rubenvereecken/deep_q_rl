"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
import re
import logging
import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_channels, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, network_params, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng

        self.lstm = None
        self.next_lstm = None

        logging.debug('network parameters', network_params)
        self.network_params = network_params

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        networks = self.build_network(network_type, num_channels, input_width, input_height,
                                        num_actions, num_frames, None)
        if isinstance(networks, tuple):
            self.l_out = networks[0]
            self.lstm = networks[1]
        else:
            self.l_out = networks

        # theano.compile.function_dump('network.dump', self.l_out)
        if self.freeze_interval > 0:
            next_networks = self.build_network(network_type, num_channels, input_width,
                                                 input_height, num_actions,
                                                 num_frames, None)

            if isinstance(next_networks, tuple):
                self.next_l_out = next_networks[0]
                self.next_lstm = next_networks[1]
            else:
                self.next_l_out = next_networks

            self.reset_q_hat()

        # This really really needs to be floats for now.
        # It makes sense if they use it for computations
        btensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
        states = btensor5('states')
        next_states = btensor5('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        # Apparently needed for some layers with a variable input size
        # Weird, because the others just allow a None batch size,
        # but let's just play safe for now
        # For now, it should always look exactly like states
        # (n_batch, n_time_steps)
        # mask = T.imatrix('mask')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, num_channels, input_height, input_width),
                     dtype=theano.config.floatX), name='states')

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, num_channels, input_height, input_width),
                     dtype=theano.config.floatX), name='next_states')

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True), name='rewards')

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True), name='actions')

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # self.mask_shared = theano.shared(np.ones((batch_size, num_frames),
        #     dtype='int32'))

        # lstmout = lasagne.layers.get_output(self.lstm, states / input_scale)

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
                # mask_input=mask)

        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale
                                                    )
                                                    # mask_input=mask)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale
                                                    )
                                                    # mask_input=mask)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = (rewards +
                  (T.ones_like(terminals) - terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(target.shape[0]),
                               actions.reshape((-1,))].reshape((-1, 1))

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            #
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)
        # print params
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        if update_rule == 'deepmind_rmsprop':
            update_for = lambda params: deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            update_for = lambda params: lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            update_for = lambda params: lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        updates = update_for(params)

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        # # Super mega shady stuff
        # # Somehow an update sneaks in for cell and hid. Kill it with fire
        if self.lstm:
            delete_keys = [k for k, v in updates.items() if k.name in ['cell', 'hid']]
            # print delete_keys
            for key in delete_keys:
                del updates[key]

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})

        # self.lstmout = theano.function([], lstmout,
        #                                givens={states: self.states_shared},
        #                                on_unused_input='warn')

    def build_network(self, network_type, num_channels, input_width, input_height,
                      output_dim, num_frames, batch_size):
        if (network_type.endswith('cuda') or network_type.endswith('cudnn')) and \
                not theano.config.device.startswith("gpu"):
            prefix = re.sub(r'(_cudnn)|(_cuda)', '', network_type)
            cpu_version = "{}_cpu".format(prefix)
            logging.warn(network_type + " requested but no GPU found " +
                         "(device {}) ".format(theano.config.device) +
                         "defaulting to {}".format(cpu_version))
            network_type = cpu_version

        if network_type.endswith('cuda'):
            from lasagne.layers import cuda_convnet
            conv_layer = cuda_convnet.Conv2DCCLayer

        # Requires cuDNN which is not freely available.
        if network_type.endswith('cudnn') or network_type.endswith('gpu'):
            from lasagne.layers import dnn
            conv_layer = dnn.Conv2DDNNLayer

        if network_type.endswith('cpu'):
            from lasagne.layers import conv
            conv_layer = conv.Conv2DLayer

        input_dim = (batch_size, num_frames, num_channels, input_width, input_height)

        if network_type.startswith("nature"):
            return self.build_nature_network(input_dim, output_dim, conv_layer)
        if network_type.startswith('nips'):
            return self.build_nips_network(input_dim, output_dim, conv_layer)
        if network_type.startswith('lstm'):
            return self.build_lstm(input_dim, output_dim, conv_layer)
        if re.match(r'late.?fusion', network_type, re.IGNORECASE):
            return self.build_late_fusion(input_dim, output_dim, conv_layer)
        if network_type.startswith("linear"):
            return self.build_linear_network(input_dim, output_dim, conv_layer)
        if network_type.startswith('conv3d'):
            return self.build_conv3d(input_dim, output_dim, conv_layer)

        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        # Might be a slightly cheaper way by reshaping the passed-in state,
        # though that might destroy the original
        states = np.empty((1, self.num_frames, self.num_channels, self.input_height,
                           self.input_width), dtype=theano.config.floatX)
        states[0, ...] = state
        return self.more_q_vals(states)[0]

    # TODO use this when testing instead of one at a time
    def more_q_vals(self, states):
        self.states_shared.set_value(states)
        qvals = self._q_vals()
        # print np.sum(lstmout == self.lstm.hid.get_value())
        return qvals

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def build_nature_network(self, input_dim, output_dim, conv_layer):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        batch_size, num_frames, num_channels, input_width, input_height = input_dim

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, num_channels, input_width, input_height)
        )

        # Frames basically become channels - better only use this with a single channel
        l_reshape = lasagne.layers.ReshapeLayer(l_in,
            shape=(batch_size or -1, num_channels * num_frames, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_reshape,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = conv_layer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_prev = l_conv3
        pool_size = self.network_params['network_final_pooling_size']
        if pool_size:
            l_prev = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=pool_size)

        l_hidden1 = lasagne.layers.DenseLayer(
            l_prev,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_late_fusion(self, input_dim, output_dim, conv_layer):
        batch_size, num_frames, num_channels, input_width, input_height = input_dim

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, num_channels, input_width, input_height)
        )

        # We'll build 2 towers, so first split up into two streams
        # Take only the first and the last time frames
        # Only makes sense with at least two time frames, pref more
        previous = [
            # used to be I used slices to get a single frame but not that we
            # support channels we can drop the frame dimension
            lasagne.layers.SliceLayer(l_in, indices=0, axis=1),
            lasagne.layers.SliceLayer(l_in, indices=-1, axis=1),
        ]

        print lasagne.layers.get_output_shape(previous[0])

        # Have 2 layers
        conv_params_by_layer=[(16, (8, 8), (4, 4)), (32, (4, 4), (2, 2))]

        if self.network_params['network_late_fusion_share']:
            # Share weights across two layer at the same level
            shared_weights=[
                theano.shared(lasagne.init.Normal(.01)((16, num_channels, 8, 8))),
                # The 16 channels come from the filter size 16 of the previous one
                theano.shared(lasagne.init.Normal(.01)((32, 16, 4, 4))),
            ]

            shared_biases = [
                theano.shared(lasagne.init.Normal(.1)((16,))),
                theano.shared(lasagne.init.Normal(.1)((32,))),
            ]

        for layer_i, params in enumerate(conv_params_by_layer):
            for conv_i, prev in enumerate(previous):
                if self.network_params['network_late_fusion_share']:
                    W = shared_weights[layer_i]
                    b = shared_biases[layer_i]
                else:
                    W = lasagne.init.Normal(.01)
                    b = lasagne.init.Normal(.1)

                conv = conv_layer(
                    prev,
                    num_filters=params[0],
                    filter_size=params[1],
                    stride=params[2],
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=W,
                    b=b
                )
                print lasagne.layers.get_output_shape(conv)
                # print conv.get_W_shape()
                previous[conv_i] = conv

        pool_size = self.network_params['network_final_pooling_size']
        if pool_size:
            previous[0] = lasagne.layers.MaxPool2DLayer(previous[0], pool_size=pool_size)
            previous[1] = lasagne.layers.MaxPool2DLayer(previous[1], pool_size=pool_size)

        l_merge = lasagne.layers.ConcatLayer( previous,
            axis=1 # Merge along the time axis
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_merge,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_nips_network(self, input_dim, output_dim, conv_layer):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        batch_size, num_frames, num_channels, input_width, input_height = input_dim

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, num_channels, input_width, input_height)
        )

        l_reshape = lasagne.layers.ReshapeLayer(l_in,
            shape=(batch_size or -1, num_frames * num_channels, input_width, input_height)
        )

        print lasagne.layers.get_output_shape(l_in)

        l_conv1 = conv_layer(
            l_reshape,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        print lasagne.layers.get_output_shape(l_conv1)

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        print lasagne.layers.get_output_shape(l_conv2)

        l_prev = l_conv2
        pool_size = self.network_params['network_final_pooling_size']
        if pool_size:
            l_prev = lasagne.layers.MaxPool2DLayer(l_prev, pool_size=pool_size)

        l_hidden1 = lasagne.layers.DenseLayer(
            l_prev,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_conv3d(self, input_dim, output_dim, conv_layer):
        batch_size, num_frames, num_channels, input_width, input_height = input_dim

        l_in = lasagne.layers.InputLayer(
            # Author noted batch_size cannot be None, so wassup?
            # Only 1 channel
            shape=(batch_size, num_frames, num_channels, input_width, input_height)
        )

        # Conv3ddnn wants channels first, then frames, because depth gets treated differently
        l_shuffle = lasagne.layers.DimshuffleLayer(l_in,
            (0, 2, 1, 3, 4)
        )

        from lasagne.layers import dnn
        conv_layer = dnn.Conv3DDNNLayer

        l_conv1 = conv_layer(
            l_shuffle,
            num_filters=16,
            # Vary the temporal filter
            filter_size=(self.network_params['network_temp_filter1'] or 2, 8, 8),
            stride=(1, 4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        print lasagne.layers.get_output_shape(l_conv1)

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=32,
            # Vary the temporal filter
            filter_size=(self.network_params['network_temp_filter2'] or 2, 4, 4),
            stride=(1, 2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        previous_layer = l_conv2
        print lasagne.layers.get_output_shape(l_conv2)

        pool_size = self.network_params['network_final_pooling_size']
        if pool_size:
            print 'Using an additional max pool layer of size', pool_size

            previous_layer = lasagne.layers.dnn.Pool3DDNNLayer(l_conv2,
                    pool_size=pool_size)

        print lasagne.layers.get_output_shape(previous_layer)

        l_hidden1 = lasagne.layers.DenseLayer(
            previous_layer,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_lstm(self, input_dim, output_dim, conv_layer):
        batch_size, num_frames, num_channels, input_width, input_height = input_dim

        if self.network_params['network_lstm_type'] == 'mine':
            from stateful_lstm import LSTMLayer
        else:
            from lasagne.layers.recurrent import LSTMLayer

        l_in = lasagne.layers.InputLayer(
            # Batch size is undefined so we can chuck in as many as we please
            shape=(None, num_frames, num_channels, input_width, input_height)
        )

        if (num_frames > 1):
            logging.warn('Building LSTM with more than one frame input at a time')

        # Treat any incoming frames as channels
        l_reshape = lasagne.layers.ReshapeLayer(l_in,
                (-1, num_frames * num_channels, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_reshape,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
            # dimshuffle=True
        )

        # TODO check out diff between gates? seriously where did I get this
        default_gate = lasagne.layers.Gate(
            W_in=lasagne.init.Normal(.01),
            W_hid=lasagne.init.Normal(.01),
            W_cell=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            nonlinearity=lasagne.nonlinearities.sigmoid
        )


        from lasagne.layers.recurrent import Gate

        l_lstm1 = LSTMLayer(
                l_conv2,
                num_units=self.network_params['network_lstm_layer_size'] or 256,
                # ingate = default_gate,
                # outgate = default_gate,

                # If I use this most out values quickly become 0
                # For :class:`LSTMLayer` the bias of the forget gate is often initialized to
                # a large positive value to encourage the layer initially remember the cell
                # forgetgate = Gate(b=lasagne.init.Constant(5.0)),
                # forgetgate=default_gate,
                # cell = default_gate,
                # DRQ Paper says adding a ReLU after LSTM sucked,
                # Not sure if they meant as activation function or on top of it
                # TODO this is usually a tanh
                # TODO mention in thesis drqn says rectify sucks
                # nonlinearity=lasagne.nonlinearities.rectify,
                # This was for the stateless LSTM, unneeded now
                learn_init=self.network_params['network_lstm_learn_init'] or False,
                peepholes=True, # Internal connection from cell to gates
                gradient_steps=self.network_params['network_lstm_steps'] or 10, # -1 is entire history
                # grad_clipping=1, # From alex graves' paper, not sure here
                # This value comes from the other LSTM paper
                grad_clipping=self.network_params['network_lstm_grad_clipping'] or 10,
                only_return_final=True # Only need output for last frame
        )

        l_out = lasagne.layers.DenseLayer(
            l_lstm1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out, l_lstm1

    def build_linear_network(self, input_dim, output_dim, conv_layer):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """
        batch_size, num_frames, num_channels, input_width, input_height = input_dim

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, num_channels, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out

def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'nature_cuda')


if __name__ == '__main__':
    main()
