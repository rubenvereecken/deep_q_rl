from ale_agent import NeuralAgent
import os
import cPickle
import time
import logging

import numpy as np

import ale_data_set

import sys

class RecurrentAgent(NeuralAgent):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, replay_start_size,
                 update_frequency, rng, save_path, profile, network_params):
        super(RecurrentAgent, self).__init__(
                 q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, replay_start_size,
                 update_frequency, rng, save_path, profile
                )
        self.network_params = network_params
        logging.debug('Initialized Recurrent Agent')


    def start_episode(self, observation):
        action = super(RecurrentAgent, self).start_episode(observation)

        # Reset LSTM state, I think we should start anew each episode
        if self.network_params['network_lstm_reset_on_start']:
            self.network.lstm.hid.set_value(np.zeros_like(self.network.lstm.hid,
                dtype=np.config.floatX))
            self.network.lstm.cell.set_value(np.zeros_like(self.network.lstm.cell,
                dtype=np.config.floatX))

        return action


    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
                self.data_set.random_sequential_batch(self.network.batch_size)
        # Save old state
        if self.network_params['network_lstm_reset_on_training']:
            hid = self.network.lstm.hid.get_value(borrow=False)
            cell = self.network.lstm.cell.get_value(borrow=False)
            self.network.lstm.hid.set_value(np.zeros_like(hid))
            self.network.lstm.cell.set_value(np.zeros_like(cell))

        # Reset the target network's inner state for this episode
        if self.network_params['network_lstm_reset_on_start']:
            self.network.next_lstm.hid.set_value(np.zeros_like(hid))
            self.network.next_lstm.cell.set_value(np.zeros_like(cell))

        training_result = self.network.train(states, actions, rewards,
                                  next_states, terminals)

        # Since the network is also used to generate exploration Q values,
        # better reset its previous state
        if self.network_params['network_lstm_reset_on_training']:
            self.network.lstm.hid.set_value(hid)
            self.network.lstm.cell.set_value(cell)

        return training_result

    def finish_testing(self, epoch):
        start_time = time.time()
        self.testing = False
        holdout_size = 3200

        # Evaluating Q values with a stateful LSTM seems less meaningful
        # Reset inner state anyway
        # self.network.lstm.hid.set_value(np.zeros_like(hid))
        # self.network.lstm.cell.set_value(np.zeros_like(cell))

        # Keep a random subset of transitions to evaluate performance over time
        if self.holdout_data is None and len(self.data_set) > holdout_size:
            self.holdout_data = self.data_set.random_batch(holdout_size)[0]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i, ...]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)

        total_time = time.time() - start_time
        logging.info("Finishing up testing took {:.2f} seconds".format(total_time))
