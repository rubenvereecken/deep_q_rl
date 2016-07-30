"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging
import time
import numpy as np
import cv2

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class ALEExperiment(object):
    def __init__(self, ale, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 death_ends_episode, max_start_nullops, rng, progress_frequency,
                 start_epoch=1):
        self.ale = ale
        self.agent = agent
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        # self.epoch = start_epoch        # Above 1 when resuming
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.death_ends_episode = death_ends_episode
        self.min_action_set = ale.getMinimalActionSet()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = ale.getScreenDims()

        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length,
                                       self.height, self.width),
                                      dtype=np.uint8)

        self.terminal_lol = False # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng
        self.progress_frequency = progress_frequency
        self.experiment_start_time = time.time()


    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for self.epoch in range(self.start_epoch, self.num_epochs + 1):
            epoch = self.epoch
            epoch_start_time = time.time()
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)

            total_epoch_time = time.time() - epoch_start_time
            average_epoch_speed = (self.epoch_length+self.test_length)/total_epoch_time
            epochs_left = self.num_epochs - epoch
            logging.info("Finished training + testing epoch {}, took {:.2f}s for {}+{} steps".format(
                         epoch, total_epoch_time, self.epoch_length, self.test_length) +
                         " ({:.2f} steps/s on avg)".format(average_epoch_speed))
            logging.info("Expecting the experiment ({} epochs ) to take about {:.2f} seconds longer".format(
                         epochs_left, epochs_left * total_epoch_time))

        logging.info("Finished experiment, took {}s".format(
                    time.time() - self.experiment_start_time))
        logging.shutdown()


    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined by
        the number of steps executed. An epoch will be cut short if not enough
        steps are left. Prints a progress report after every trial.

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training
        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        self.steps_left_this_epoch = num_steps
        prefix = "testing" if testing else "training"
        logging.info("Starting {} epoch {}/{}".format(prefix, epoch,
            self.num_epochs))
        epoch_start_time = time.time()
        self.last_progress_time = epoch_start_time

        # It's less pretty, keeping track through self.steps_left_this_epoch,
        # but it's decidedly better for logging throughout the experiment
        while self.steps_left_this_epoch > 0:
            logging.debug(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(self.steps_left_this_epoch))
            _, steps_run = self.run_episode(self.steps_left_this_epoch, testing)

        total_time = time.time() - epoch_start_time
        logging.info("Finished {} epoch {}; took {:.2f} seconds for {} steps ({:.2f} steps/s on avg)".format(
                        prefix, epoch, total_time, num_steps, num_steps / total_time))


    def _init_episode(self):
        """
        This method resets the game if needed, performs enough null actions to
        ensure that the screen buffer is ready and optionally performs a
        randomly determined number of null action to randomize the initial game
        state.
        """

        if not self.terminal_lol or self.ale.game_over():
            self.ale.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self._act(0) # Null action

        # I removed the buffer-filling noops since every noop now effectively
        # takes 4 frames. That's a tad too long.
        # Alright... I added it back in
        self._act(0)


    def _act(self, action):
        """
        Perform the indicated action for a single frame, return the resulting
        reward and store the resulting screen image in the buffer
        """
        reward = self.ale.act(action)
        index = self.buffer_count % self.buffer_length

        # TODO DeepMind uses luminance, not grayscale. Big diff?
        self.ale.getScreenGrayscale(self.screen_buffer[index, ...])

        self.buffer_count += 1
        return reward


    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()

        start_lives = self.ale.lives()

        action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        while True:
            reward = self._act(self.min_action_set[action])
            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.ale.lives() < start_lives)
            terminal = self.ale.game_over() or self.terminal_lol
            num_steps += 1
            self.steps_left_this_epoch -= 1

            if self.steps_left_this_epoch % self.progress_frequency == 0:
                time_since_last = time.time() - self.last_progress_time
                logging.info("steps_left:\t{}\ttime spent on {} steps:\t{:.2f}s\tsteps/second:\t{:.2f}".format
                             (self.steps_left_this_epoch, self.progress_frequency,
                              time_since_last, self.progress_frequency / time_since_last))
                self.agent.report()
                self.last_progress_time = time.time()

            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, self.epoch, terminal)
                break

            action = self.agent.step(reward, self.get_observation())
        return terminal, num_steps


    def get_observation(self):
        """ Resize and merge the previous two screen images """
        # This used to take the average, instead of simply ALE's builtin
        # color_averaging. DeepMind's code seems to use averaging by default
        # (through alewrap), so so shall we. Averaging is now set through ALE.

        assert self.buffer_count >= 1

        index = self.buffer_count % self.buffer_length - 1
        image = self.screen_buffer[index, ...]

        return self.resize_image(image)


    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError('Unrecognized image resize method.')

