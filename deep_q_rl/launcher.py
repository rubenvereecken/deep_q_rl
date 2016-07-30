#! /usr/bin/env python
"""This script handles reading command line arguments and starting the
training process.  It shouldn't be executed directly; it is used by
run_nips.py or run_nature.py.

"""
import os
import time
import argparse
import logging
import ale_python_interface
import cPickle
import numpy as np
import theano
import simplejson as json

import ale_experiment
import q_network
import profile


def parameters_as_dict(parameters):
    args_dict = {}
    args = [arg for arg in dir(parameters) if not arg.startswith('_')]
    for arg in args:
        args_dict[arg] = getattr(parameters, arg)
    return args_dict

def save_parameters(args, save_path):
    name = '/'.join((save_path, 'parameters' + '.json'))
    with open(name,'wb') as f:
        json.dump(parameters_as_dict(args), f, sort_keys=True, indent='\t')

# hackity hack
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def process_args(args, defaults, description):
    """
    Handle the command line.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('--display-screen', dest="display_screen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                        '(default is the name of the game)')
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                        '(default: %(default)s)')
    parser.add_argument('--repeat-action-probability',
                        dest="repeat_action_probability",
                        default=defaults.REPEAT_ACTION_PROBABILITY, type=float,
                        help=('Probability that action choice will be ' +
                              'ignored (default: %(default)s)'))

    parser.add_argument('--update-rule', dest="update_rule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('deepmind_rmsprop|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    parser.add_argument('--batch-accumulator', dest="batch_accumulator",
                        type=str, default=defaults.BATCH_ACCUMULATOR,
                        help=('sum|mean (default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    parser.add_argument('--clip-delta', dest="clip_delta", type=float,
                        default=defaults.CLIP_DELTA,
                        help=('Max absolute value for Q-update delta value. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate')
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--phi-length', dest="phi_length",
                        type=int, default=defaults.PHI_LENGTH,
                        help=('Number of recent frames used to represent ' +
                              'state. (default: %(default)s)'))
    parser.add_argument('--max-history', dest="replay_memory_size",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--network-type', dest="network_type",
                        type=str, default=defaults.NETWORK_TYPE,
                        help=('nips_cuda|nips_dnn|nature_cuda|nature_dnn' +
                              '|linear (default: %(default)s)'))
    parser.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+
                              '(default: %(default)s)'))
    parser.add_argument('--replay-start-size', dest="replay_start_size",
                        type=int, default=defaults.REPLAY_START_SIZE,
                        help=('Number of random steps before training. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--resize-method', dest="resize_method",
                        type=str, default=defaults.RESIZE_METHOD,
                        help=('crop|scale (default: %(default)s)'))
    parser.add_argument('--nn-file', dest="nn_file", type=str, default=None,
                        help='Pickle file containing trained net.')
    parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                        type=str, default=defaults.DEATH_ENDS_EPISODE,
                        help=('true|false (default: %(default)s)'))
    parser.add_argument('--max-start-nullops', dest="max_start_nullops",
                        type=int, default=defaults.MAX_START_NULLOPS,
                        help=('Maximum number of null-ops at the start ' +
                              'of games. (default: %(default)s)'))
    parser.add_argument('--deterministic', dest="deterministic",
                        type=bool, default=defaults.DETERMINISTIC,
                        help=('Whether to use deterministic parameters ' +
                              'for learning. (default: %(default)s)'))
    parser.add_argument('--cudnn_deterministic', dest="cudnn_deterministic",
                        type=bool, default=defaults.CUDNN_DETERMINISTIC,
                        help=('Whether to use deterministic backprop. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--color_averaging', dest="color_averaging",
                        type=bool, default=defaults.COLOR_AVERAGING,
                        help=('Whether to use ALE color averaging. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--log_level', dest="log_level",
                        type=str, default=logging.INFO,
                        help=('Log level to terminal. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--progress-frequency', dest="progress_frequency",
                        type=str, default=defaults.PROGRESS_FREQUENCY,
                        help=('Progress report frequency. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--save-path', dest='save_path',
                        type=str, default='../logs')
    parser.add_argument('--dont-generate-logdir', dest='generate_logdir',
            default=True, action='store_false')
    parser.add_argument('--profile', dest='profile', action='store_true')
    parser.add_argument('--resume', dest='resume', default=False,
            action='store_true', help='Resume from save_path')
    parser.add_argument('--screen-mode', dest='screen_mode', default='grayscale',
            help='grayscale|rgb')

    parser.add_argument('--network_lstm_layer_size', type=int, default=256)
    parser.add_argument('--network_lstm_steps', type=int)
    parser.add_argument('--network_lstm_grad_clipping', type=int)
    parser.add_argument('--network_temp_filter1', type=int)
    parser.add_argument('--network_temp_filter2', type=int)
    parser.add_argument('--network_final_pooling_size', type=int, default=0)
    parser.add_argument('--network_lstm_reset_on_start', type=bool, default=True)
    parser.add_argument('--network_lstm_reset_on_training', type=bool, default=True)

    parameters = parser.parse_args(args)
    if parameters.resume:
        with open(parameters.save_path + '/parameters.json', 'r') as f:
            param_dict = json.load(f)
            param_dict['resume'] = True
            param_dict['save_path'] = parameters.save_path
            parameters = Struct(**param_dict)
            print parameters.death_ends_episode
        print "Resuming operation on {}".format(parameters.save_path)

    if parameters.experiment_prefix is None:
        name = os.path.splitext(os.path.basename(parameters.rom))[0]
        parameters.experiment_prefix = name

    if parameters.death_ends_episode in ['true', True]:
        parameters.death_ends_episode = True
    elif parameters.death_ends_episode in ['false', False]:
        parameters.death_ends_episode = False
    else:
        raise ValueError("--death-ends-episode must be true or false")

    # If some kind of recurrent network
    if parameters.network_type.find('lstm') >= 0:
        print parameters.phi_length
        # parameters.phi_length = 1

    # This addresses an inconsistency between the Nature paper and the Deepmind
    # code. The paper states that the target network update frequency is
    # "measured in the number of parameter updates". In the code it is actually
    # measured in the number of action choices.
    # The default still has the same result as DeepMind's code, only the result
    # is achieved like DeepMind's paper describes it.
    parameters.freeze_interval = (parameters.freeze_interval //
                                  parameters.update_frequency)

    return parameters



def launch(args, defaults, description):
    """
    Execute a complete training run.
    """

    parameters = process_args(args, defaults, description)
    start_epoch = 1
    save_path = parameters.save_path
    if parameters.resume:
        # Resume from last saved network
        network_file_tpl = os.path.join(parameters.save_path,'network_file_{}.pkl')
        for i in range(parameters.epochs):
            if not os.path.exists(network_file_tpl.format(i+1)):
                break

        if i == 0:
            parameters.nn_file = None
        else:
            parameters.nn_file = network_file_tpl.format(i)
        start_epoch = i + 1
        print('Starting epoch {}'.format(start_epoch))
    else:
        if parameters.generate_logdir:
            try:
                # CREATE A FOLDER TO HOLD RESULTS
                time_str = time.strftime("_%d-%m-%Y-%H-%M-%S", time.gmtime())
                save_path = parameters.save_path + '/' + parameters.experiment_prefix + time_str
                os.makedirs(save_path)
            except OSError as ex:
                # Directory most likely already exists
                pass
            try:
                link_path = parameters.save_path + '/last_' + parameters.experiment_prefix
                os.symlink(save_path, link_path)
            except OSError as ex:
                os.remove(link_path)
                os.symlink(save_path, link_path)

        save_parameters(parameters, save_path)

    logger = logging.getLogger()
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # log to file
    fileHandler = logging.FileHandler("{0}/out.log".format(save_path))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    # log to stdout
    import sys
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logFormatter)
    logger.addHandler(streamHandler)
    logger.setLevel(parameters.log_level)

    if parameters.profile:
        profile.configure_theano_for_profiling(save_path)

    if parameters.rom.endswith('.bin'):
        rom = parameters.rom
    else:
        rom = "%s.bin" % parameters.rom
    full_rom_path = os.path.join(defaults.BASE_ROM_PATH, rom)

    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    if parameters.cudnn_deterministic:
        theano.config.dnn.conv.algo_bwd = 'deterministic'


    ale = ale_python_interface.ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))

    # TODO make it display
    if parameters.display_screen:
        import sys
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX

    ale.setBool('display_screen', parameters.display_screen)
    ale.setFloat('repeat_action_probability',
                 parameters.repeat_action_probability)
    ale.setInt('frame_skip', parameters.frame_skip)
    ale.setBool('color_averaging', parameters.color_averaging)

    ale.loadROM(full_rom_path)

    num_actions = len(ale.getMinimalActionSet())

    network_params = {k:v for k,v in parameters.__dict__.iteritems() if
            k.find('network') >= 0}

    if parameters.nn_file is None:
        network = q_network.DeepQLearner(defaults.RESIZED_WIDTH,
                                         defaults.RESIZED_HEIGHT,
                                         num_actions,
                                         parameters.phi_length,
                                         parameters.discount,
                                         parameters.learning_rate,
                                         parameters.rms_decay,
                                         parameters.rms_epsilon,
                                         parameters.momentum,
                                         parameters.clip_delta,
                                         parameters.freeze_interval,
                                         parameters.batch_size,
                                         parameters.network_type,
                                         parameters.update_rule,
                                         parameters.batch_accumulator,
                                         rng,
                                         network_params)
    else:
        handle = open(parameters.nn_file, 'r')
        network = cPickle.load(handle)

    if parameters.network_type.find('lstm') >= 0:
        import recurrent_agent
        agent_class = recurrent_agent.RecurrentAgent
    else:
        import ale_agent
        agent_class = ale_agent.NeuralAgent

    agent = agent_class(network,
                                  parameters.epsilon_start,
                                  parameters.epsilon_min,
                                  parameters.epsilon_decay,
                                  parameters.replay_memory_size,
                                  parameters.replay_start_size,
                                  parameters.update_frequency,
                                  rng, save_path,
                                  parameters.profile,
                                  parameters.screen_mode,
                                  network_params)

    experiment = ale_experiment.ALEExperiment(ale, agent,
                                              defaults.RESIZED_WIDTH,
                                              defaults.RESIZED_HEIGHT,
                                              parameters.resize_method,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              parameters.death_ends_episode,
                                              parameters.max_start_nullops,
                                              rng,
                                              parameters.progress_frequency,
                                              start_epoch)


    experiment.run()



if __name__ == '__main__':
    pass
