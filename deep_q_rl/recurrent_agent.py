from q_agent import NeuralAgent

class RecurrentAgent(NeuralAgent):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, replay_start_size, 
                 update_frequency, rng, save_path, profile):
        super(RecurrentAgent, self).__init__(
                 q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, replay_start_size, 
                 update_frequency, rng, save_path, profile
                )


