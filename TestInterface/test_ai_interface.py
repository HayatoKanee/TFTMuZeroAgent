import config
import datetime
import numpy as np
from TestInterface.test_global_buffer import GlobalBuffer
from Simulator.tft_simulator import parallel_env
import time

from TestInterface.test_replay_wrapper import BufferWrapper

from Simulator import utils

if config.ARCHITECTURE == 'Pytorch':
    from Models.MCTS_torch import MCTS
    from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
    from Models import MuZero_torch_trainer as MuZero_trainer
    from torch.utils.tensorboard import SummaryWriter
else:
    from Models.MCTS import MCTS
    from Models.MuZero_keras_agent import TFTNetwork
    from Models import MuZero_trainer
    import tensorflow as tf



class DataWorker(object):
    def __init__(self, rank):
        self.agent_network = TFTNetwork()
        self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]
        self.rank = rank
        self.placements = {}
    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, buffers, weights):

        self.agent_network.set_weights(weights)
        agent = MCTS(self.agent_network)
        # Reset the environment
        player_observation = env.reset()
        # This is here to make the input (1, observation_size) for initial_inference
        player_observation = self.observation_to_input(player_observation)

        # Used to know when players die and which agent is currently acting
        terminated = {player_id: False for player_id in env.possible_agents}

        # While the game is still going on.
        while not all(terminated.values()):
            # Ask our model for an action and policy
            actions, policy = agent.policy(player_observation)
            step_actions = self.getStepActions(terminated, actions)
            storage_actions = utils.decode_action(actions)

            # Take that action within the environment and return all of our information for the next player
            next_observation, reward, terminated, _, info = env.step(step_actions)
            # store the action for MuZero
            for i, key in enumerate(terminated.keys()):
                if not info[key]["state_empty"]:
                    # Store the information in a buffer to train on later.
                    buffers.store_replay_buffer(key, player_observation[0][i], storage_actions[i], reward[key],
                                                policy[i])

            # Set up the observation for the next action
            player_observation = self.observation_to_input(next_observation)

        # buffers.rewardNorm()
        buffers.store_global_buffer()

    def getStepActions(self, terminated, actions):
        step_actions = {}
        i = 0
        for player_id, terminate in terminated.items():
            if not terminate:
                step_actions[player_id] = self.decode_action_to_one_hot(actions[i], player_id)
                i += 1
        return step_actions

    def observation_to_input(self, observation):
        tensors = []
        masks = []
        for obs in observation.values():
            tensors.append(obs["tensor"])
            masks.append(obs["mask"])
        return [np.asarray(tensors), masks]

    def decode_action_to_one_hot(self, str_action, key):
        # if key == "player_0":
        #     print(str_action)
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0, 0, 0]
        for i in range(num_items + 1):
            element_list[i] = int(split_action[i])

        decoded_action = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])
        decoded_action[0:6] = utils.one_hot_encode_number(element_list[0], 6)

        if element_list[0] == 1:
            decoded_action[6:11] = utils.one_hot_encode_number(element_list[1], 5)

        if element_list[0] == 2:
            decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38) + \
                                   utils.one_hot_encode_number(element_list[2], 38)

        if element_list[0] == 3:
            decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38)
            decoded_action[44:54] = utils.one_hot_encode_number(element_list[2], 10)
        return decoded_action
    
    def evaluate_agents(self, env):
        agents = {"player_" + str(r): MCTS(TFTNetwork())
                  for r in range(config.NUM_PLAYERS)}
        agents["player_1"].network.tft_load_model(100)
        agents["player_2"].network.tft_load_model(100)
        agents["player_3"].network.tft_load_model(200)
        agents["player_4"].network.tft_load_model(200)
        agents["player_5"].network.tft_load_model(300)
        agents["player_6"].network.tft_load_model(300)
        agents["player_7"].network.tft_load_model(400)
        agents["player_0"].network.tft_load_model(400)

        while True:
            # Reset the environment
            player_observation = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = self.observation_to_input(player_observation)
            # Used to know when players die and which agent is currently acting
            terminated = {
                player_id: False for player_id in env.possible_agents}
            # Current action to help with MuZero
            self.placements = {
                player_id: 0 for player_id in env.possible_agents}
            current_position = 7
            info = {player_id: {"player_won": False}
                    for player_id in env.possible_agents}
            # While the game is still going on.
            
            while not all(terminated.values()):
                # Ask our model for an action and policy
                actions = []
                for i,key in enumerate(agents):
                    action, _ = agents[key].policy([np.asarray([player_observation[0][i]]), [player_observation[1][i]]])
                    actions.append(action)
                actions = [i[0] for i in actions]

                step_actions = self.getStepActions(terminated, actions)

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                # store the action for MuZero
                # Set up the observation for the next action
                player_observation = self.observation_to_input(next_observation)
                for key, terminate in terminated.items():
                    if terminate:
                        self.placements[key] = current_position
                        current_position -= 1
                        print(key)
                        del agents[key]

            for key, value in info.items():
                if value["player_won"]:
                    self.placements[key] = 0
            print(self.placements)
            for key in self.placements.keys():
                # Increment which position each model got.
                self.placements[key][self.placements[key]] += 1
            print("recorded places {}".format(self.placements))
            self.rank += config.CONCURRENT_GAMES

class AIInterface:

    def __init__(self):
        ...

    def train_model(self, starting_train_step=0):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_step = starting_train_step

        global_buffer = GlobalBuffer()

        env = parallel_env()
        data_workers = DataWorker(0)
        global_agent = TFTNetwork()
        global_agent.tft_load_model(train_step)

        if config.ARCHITECTURE == "Pytorch":
            trainer = MuZero_trainer.Trainer(global_agent)
            train_summary_writer = SummaryWriter(train_log_dir)
        else:
            trainer = MuZero_trainer.Trainer()
            tf.config.optimizer.set_jit(True)
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        while True:
            weights = global_agent.get_weights()
            buffers = BufferWrapper(global_buffer)
            data_workers.collect_gameplay_experience(env, buffers, weights)

            while global_buffer.available_batch():
                gameplay_experience_batch = global_buffer.sample_batch()
                trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
                train_step += 1
                if train_step % 100 == 0:
                    global_agent.tft_save_model(train_step)
    
    def evaluate(self):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


        env = parallel_env()
        data_workers = DataWorker(0)
        data_workers.evaluate_agents(env)

