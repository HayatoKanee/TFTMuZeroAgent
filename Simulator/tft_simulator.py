import config
import functools
import gymnasium as gym
import numpy as np
from typing import Dict
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    local_env = TFT_Simulator(env_config=None)

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env


parallel_env = parallel_wrapper_fn(env)


class TFT_Simulator(AECEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self, env_config):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.actions_taken_this_turn = 0
        self.game_round.play_game_round()

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.kill_list = []
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(low=-5.0, high=5.0, shape=(config.OBSERVATION_SIZE,), dtype=np.float64)
                    for _ in enumerate(self.agents)
                ],
            )
        )

        self.action_spaces = {agent: MultiDiscrete(config.ACTION_DIM)
                              for agent in self.agents}
        super().__init__()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        return self.action_spaces[agent]

    def check_dead(self):
        num_alive = 0
        for key, player in self.PLAYERS.items():
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.game_round.NUM_DEAD = self.NUM_DEAD
                    self.pool_obj.return_hero(player)
                    self.PLAYERS[key] = None
                    self.kill_list.append(key)
                    self.game_round.update_players(self.PLAYERS)
                else:
                    num_alive += 1
        return num_alive

    def observe(self, player_id):
        return self.observations[player_id]

    def reset(self, seed=None, options=None):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.NUM_DEAD = 0
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.game_round.play_game_round()
        self.game_round.play_game_round()

        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.observations = {agent: self.game_observations[agent].get_lobo_observation(self.PLAYERS[agent], 
                            self.step_function.shops[self.PLAYERS[agent].player_num], self.PLAYERS) for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        super().__init__()
        return self.observations

    def render(self):
        ...

    def close(self):
        self.reset()

    def step(self, action):
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return
        action = np.asarray(action)
        if action.ndim == 0:
            self.step_function.action_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                 self.agent_selection, self.game_observations)
        elif action.ndim == 1:
            reward, self.observations[self.agent_selection] = self.step_function.single_step_action_controller(action, 
                                                                        self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                                        self.agent_selection, self.game_observations)
            # self.step_function.batch_2d_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
            #                                        self.agent_selection, self.game_observations, self.PLAYERS)

        # if we don't use this line, rewards will compound per step 
        # (e.g. if player 1 gets reward in step 1, he will get rewards in steps 2-8)
        # self._clear_rewards()
        # self.rewards[self.agent_selection] = \
        #     self.PLAYERS[self.agent_selection].reward - self.previous_rewards[self.agent_selection]
        # self.previous_rewards[self.agent_selection] = self.PLAYERS[self.agent_selection].reward
        # self._cumulative_rewards[self.agent_selection] = \
        #     self._cumulative_rewards[self.agent_selection] + self.rewards[self.agent_selection]

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}

        # Also called in many environments but the line above this does the same thing but better
        # self._accumulate_rewards()
        if self._agent_selector.is_last():
            self.actions_taken += 1

            # If at the end of the turn
            if self.actions_taken >= config.ACTIONS_PER_TURN:
                # Take a game action and reset actions taken
                self.actions_taken = 0
                self.game_round.play_game_round()

                # Check if the game is over
                if self.check_dead() == 1 or self.game_round.current_round > 48:
                    # Anyone left alive (should only be 1 player unless time limit) wins the game
                    for player_id in self.agents:
                        if self.PLAYERS[player_id]:
                            self.PLAYERS[player_id].won_game()

                    self.terminations = {a: True for a in self.agents}

            for k in self.kill_list:
                self.terminations[k] = True
                self.agents.remove(k)
                self.rewards[k] = 3 - len(self.agents)

            self.kill_list = []
            self._agent_selector.reinit(self.agents)

        # I think this if statement is needed in case all the agents die to the same minion round. a little sad.
        if len(self.agents) != 0:
            self.agent_selection = self._agent_selector.next()
            # self.observations[self.agent_selection] = self.game_observations[self.agent_selection].observation(self.agent_selection,
            #     self.PLAYERS[self.agent_selection], self.PLAYERS[self.agent_selection].action_vector)

        # Probably not needed but doesn't hurt?
        # self._deads_step_first()
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
