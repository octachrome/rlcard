# Copyright 2023 Chris Brown
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from rlcard.games.base import Game
from .coup import Coup
from .constants import *
from .utils import *

class CoupGame(Game):
    ''' Represents a game of Coup in the RLCard environment
    '''
    def __init__(self, num_players=4):
        self.np_random = np.random.RandomState()
        self.num_players = num_players

    def get_num_players(self):
        ''' Return the number of players in the game

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    def get_num_actions(self):
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions.
        '''
        return ActionEncoder(self.num_players).get_num_actions()

    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        self.coup = Coup(self.num_players, self.np_random)
        self.coup.init_game()
        player_id = self.get_player_id()
        return (self.get_state(player_id), player_id)

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        return self.coup.is_game_over()

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.coup.player_to_act()

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = self.coup.get_state()
        if not self.is_over():
            assert state['game']['player_to_act'] == player_id
        return state

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''
        self.coup.play_action(action)
        player_id = self.get_player_id()
        return (self.get_state(player_id), player_id)

    def get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (list of str): A list of legal action names
        '''
        return self.coup.get_legal_actions()

    def get_winner(self):
        ''' Get the id of the player who won the game

        Returns:
            (int): id of the player who won the game
        '''
        assert self.is_over()
        state = self.coup.get_state()
        return state['game']['winning_player']

    def get_perfect_information(self):
        ''' Get the perfect information of the current game state

        Returns:
            (dict): A dictionary of all the information in the current game state
        '''
        return self.coup.get_state()

class ActionEncoder:
    ''' Encodes Coup actions using numerical ids
    '''
    def __init__(self, num_players):
        ''' Construcs an action encoder

        Args:
            num_players (int): the number of players in the game
        '''
        self.num_players = num_players
        # A list which functions as a map from id to action name
        self.id_to_action = (
            self.get_simple_actions()
            + get_keep_actions(ALL_ROLES, 1)
            + get_keep_actions(ALL_ROLES * 2, 2)
        )
        # A dict which maps from action name to id
        self.action_to_id = {a: i for i, a in enumerate(self.id_to_action)}

    def get_simple_actions(self):
        ''' Returns the simple actions in the game

        Simple actions are those which are one-hot encoded, which includes every
        action except for the KEEP action played at the end of an exchange.

        Returns:
            (list of str): the simple actions in the game
        '''
        return (
            UNTARGETED_ACTIONS
            + [
                f'{a}:{p}'
                for a in TARGETED_ACTIONS
                # From the point of view of player 0,
                # can only target players 1-3
                for p in range(1, self.num_players)
            ]
            + [block(r) for r in BLOCKING_ROLES]
            + [reveal(r) for r in ALL_ROLES]
            + [CHALLENGE, PASS]
        )

    def get_num_actions(self):
        ''' Returns the total number of possible actions
        '''
        return len(self.id_to_action)

    def encode_action(self, action):
        ''' Encodes the given action as an id

        Args:
            action (str): the action to encode

        Returns:
            (int): the id of the action
        '''
        return self.action_to_id[action]

    def decode_action(self, action_id):
        ''' Decodes the given action id to a string

        Args:
            action_id (int): the action id to decode

        Returns:
            (str): the action name
        '''
        return self.id_to_action[action_id]
