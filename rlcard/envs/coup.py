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

from collections import OrderedDict
import numpy as np

from coupml.constants import *
from coupml.observer import Observer
from coupml.utils import *
from coupml.view import PlayerView

from rlcard.games.coup.game import CoupGame
from rlcard.envs import Env


DEFAULT_GAME_CONFIG = {
    'game_num_players': 4
}


class CoupEnv(Env):
    ''' RLCard environment for the game Coup

    Encodes game states and action states as feature vectors. Most of the logic
    is deferred to PlayerView, Observer and ActionEncoder.
    '''
    def __init__(self, config):
        ''' Constructs a Coup environment

        See Env.__init__.
        '''
        self.name = 'coup'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = CoupGame()
        super().__init__(config)
        self.action_encoder = ActionEncoder(self.num_players)
        self.game_view = PlayerView(self.num_players)
        self.observer = Observer(self.num_players)
        self.state_shape = [[self.observer.get_state_size()] for _ in range(self.num_players)]
        self.action_shape = [[self.observer.get_action_size()] for _ in range(self.num_players)]

    def get_payoffs(self):
        ''' Get the payoffs of players.

        Returns:
            (list): A list of payoffs for each player.
        '''
        winner = self.game.get_winner()
        return np.array([1 if p == winner else -1 for p in range(self.num_players)])

    def get_perfect_information(self):
        ''' Get the perfect information of the current game state

        Returns:
            (dict): A dictionary of all the information in the current game state
        '''
        return self.game.get_perfect_information()

    def get_action_feature(self, action_id):
        ''' Get the feature vector for the given action

        Returns:
            (numpy.array): The action features
        '''
        action = self.action_encoder.decode_action(action_id)
        return np.array(self.observer.get_action_features(action))

    def _extract_state(self, state):
        ''' Extract useful information from state for RL

        Args:
            state (dict): The raw state

        Returns:
            (dict) that contains:
                obs (numpy.array) state in a useful form for learning
                legal_actions (dict) legal actions from this state, where the key is an id and the value is the features (or None)
                raw_legal_actions (list) legal actions as represented by the Game class (typically strings)
            it also typically contains the following optional entries, used by the human agents:
                raw_obs (dict) state as represented by the Game class
                action_record (list) a record of all actions taken since the start of the game
        '''
        # Map the state to the current player's perspective, then encode it
        state_view = self.game_view.view_of_state(state)
        obs = np.array(self.observer.observe_state(state_view))
        # Map the actions to the current player's perspective, then encode them
        raw_legal_actions = self.game.get_legal_actions()
        actions_view = self.game_view.view_of_actions(raw_legal_actions, state['game'].get('player_to_act'))
        legal_actions = OrderedDict(
            (self.action_encoder.encode_action(a), np.array(self.observer.get_action_features(a)))
            for a in actions_view
        )
        return dict(obs=obs, raw_obs=state, legal_actions=legal_actions, raw_legal_actions=raw_legal_actions)

    def _decode_action(self, action_id):
        ''' Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (string): The action that will be passed to the game engine.
        '''
        action = self.action_encoder.decode_action(action_id)
        return self.game_view.unmap_action_target(action, self.game.get_player_id())

    def _get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (list of int): A list of legal actions ids
        '''
        raw_legal_actions = self.game.get_legal_actions()
        actions_view = self.game_view.view_of_actions(raw_legal_actions, self.game.get_player_id())
        return [self.action_encoder.encode_action(a) for a in actions_view]
