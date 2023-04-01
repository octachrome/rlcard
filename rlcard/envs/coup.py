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

from rlcard.games.coup.game import CoupGame, ActionEncoder
from rlcard.games.coup.constants import *
from rlcard.games.coup.utils import GameView

from rlcard.envs import Env

DEFAULT_GAME_CONFIG = {
    'game_num_players': 4
}

class CoupEnv(Env):
    ''' RLCard environment for the game Coup

    Encodes game states and action states as feature vectors. Most of the logic
    is deferred to GameView, CoupObserver and ActionEncoder.
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
        self.game_view = GameView(self.num_players)
        self.observer = CoupObserver(self.num_players)
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
        return self.observer.get_action_features(action)

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
        obs = self.observer.observe_state(state_view)
        # Map the actions to the current player's perspective, then encode them
        raw_legal_actions = self.game.get_legal_actions()
        actions_view = self.game_view.view_of_actions(raw_legal_actions, state['game'].get('player_to_act'))
        legal_actions = OrderedDict(
            (self.action_encoder.encode_action(a), self.observer.get_action_features(a))
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

class CoupObserver:
    ''' Encodes the state of a game of Coup as a vector

    To experiment with alternative was of representing the game state,
    create a subclass or replacement of this class.
    '''
    def __init__(self, num_players):
        ''' Constructs an observer

        Args:
            num_players (int): number of players in the game
        '''
        self.num_players = num_players
        self.action_encoder = ActionEncoder(num_players)

    def observe_state(self, state):
        ''' Returns a representation of the game and players in the given state

        Args:
            state (dict): the state from a Coup instance

        Returns:
            (numpy.ndarray): a representation of the state
        '''
        obs = []
        obs.extend(self.observe_game(state['game']))
        for p in range(self.num_players):
            obs.extend(self.observe_player(state['players'][p], p))
        obs.extend(self.observe_available_cards(state))
        return np.array(obs)

    def get_state_size(self):
        ''' Returns the size of the vector returned by observe_state
        '''
        return len(self.observe_state({
            'game': {'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0},
            'players': [{'cash': 0, 'hidden': [], 'revealed': [], 'trace': []}] * self.num_players
        }))

    def observe_game(self, game_state):
        ''' Returns a vector representing the given game state

        Args:
            game_state (dict): the game state from a Coup instance

        Returns:
            (list of int): a representation of the game
        '''
        obs = [1 if game_state.get('whose_turn') == i else 0 for i in range(self.num_players)]
        action = game_state.get('action')
        obs += [1 if a == action else 0 for a in UNTARGETED_ACTIONS + TARGETED_ACTIONS]
        target_player = game_state.get('target_player')
        obs += [1 if p == target_player else 0 for p in range(self.num_players)]
        blocked_with = game_state.get('blocked_with')
        obs += [1 if blocked_with == r else 0 for r in BLOCKING_ROLES]
        blocking_player = game_state.get('blocking_player')
        obs += [1 if p == blocking_player else 0 for p in range(self.num_players)]
        obs += [1 if p == game_state['phase'] else 0 for p in PHASES]
        return obs

    def observe_player(self, player_state, player_id):
        ''' Returns a vector representing the given player

        Args:
            player_state (dict): a player state from a Coup instance
            player_id (int): id of the player to represent

        Returns:
            (list of int): a representation of the player

        Player 0 is assumed to be the player observing the state. His view of
        himself contains a lot more information than his view of the opponents.

        Player state includes his roles (for opponents just a count), cash and
        which roles have been claimed by opponents.
        '''
        influence = player_state['hidden']
        obs = self.encode_cash(player_state['cash'])
        if player_id == 0:
            # We can see our own roles
            obs += self.encode_role(influence[0] if len(influence) > 0 else None)
            obs += self.encode_role(influence[1] if len(influence) > 1 else None)
        else:
            # Encode the number of hidden cards of the opponent
            obs += [1 if len(influence) == i else 0 for i in range(3)]
            # Encode the historic actions of the player
            obs += self.encode_history(player_state['trace'])
        return obs

    def observe_available_cards(self, state):
        ''' Returns an encoding of all cards that the opponents could hold

        Args:
            state (dict): the state from a Coup instance

        Returns:
            (list of int): one-hot encoding of the count of each role card

        The encoding is constructed by subtracting from a deck the roles that
        the player himself holds and the roles that his opponents have revealed.

        In future it could be extended to include the roles that he has seen in
        the deck during the exchange action.
        '''
        role_counts = {r: 3 for r in ALL_ROLES}
        for r in state['players'][0]['hidden']:
            role_counts[r] -= 1
        for p in range(self.num_players):
            for r in state['players'][p]['revealed']:
                role_counts[r] -= 1
        # Encode the number of cards of each role (0..3)
        # Sorting dict items will arrange them by role name
        return [1 if c == i else 0 for _, c in sorted(role_counts.items()) for i in range(4)]

    def get_action_features(self, action):
        ''' Gets a feature vector describing the given action

        Args:
            action (str): the action to encode

        Returns:
            (numpy.ndarray): encoding of the action features

        All actions are one-hot encoded except for the KEEP action played at
        the end of an exchange, where the 1 or 2 roles are one-hot encoded
        separately.
        '''
        if action.startswith(KEEP):
            roles = keep_decode(action)
            return np.array(
                [0 for _ in self.action_encoder.get_simple_actions()]
                + self.encode_role(roles[0])
                + self.encode_role(roles[1] if len(roles) > 1 else None)
            )
        else:
            return np.array(
                [
                    1 if action == a else 0
                    for a in self.action_encoder.get_simple_actions()
                ]
                + [0] * len(ALL_ROLES) * 2
            )

    def get_action_size(self):
        ''' Returns the length of the vector returned by get_action_features
        '''
        return len(self.get_action_features(TAX))

    def encode_history(self, trace):
        ''' Encodes the action history of a player

        Args:
            trace (list of tuple): selected actions taken by a player

        Returns:
            (list of int) encoding of the action history

        For each role we encode whether or not the player has ever claimed
        that role, but we reset if the player subsequently revealed that role.
        (After revealing the card is lost or replaced.)
        '''
        claims = {r: 0 for r in ALL_ROLES}
        for t in trace:
            if t[0] == 'claim':
                claims[t[1]] = 1
            elif t[0] == 'reveal':
                claims[t[1]] = 0
        # Sorting dict items will arrange them by role name
        return [c for _, c in sorted(claims.items())]

    def encode_cash(self, cash):
        ''' Returns a one-hot encoding of a player's cash value

        Args:
            cash (int): the cash balance to encode

        Returns:
            (list of int) encoding of the case value

        All amounts of cash from 0 to 9 are encoded exactly. All amounts
        greator or equal to 10 are encoded in a single bit.
        '''
        return (
            [1 if cash == i else 0 for i in range(10)]
            + [1 if cash >= 10 else 0]
        )

    def encode_role(self, role):
        ''' Returns a one-hot encoding of a role

        Args:
            role (str): the role to encode

        Returns:
            (list of int) one-hot encoding of the role
        '''
        return [1 if role == r else 0 for r in ALL_ROLES]
