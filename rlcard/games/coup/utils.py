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

''' Utilities for Coup game
'''

import re
from itertools import combinations

from .constants import *

def get_keep_actions(roles, n):
    ''' Gets all the possible actions for keeping n of the given roles

    Args:
        roles (list of str): roles from which to choose
        n (int): how many roles to choose

    Returns:
        (list of tuple): all possible KEEP actions

    At the end of an exchange action, the player may choose to keep 1 or 2
    cards from those drawn. Tuples are sorted and de-duplicated so that
    (duke, captain) is the same as (captain, duke).
    '''
    tuples = combinations(roles, n)
    choices = list(set([tuple(sorted(t)) for t in tuples]))
    return sorted([keep(c) for c in choices])

class GameView:
    ''' Provides a view of the game state from the current player's perspective

    In the view, the current player is player 0 and every other player's hidden
    cards are masked.

    Note: the 'drawn_cards' value is not masked because the only player who may
    act in this state is the player who is performing the exchange.
    '''
    def __init__(self, num_players):
        ''' Constructs a game view

        Args:
            num_players (int): number of players in the game
        '''
        self.num_players = num_players

    def view_of_state(self, state):
        ''' Returns a view of the game state from the perspective of the current player

        Args:
            state (dict): the game+player state

        Returns:
            (dict): the state from the perspective of the current player
        '''
        if state['game']['phase'] == GAME_OVER:
            # No player to act in this state, so cannot map
            return state
        return {
            'players': [
                self.view_of_player_state(state, p)
                for p in range(self.num_players)
            ],
            'game': self.view_of_game_state(state['game'])
        }

    def view_of_player_state(self, state, rel_player_id):
        ''' Gets the state of the given player relative to the current player

        Args:
            state (dict): the game+player state
            rel_player_id (int): relative id of the player whose state to return

        Returns:
            (dict): state of the requested player with hidden information masked

        If rel_player_id is 0, the state of the current player is returned; if
        rel_player_id is 1, the state of the player whose turn will be next is
        returned, etc.
        '''
        player_id = (rel_player_id + state['game']['player_to_act']) % len(state['players'])
        player_state = state['players'][player_id]
        if rel_player_id == 0:
            return player_state
        else:
            return self.mask_player_state(player_state)

    def mask_player_state(self, player_state):
        ''' Mask out information that is hidden from the current player

        Args:
            player_state (dict): state of an opponent player

        Returns:
            (dict): player state with hidden information masked

        The only hidden information is the hidden roles of the player. The
        number of roles is visible, but the roles themselves are replaced
        with the string 'hidden'.
        '''
        return {
            key: (['hidden' for _ in val] if key == 'hidden' else val)
            for key, val in player_state.items()
        }

    def view_of_game_state(self, state):
        ''' Gets the state of the game relative to the current player

        Args:
            state (dict): the game state
            player_id (int): the current player

        Returns:
            (dict): the game state with player ids remapped

        Player ids are remapped so that the current player appears as player 0,
        the player whose turn will be next appears as player 1, etc.
        '''
        player_id = state['player_to_act']
        return {
            key: (
                (val - player_id) % self.num_players
                if key in ['whose_turn', 'player_to_act', 'target_player', 'blocking_player']
                else val
            )
            for key, val in state.items()
        }

    def view_of_actions(self, actions, player_id):
        ''' Maps the given actions from the perspective of the given player

        Args:
            actions (list of str): actions to map
            player_id (int): the current player

        Targeted actions will be renamed such that 'coup:1' targets the player
        who follows the current player in turn order, etc. Untargeted actions
        are unaffected.
        '''
        return sorted([self.map_action_target(a, player_id) for a in actions])

    def map_action_target(self, action, player_id):
        ''' Maps the given action from the perspective of the given player

        Args:
            action (str): action to map
            player_id (int): the current player

        Targeted actions will be renamed such that 'coup:1' targets the player
        who follows the current player in turn order, etc. Untargeted actions
        are unaffected.
        '''
        m = re.match('.*?(\d+)', action)
        if m:
            target_player = int(m.group(1))
            return action[:m.start(1)] + str((target_player - player_id) % self.num_players)
        else:
            return action

    def unmap_action_target(self, action, player_id):
        ''' Maps the given action to the global perspective

        Args:
            action (str): action from the perspective of the current player
            player_id (int): id of the current player

        Returns:
            (str): the action from the global perspective

        If an agent wants to steal from the player who is next in turn order,
        the agent would play the action 'steal:1'. If the agent is player 2
        in the game, this action is mapped to 'steal:3', targeting the player
        who follows player 2 in turn order.
        '''
        return self.map_action_target(action, -player_id)
