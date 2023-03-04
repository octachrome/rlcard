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

import re
from itertools import combinations

from .constants import *

class ActionEncoder:
    def __init__(self, num_players):
        self.num_players = num_players
        self.id_to_action = (
            self.get_simple_actions()
            + ActionEncoder.get_exchange_actions(ALL_ROLES, 1)
            + ActionEncoder.get_exchange_actions(ALL_ROLES * 2, 2)
        )
        self.action_to_id = {a: i for i, a in enumerate(self.id_to_action)}

    def get_simple_actions(self):
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
        return len(self.id_to_action)

    def encode_action(self, action):
        return self.action_to_id[action]

    def decode_action(self, action_id):
        return self.id_to_action[action_id]

    @staticmethod
    def get_exchange_actions(roles, n):
        tuples = combinations(roles, n)
        # Sort each tuple and then de-duplicate: (duke,captain) and (captiain,duke) are equivalent
        choices = list(set([tuple(sorted(t)) for t in tuples]))
        return sorted([keep(c) for c in choices])

class GameView:
    ''' Provides a view of the game state from a given player's perspective,
    such that the current player is player 0 and every other player's
    hidden cards are masked
    '''
    def __init__(self, num_players):
        self.num_players = num_players

    def view_of_state(self, state):
        if state['game']['phase'] == GAME_OVER:
            return state
        player_id = state['game']['player_to_act']
        return {
            'players': [
                self.view_of_player_state(state, p)
                for p in range(self.num_players)
            ],
            'game': self.map_player_ids(state['game'], player_id)
        }

    def view_of_player_state(self, state, rel_player_id):
        player_id = (rel_player_id + state['game']['player_to_act']) % len(state['players'])
        player_state = state['players'][player_id]
        if rel_player_id == 0:
            return player_state
        else:
            return self.mask_player_state(player_state)

    def mask_player_state(self, player_state):
        return {
            key: (['hidden' for _ in val] if key == 'hidden' else val)
            for key, val in player_state.items()
        }

    def map_player_ids(self, state, player_id):
        if type(state) == dict:
            return {
                key: (
                    (val - player_id) % self.num_players
                    if key in ['whose_turn', 'player_to_act', 'target_player', 'blocking_player']
                    else self.map_player_ids(val, player_id)
                )
                for key, val in state.items()
            }
        elif type(state) == list:
            return [self.map_player_ids(i, player_id) for i in state]
        else:
            return state

    def view_of_actions(self, actions, player_id):
        return sorted([self.map_action_target(a, player_id) for a in actions])

    def map_action_target(self, action, player_id):
        m = re.match('.*?(\d+)', action)
        if m:
            target_player = int(m.group(1))
            return action[:m.start(1)] + str((target_player - player_id) % self.num_players)
        else:
            return action

    def unmap_action_target(self, action, player_id):
        return self.map_action_target(action, -player_id)
