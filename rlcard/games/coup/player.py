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

class CoupPlayer:
    ''' Represents a player in a game of Coup
    '''
    def __init__(self, player_id):
        ''' Constructs a player

        Args:
            player_id (int): id of the player
        '''
        self.player_id = player_id
        self.hidden = []
        self.revealed = []
        self.cash = 2
        self.trace = []

    def get_state(self):
        ''' Returns a dictionary describing the state of the player

        Returns:
            (dict) that contains:
                cash (int): number of credits the player has
                hidden (list of str): hidden role cards of the player
                revealed (list of str): revealed role cards of the player
                trace (list of tuple): selected actions the player has taken which hint at his roles
        '''
        return {
            'cash': self.cash,
            'hidden': sorted(self.hidden),
            'revealed': sorted(self.revealed),
            'trace': list(self.trace)
        }

    def reset_state(self, state):
        ''' Resets the player to a given state

        Args:
            state (dict): a state dictionary such as returned by get_state
        '''
        self.cash = state['cash']
        self.hidden = list(state['hidden'])
        self.revealed = list(state['revealed'])
        self.trace = list(state.get('trace', []))
