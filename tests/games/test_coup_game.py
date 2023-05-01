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

import os
import unittest

from rlcard.games.coup.game import CoupGame
from rlcard.utils import seeding


class TestCoupGame(unittest.TestCase):
    ''' Tests for Coup

    CoupML has its own test suite for the game rules, so here
    we just test that the methods of TestCoupGame are working.
    '''

    def setUp(self):
        self.game = CoupGame(4)
        self.game.np_random = seeding.np_random(3)[0]
        self.game.init_game()

    def test_get_num_players(self):
        self.assertEqual(self.game.get_num_players(), 4)

    def test_is_over(self):
        self.assertEqual(self.game.is_over(), False)

    def test_get_player_id(self):
        self.assertEqual(self.game.get_player_id(), 1)

    def test_get_state(self):
        self.assertEqual(self.game.get_state(1), {
            'dealer': {'deck': ['contessa', 'contessa', 'ambassador', 'assassin', 'duke', 'ambassador', 'duke']},
            'game': {'phase': 'start_of_turn', 'player_to_act': 1, 'whose_turn': 1},
            'players': [
                {'cash': 2, 'hidden': ['assassin', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['assassin', 'contessa'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['captain', 'duke'], 'revealed': [], 'trace': []}
            ]
        })

    def test_step(self):
        state, player_id = self.game.step('income')
        self.assertEqual(player_id, 2)
        self.assertEqual(state, {
            'dealer': {'deck': ['contessa', 'contessa', 'ambassador', 'assassin', 'duke', 'ambassador', 'duke']},
            'game': {'phase': 'start_of_turn', 'player_to_act': 2, 'whose_turn': 2},
            'players': [
                {'cash': 2, 'hidden': ['assassin', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 3, 'hidden': ['assassin', 'contessa'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['captain', 'duke'], 'revealed': [], 'trace': []}
            ]
        })

    def test_get_legal_actions(self):
        self.assertEqual(self.game.get_legal_actions(), [
            'exchange',
            'foreign_aid',
            'income',
            'steal:0',
            'steal:2',
            'steal:3',
            'tax'
        ])

    def test_get_perfect_information(self):
        self.assertEqual(self.game.get_perfect_information(), self.game.get_state(1))

    def test_complete_game(self):
        np_random = seeding.np_random(10)[0]
        while not self.game.is_over():
            actions = self.game.get_legal_actions()
            action = np_random.choice(actions)
            self.game.step(action)
        self.assertEqual(self.game.get_winner(), 3)


if __name__ == '__main__':
    unittest.main()
