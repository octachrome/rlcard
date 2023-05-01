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
import numpy as np
from coupml.observer import Observer

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.dmc_agent.model import DMCAgent
from .determism_util import is_deterministic


''' Set this env var to a large number to run many games and find bugs '''
EXHAUSTIVE_RUNS = int(os.environ.get('EXHAUSTIVE_RUNS', '1'))


class CoupEnvTest(unittest.TestCase):
    ''' Tests for the Coup RLCard environment
    '''
    def test_reset_and_extract_state(self):
        ''' Tests that the initial state of a new game is as expected
        '''
        env = rlcard.make('coup', config={'seed': 0})
        state, _ = env.reset()
        self.assertEqual(state['raw_obs'], {
            'game': {'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0},
            'players': [
                {'cash': 2, 'hidden': ['ambassador', 'assassin'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['assassin', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['contessa', 'contessa'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['ambassador', 'duke'], 'revealed': [], 'trace': []}
            ],
            'dealer': {
                'deck': [
                    'duke',
                    'assassin',
                    'captain',
                    'duke',
                    'ambassador',
                    'contessa',
                    'captain'
                ]
            }
        })
        self.assertEqual(state['obs'].size, Observer(4).get_state_size())
        self.assertEqual(state['raw_legal_actions'], [
            'exchange',
            'foreign_aid',
            'income',
            'steal:1',
            'steal:2',
            'steal:3',
            'tax'
        ])
        self.assertEqual(len(state['raw_legal_actions']), len(state['legal_actions']))
        self.assertEqual(state['legal_actions'][0].size, Observer(4).get_action_size())

    def test_is_deterministic(self):
        ''' Tests that Coup implementation is deterministic
        '''
        self.assertTrue(is_deterministic('coup'))

    def test_step(self):
        ''' Tests taking a single step through the game
        '''
        env = rlcard.make('coup', {'seed': 0})
        state0, player0 = env.reset()
        self.assertEqual(state0['raw_obs']['game']['player_to_act'], player0)
        state1, player1 = env.step(0)
        self.assertEqual(state1['raw_obs']['game']['player_to_act'], player1)
        self.assertNotEqual(player0, player1)

    def test_run(self):
        ''' Tests making a run through a complete game
        '''
        env = rlcard.make('coup', {'seed': 0})
        env.set_agents([RandomAgent(env.num_actions) for _ in range(env.num_players)])
        trajectories, payoffs = env.run(is_training=False)
        self.assertEqual(sorted(payoffs), [-1, -1, -1, 1])
        self.assertEqual(len(trajectories), 4)
        trajectories, payoffs = env.run(is_training=True)
        self.assertEqual(len(trajectories), 4)
        self.assertEqual(sorted(payoffs), [-1, -1, -1, 1])

    def test_run_dmc(self):
        ''' Tests that a probabilistic DMC agent runs with no errors
        '''
        env = rlcard.make('coup', {'seed': 0})
        env.set_agents([create_dmc_agent(env, p) for p in range(env.num_players)])
        trajectories, payoffs = env.run(is_training=False)
        self.assertEqual(sorted(payoffs), [-1, -1, -1, 1])
        self.assertEqual(len(trajectories), 4)
        trajectories, payoffs = env.run(is_training=True)
        self.assertEqual(len(trajectories), 4)
        self.assertEqual(sorted(payoffs), [-1, -1, -1, 1])

    def test_exhaustive(self):
        for seed in range(EXHAUSTIVE_RUNS):
            try:
                env = rlcard.make('coup', config={'seed': seed})
                np_random = np.random.RandomState(seed)
                state, _ = env.reset()
                while not env.is_over():
                    legals = list(state['legal_actions'].keys())
                    action_idx = np_random.randint(0, len(legals))
                    action = legals[action_idx]
                    state, _ = env.step(action)
            except Exception as e:
                raise Exception(f'Failed exhaustive test with seed {seed}', e)


def create_dmc_agent(
    env, player_id,
    mlp_layers=[512,512,512,512,512],
    exp_epsilon=0.01,
    device='cpu',
):
    return DMCAgent(
        env.state_shape[player_id],
        env.action_shape[player_id],
        mlp_layers,
        exp_epsilon,
        device,
        if_probabilistic=True
    )


if __name__ == '__main__':
    unittest.main()
