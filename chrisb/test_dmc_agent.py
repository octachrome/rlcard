''' Prints the action probabilities for a trained DMC agent in a given Coup game state

This is for manually evaluating how an agent behaves in different situations.

First train a probabilistic DMC agent on Coup:

    python examples/run_dmc.py --env coup --xpid coup_dmc_prob --load_model --probabilistic --num_actors 2

Then run this script on the trained model:

    python test_dmc_agent.py --model my_model.pth
'''

import argparse
import torch

import rlcard
from rlcard.agents.random_agent import RandomAgent

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    default='chrisb/0_22454400.pth',
)
args = parser.parse_args()

dmc_agent = torch.load(args.model, map_location='cpu')
dmc_agent.set_device('cpu')

env = rlcard.make('coup')
env.set_agents([dmc_agent] + [RandomAgent(env.num_actions) for _ in range(3)])
env.reset()

def print_actions(raw_state):
  env.game.coup.reset_state(raw_state)
  state = env._extract_state(raw_state)
  action_keys, rewards = dmc_agent.predict(state)
  probs = dmc_agent.rewards_to_probs(rewards)
  for action_id, prob, reward in zip(action_keys, probs, rewards):
    action = env.action_encoder.decode_action(action_id)
    print(action, prob, reward)

# Coup is marginally the most likely action when it is affordable
print('Agent can afford to coup:')
print_actions({
  'game': {'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0},
  'players': [
      {'cash': 7, 'hidden': ['contessa'], 'revealed': ['assassin'], 'trace': []},
      {'cash': 2, 'hidden': ['duke'], 'revealed': ['contessa'], 'trace': []},
      {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': [], 'trace': []},
      {'cash': 2, 'hidden': ['assassin', 'duke'], 'revealed': [], 'trace': []}
  ]
})
print()

# Tax is marginally the most likely action when the agent has a duke
print('Agent has a duke:')
print_actions({
  'game': {'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0},
  'players': [
      {'cash': 2, 'hidden': ['duke'], 'revealed': ['assassin'], 'trace': []},
      {'cash': 2, 'hidden': ['duke'], 'revealed': ['contessa'], 'trace': []},
      {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': [], 'trace': []},
      {'cash': 2, 'hidden': ['assassin', 'duke'], 'revealed': [], 'trace': []}
  ]
})
print()
