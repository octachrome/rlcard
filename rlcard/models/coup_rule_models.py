''' Coup rule models
'''

import numpy as np
from coupml.rule_ai import RuleAI

import rlcard
from rlcard.models.model import Model

class CoupRuleAgentV1(object):
    ''' Coup Rule agent version 1
    '''

    def __init__(self):
        self.use_raw = True
        self.np_random = np.random.RandomState()

    def step(self, state):
        ''' Predict the action given raw state.

        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        ai = RuleAI(state, self.np_random)
        action = ai.get_action()
        assert action in legal_actions
        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

class CoupRuleModelV1(Model):
    ''' Coup Rule Model version 1
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('coup')

        rule_agent = CoupRuleAgentV1()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True
