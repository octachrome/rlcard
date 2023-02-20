import numpy as np

from rlcard.games.base import Game
from .coup import Coup
from .constants import *

class CoupGame(Game):
    ''' Represents a game of Coup in the RLCard environment

    Exposes a view of the game state from the perspective of any given player
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
        return self.get_num_simple_actions() + self.get_num_exchange_actions()

    def get_num_simple_actions(self):
        return (
            len(TARGETED_ACTIONS) * (self.num_players - 1)  # Targeted initial actions
            + len(UNTARGETED_ACTIONS)                       # Untargeted initial actions
            + len(BLOCKING_ROLES)                           # Blocks (duke, captain, ambassador, contessa)
            + 2                                             # Challenge, allow
        )

    def get_num_exchange_actions(self):
        return (
            len(ALL_ROLES)                          # Number of ways to choose 1 of 5 cards
            + sum(range(1, len(ALL_ROLES) + 1))     # Number of ways to choose 2 of 5 cards
        )

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
        assert state['game']['player_to_act'] == player_id
        # Map player state so that the current player is player 0
        # and every other player's hidden cards are masked
        return {
            'players': [
                self.get_player_state(state, p)
                for p in range(self.num_players)
            ],
            'game': self.map_player_ids(state['game'], player_id)
        }

    def get_player_state(self, state, rel_player_id):
        player_id = (rel_player_id + state['game']['player_to_act']) % self.num_players
        player_state = state['players'][player_id]
        if rel_player_id == 0:
            return player_state
        else:
            return self.mask_player_state(player_state)

    def mask_player_state(self, player_state):
        return {
            key: (['hidden' for i in val] if key == 'hidden' else val)
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
