import numpy as np

from rlcard.games.base import Game
from .coup import Coup
from .constants import *
from .utils import *

class CoupGame(Game):
    ''' Represents a game of Coup in the RLCard environment
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
        return ActionEncoder(self.num_players).get_num_actions()

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
        if not self.is_over():
            assert state['game']['player_to_act'] == player_id
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

    def get_legal_actions(self):
        return self.coup.get_legal_actions()

    def get_winner(self):
        assert self.is_over()
        state = self.coup.get_state()
        return state['game']['winning_player']

    def get_perfect_information(self):
        return self.coup.get_state()
