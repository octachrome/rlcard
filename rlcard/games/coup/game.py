import numpy as np

from rlcard.games.base import Game
from .coup import Coup

class CoupGame(Game):
    def __init__(self, num_players=4):
        self.np_random = np.random.RandomState()
        self.num_players = num_players

    def get_num_players(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game

        Note: Must be implemented in the child class.
        '''
        return self.num_players

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions.

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id

        Note: Must be implemented in the child class.
        '''
        self.coup = Coup(self.num_players, self.np_random)
        self.coup.init_game()

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over

        Note: Must be implemented in the child class.
        '''
        return False

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            Status (bool): check if the step back is success or not

        Note: Not needed unless allow_step_back=True is passed to rlcard.make
        '''
        raise NotImplementedError
