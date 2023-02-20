''' Game-related base classes
'''

import numpy as np

class Game(object):
    def __init__(self):
        ''' Your Env constructor should construct your Game class so this constructor
        can take any args you choose

        Note: the Env constructor assigns game.allow_step_back according to whether
        the value passed to rlcard.make.

        Note: the Env.seed function re-assigns game.np_random, and this should be
        used for any random number generation.
        '''
        self.np_random = np.random.RandomState()

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players

        Note: Currently only called in some games (see supported_envs in env.py)
        '''
        raise NotImplementedError

    def get_num_players(self):
        ''' Return the number of players in the game

        Returns:
            (int): The number of players in the game

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

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
        raise NotImplementedError

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

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
        return False

class Card:
    '''
    Card stores the suit and rank of a single card

    Note:
        The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ] meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]
    '''
    suit = None
    rank = None
    valid_suit = ['S', 'H', 'D', 'C', 'BJ', 'RJ']
    valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    def __init__(self, suit, rank):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''
        return self.rank + self.suit

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit+self.rank
