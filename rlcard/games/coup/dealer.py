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

from .constants import ALL_ROLES

class CoupDealer:
    ''' Dealer for Coup game
    '''
    def __init__(self, np_random):
        ''' Constructs a dealer

        Args:
            np_random (numpy.random.RandomState): used for randomized decisions
        '''
        self.np_random = np_random
        self.deck = ALL_ROLES * 3
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, n):
        ''' Deals some cards from the deck

        Args:
            n (int): Number of cards to deal

        Returns:
            (list of str) n cards from the deck
        '''
        return [self.deck.pop() for _ in range(n)]

    def replace_cards(self, cards):
        ''' Returns some cards to the deck

        Args:
            cards (list of str): cards to replace into the deck
        '''
        self.deck += cards
        self.shuffle()

    def choose(self, items):
        ''' Picks one item at random from a list

        Args:
            items (list): items to choose from

        Returns:
            an item randomly chosen from the list

        Used to randomly select which player gets to block, in the event that
        more than one player tried to block foreign aid.
        '''
        i = self.np_random.choice(len(items))
        return items[i]

    def get_state(self):
        return {'deck': list(self.deck)}

    def reset_state(self, state):
        ''' Resets the deck to a given state

        Args:
            state (dict): a state dictionary such as returned by get_state
        '''
        self.deck = list(state['deck'])
