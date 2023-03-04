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
    def __init__(self, np_random):
        ''' Initialize a dealer
        '''
        self.np_random = np_random
        self.deck = ALL_ROLES * 3
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, n):
        ''' Deal some cards from the deck
        '''
        return [self.deck.pop() for _ in range(n)]

    def replace_cards(self, cards):
        ''' Return some cards to the deck
        '''
        self.deck += cards
        self.shuffle()

    def choose(self, items):
        ''' The dealer also arbitrates over random choices
        '''
        i = self.np_random.choice(len(items))
        return items[i]
