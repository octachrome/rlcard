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
