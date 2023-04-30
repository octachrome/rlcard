import os
import unittest
import numpy as np

from rlcard.games.coup.coup import Coup, IllegalAction
from rlcard.games.coup.utils import GameView
from rlcard.games.coup.constants import *
from rlcard.utils import seeding

''' Set this env var to a large number to run many games and find bugs '''
RESET_STATE_RUNS = int(os.environ.get('RESET_STATE_RUNS', '1000'))

class TestDealer:
    ''' A dealer for Coup which deals the given deck and never shuffles
    '''
    def __init__(self, deck):
        self.deck = deck

    def deal_cards(self, n):
        return [self.deck.pop(0) for _ in range(n)]

    def replace_cards(self, cards):
        self.deck += cards

    def choose(self, items):
        return items[0]

    def get_state(self):
        return None

class Helper(unittest.TestCase):
    ''' Superclass for Coup tests with utility functions
    '''

    def setUp(self):
        self.game = Coup(4, np.random)
        d = TestDealer(self.get_deck())
        self.game.init_game(d)

    def get_deck(self):
        ''' Tests can override this function to set the contents of the deck

        By default, every player has duke/captain, and the remaining deck contains assassins.
        '''
        return [DUKE, CAPTAIN] * 4 + [ASSASSIN, ASSASSIN]

    def assert_state(self, state):
        ''' Asserts that the current game state matches the given dict
        '''
        self.assertEqual(self.game.get_state()['game'], state)

    def assert_legal_actions(self, actions):
        ''' Asserts that the current legal actions match the given list
        '''
        self.assertEqual(self.game.get_legal_actions(), actions)

    def assert_cash(self, player_id, cash):
        ''' Asserts that a player's cash is the given value
        '''
        self.assertEqual(self.game.players[player_id].cash, cash)

    def assert_hidden(self, player_id, roles):
        ''' Asserts that a player's hidden influence matches the given list
        '''
        self.assertEqual(set(self.game.players[player_id].hidden), set(roles))

    def assert_revealed(self, player_id, roles):
        ''' Asserts that a player's revealed influence matches the given list
        '''
        self.assertEqual(set(self.game.players[player_id].revealed), set(roles))

    def assert_trace(self, player_id, trace):
        ''' Asserts that a player's history matches the given list
        '''
        self.assertEqual(self.game.players[player_id].trace, trace)

class IncomeTest(Helper):
    def test_income(self):
        ''' Test that the income action works as expected
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(INCOME)
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        self.assert_cash(0, 3)

class ForeignAidTest(Helper):
    ''' Tests for the foreign aid action
    '''

    def test_allowed(self):
        ''' Test that the foreign aid action works as expected when not blocked
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(FOREIGN_AID)
        # Player 0 played, the other three get a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody blocked, the action is allowed
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 should get his foreign aid
        self.assert_cash(0, 4)

    def setup_blocked(self):
        ''' Common setup for tests where foreign aid is played and more than
        one opponent wants to block
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(FOREIGN_AID)
        # Player 0 played, the other three get a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(block(DUKE))
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(block(DUKE))
        # Players 1 and 3 blocked, one is chosen at random, then the other three players get a chance to challenge the block
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        # Player 2 challenged, player 1 must reveal whether he has a duke
        self.assert_state({'phase': 'prove_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 1})

    def test_blocked(self):
        ''' Tests that if the block succeeds, no foreign aid is paid
        '''
        self.setup_blocked()
        self.game.play_action(reveal(DUKE))
        # Player 1 did have the duke, player 2 must reveal
        self.assert_state({'phase': 'incorrect_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(reveal(CAPTAIN))
        # End of turn
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 should not get his foreign aid, it was blocked
        self.assert_cash(0, 2)
        # Player 1 should have a replacement role
        self.assert_hidden(1, [CAPTAIN, ASSASSIN])
        self.assert_revealed(1, [])
        # Player 2 should have a revealed role
        self.assert_revealed(2, [CAPTAIN])
        self.assert_hidden(2, [DUKE])
        # Player 1 has claimed duke and revealed a duke
        self.assert_trace(1, [('claim', 'duke'), ('reveal', 'duke')])

    def test_failed_block(self):
        ''' Tests that if the block fails, foreign aid is paid
        '''
        self.setup_blocked()
        self.game.play_action(reveal(CAPTAIN))
        # Player 1 did not reveal the duke, action is performed and turn ends
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 should get his foreign aid
        self.assert_cash(0, 4)
        # Player 1 should have a revealed role
        self.assert_revealed(1, [CAPTAIN])
        self.assert_hidden(1, [DUKE])
        # Player 2 should still have his original roles
        self.assert_hidden(2, [DUKE, CAPTAIN])
        self.assert_revealed(2, [])
        # Player 1 has claimed duke, revealed a captain, and failed a duke bluff
        self.assert_trace(1, [('claim', 'duke'), ('reveal', 'captain'), ('lost_challenge', 'duke')])

class StealTest(Helper):
    ''' Tests for the steal action
    '''

    def test_multiple_challenges(self):
        ''' Tests that if multiple players challenge incorrectly, they all lose
        an influence
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(STEAL + ':2')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(CHALLENGE)
        # Players 1 and 3 challenged, player 0 must reveal whether he has a captain
        self.assert_state({'phase': 'prove_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(reveal(CAPTAIN))
        # Player 0 had the role, so players 1 and 3 must reveal
        self.assert_state({'phase': 'incorrect_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(reveal(DUKE))
        self.assert_state({'phase': 'incorrect_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(reveal(DUKE))
        # Player 2 now gets the chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        # Player 2 passes, so the steal takes place
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Money has changed hands
        self.assert_cash(0, 4)
        self.assert_cash(2, 0)
        # Players 1 and 3 have lost influence
        self.assert_revealed(1, [DUKE])
        self.assert_hidden(1, [CAPTAIN])
        self.assert_revealed(1, [DUKE])
        self.assert_hidden(1, [CAPTAIN])

    def test_blocked(self):
        ''' Tests that if a steal is blocked, the cash is not stolen
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(STEAL + ':2')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 2 now gets the chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(block(AMBASSADOR))
        # Player 2 blocks, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'steal', 'target_player': 2, 'blocked_with': 'ambassador', 'blocking_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'steal', 'target_player': 2, 'blocked_with': 'ambassador', 'blocking_player': 2, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'steal', 'target_player': 2, 'blocked_with': 'ambassador', 'blocking_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        # Nobody challenged, the steal is blocked
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Money has not changed
        self.assert_cash(0, 2)
        self.assert_cash(2, 2)

class TaxTest(Helper):
    def test_allowed(self):
        ''' Tests that if the tax action works correctly
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(TAX)
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'tax', 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'tax', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'tax', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 2 gets the money
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        self.assert_cash(0, 5)

class AssassinTest(Helper):
    ''' Tests for the assassin action
    '''

    def setUp(self):
        ''' Sets up the current player with enough cash to assassinate
        '''
        super().setUp()
        self.game.players[0].cash = 3

    def test_allowed(self):
        ''' Tests that the assassinate action works correctly when not blocked
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + ':1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 1 gets a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        # Player 1 did not block, so must reveal
        self.assert_state({'phase': 'direct_attack', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(reveal(DUKE))
        # End of turn
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 pays 3
        self.assert_cash(0, 0)
        # Player 1 loses an influence
        self.assert_hidden(1, [CAPTAIN])
        self.assert_revealed(1, [DUKE])

    def test_challenged(self):
        ''' Tests that a player does not pay for an assassination if it is
        correctly challenged
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + ':1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Player 2 challenged, player 0 must reveal whether he has an assassin
        self.assert_state({'phase': 'prove_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(reveal(DUKE))
        # Player 0 did not have an assassin, so the card is lost, turn is over
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 does not pay, since the action was challenged
        self.assert_cash(0, 3)
        # Player 0 loses an influence
        self.assert_hidden(0, [CAPTAIN])
        self.assert_revealed(0, [DUKE])

    def test_blocked(self):
        ''' Tests that a player pays for an assassination even if it is blocked
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + ':1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 1 gets a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(block(CONTESSA))
        # Player 1 blocked, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        # Nobody challenged, the action does not take place
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 pays 3
        self.assert_cash(0, 0)

    def test_failed_block(self):
        ''' Tests that a player loses two influence and dies if he fails to
        block an assassination
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + ':1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 1 gets a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(block(CONTESSA))
        # Player 1 blocked, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        # Player 0 challenged, player 1 must must reveal whether he has a contessa
        self.assert_state({'phase': 'prove_challenge', 'action': 'assassinate', 'target_player': 1, 'blocked_with': 'contessa', 'blocking_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(reveal(CAPTAIN))
        # Player 1 did not have a contessa, so the assassination continues
        self.assert_state({'phase': 'direct_attack', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(reveal(DUKE))
        # Player 1 reveals his last card and is out of the game, play passes to player 2
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 2, 'player_to_act': 2})
        # Player 0 pays
        self.assert_cash(0, 0)
        # Player 1 has no influence
        self.assert_hidden(1, [])
        self.assert_revealed(1, [CAPTAIN, DUKE])

class ExchageTest(Helper):
    def test_exchange(self):
        ''' Tests that the exchange action works correctly
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(EXCHANGE)
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'exchange', 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'exchange', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'exchange', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 0 chooses his new cards
        self.assert_state({'phase': 'choose_new_roles', 'action': 'exchange', 'drawn_roles': ['assassin', 'assassin'], 'whose_turn': 0, 'player_to_act': 0})
        # Try some illegal choices: not enough roles
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(keep([DUKE]))
        self.assertEqual(str(cm.exception), 'Must choose 2 roles')
        # Wrong roles
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(keep([DUKE, CONTESSA]))
        self.assertEqual(str(cm.exception), 'Chosen roles are not available')
        # Too many of the same role
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(keep([DUKE, DUKE]))
        self.assertEqual(str(cm.exception), 'Chosen roles are not available')
        # Make a valid choice and end the turn
        self.game.play_action(keep([DUKE, ASSASSIN]))
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 should have new roles
        self.assert_hidden(0, [DUKE, ASSASSIN])

class CoupTest(Helper):
    ''' Tests for the coup action
    '''

    def setUp(self):
        ''' Sets up the current player with enough cash to coup, and sets up an
        opponent with only one influence left
        '''
        super().setUp()
        self.game.players[0].cash = 7
        self.game.reveal_role(1, DUKE)

    def test_coup(self):
        ''' Tests that the coup action works correctly
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(COUP + ':3')
        # Player 3 must reveal
        self.assert_state({'phase': 'direct_attack', 'action': 'coup', 'target_player': 3, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(reveal(DUKE))
        # Turn ends
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 3 has lost influence
        self.assert_revealed(3, [DUKE])
        self.assert_hidden(3, [CAPTAIN])
        # Player 0 pays 7
        self.assert_cash(0, 0)

    def test_coup_to_death(self):
        ''' Tests that the dead player skips his turn after being eliminated
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(COUP + ':1')
        # Player 1 must reveal his last card
        self.assert_state({'phase': 'direct_attack', 'action': 'coup', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(reveal(CAPTAIN))
        # Turn skips over dead player to player 2
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 2, 'player_to_act': 2})
        # Player 1 has no influence
        self.assert_revealed(1, [DUKE, CAPTAIN])
        self.assert_hidden(1, [])

class TooPoorTest(Helper):
    def test_cannot_afford_coup(self):
        ''' Tests that a player cannot coup if he does not have enough cash
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(COUP + ':3')
        self.assertEqual(str(cm.exception), 'Cannot afford to coup')

class LastPlayerDies(Helper):
    ''' Tests for end of game
    '''

    def setUp(self):
        ''' Sets up a game where only two players remain and the opponent has
        only one influence left
        '''
        super().setUp()
        self.game.players[0].cash = 7
        self.game.reveal_role(1, DUKE)
        self.game.reveal_role(1, CAPTAIN)
        self.game.reveal_role(2, DUKE)
        self.game.reveal_role(2, CAPTAIN)
        # Player 3 has one role left
        self.game.reveal_role(3, DUKE)

    def test_last_player_dies(self):
        ''' Tests that the game ends when the opponent loses his last influence
        '''
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(COUP + ':3')
        # Player 3 must reveal
        self.assert_state({'phase': 'direct_attack', 'action': 'coup', 'target_player': 3, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(reveal(CAPTAIN))
        # Game over
        self.assert_state({'phase': 'game_over', 'winning_player': 0})

class StealTargetDiesTest(Helper):
    def setUp(self):
        ''' Sets up a game where an oppoinent has only one influence remaining
        '''
        super().setUp()
        self.game.reset_state({
            'game': {'phase': 'start_of_turn', 'whose_turn': 2, 'player_to_act': 2},
            'players': [
                {'cash': 2, 'hidden': ['assassin'], 'revealed': ['duke']},
                {'cash': 2, 'hidden': ['duke'], 'revealed': ['contessa']},
                {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': []},
                {'cash': 2, 'hidden': ['assassin', 'duke'], 'revealed': []}
            ]
        })

    def test_steal_target_dies(self):
        ''' Tests that if a player being stolen from dies while challenging the
        captain, no money is stolen

        This behaviour is ambiguous but it was easiest to implement.
        '''
        # Player 2 steals from player 1
        self.game.play_action('steal:1')
        self.game.play_action('pass')
        self.game.play_action('pass')
        # Player 1 challenges
        self.game.play_action('challenge')
        # Player 0 proves the captain
        self.game.play_action('reveal:captain')
        # Player 1 reveals last influence and dies
        self.game.play_action('reveal:duke')
        # Player 0 cannot steal from the dead player
        # (he is not alive to block), so end of turn
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 3, 'player_to_act': 3})

class AssassinTargetDiesTest(Helper):
    def setUp(self):
        ''' Sets up a game where an opponent has only one influence and another
        opponent is dead
        '''
        super().setUp()
        self.game.reset_state({
            'game': {'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1},
            'players': [
                {'cash': 6, 'hidden': ['ambassador', 'assassin'], 'revealed': []},
                {'cash': 3, 'hidden': ['contessa', 'duke'], 'revealed': []},
                {'cash': 2, 'hidden': [], 'revealed': ['ambassador', 'captain']},
                {'cash': 2, 'hidden': ['assassin'], 'revealed': ['duke']}
            ]
        })

    def test_assassin_target_dies(self):
        ''' Tests that the assassinate action works correctly if the target
        player dies before the assassination can take place
        '''
        # Player 1 assassinates player 3
        self.game.play_action('assassinate:3')
        self.game.play_action('pass')
        self.game.play_action('pass')
        # Player 3 blocks
        self.game.play_action('block:contessa')
        # Players 0 and 1 both challenge
        self.game.play_action('challenge')
        self.game.play_action('challenge')
        # Player 3 reveals an assassin
        self.game.play_action('reveal:assassin')
        # Player 3 dies, so the turn passes to player 0, the next living player
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        # Player 3 is dead
        self.assert_hidden(3, [])

class LegalActionsTest(Helper):
    ''' Tests for which actions are legal in a given state
    '''

    def test_legal_actions_steal(self):
        ''' Tests that the legal actions are correct at the different phases
        of the steal action
        '''
        self.assert_legal_actions([
            'exchange',
            'foreign_aid',
            'income',
            'steal:1',
            'steal:2',
            'steal:3',
            'tax'
        ])
        # Player 0 steals
        self.game.play_action(STEAL + ':1')
        # Player 3 challenges
        self.assert_legal_actions(['pass', 'challenge'])
        self.game.play_action(PASS)
        self.assert_legal_actions(['pass', 'challenge'])
        self.game.play_action(PASS)
        self.assert_legal_actions(['pass', 'challenge'])
        self.game.play_action(CHALLENGE)
        # Player 0 reveals captain
        self.assert_legal_actions([reveal(DUKE), reveal(CAPTAIN)])
        self.game.play_action(reveal(CAPTAIN))
        # Player 3 reveals duke
        self.assert_legal_actions([reveal(DUKE), reveal(CAPTAIN)])
        self.game.play_action(reveal(DUKE))
        # Player 1 blocks
        self.assert_legal_actions([PASS, block(AMBASSADOR), block(CAPTAIN)])

    def test_legal_actions_exchange(self):
        ''' Tests that the legal actions are correct when choosing which cards
        to keep as part of an exchange
        '''
        # Player 0 exchanges
        self.game.play_action(EXCHANGE)
        # Nobody challenges
        self.game.play_action(PASS)
        self.game.play_action(PASS)
        self.game.play_action(PASS)
        # Player 0 can choose roles
        self.assert_legal_actions([
            'keep:assassin,assassin',
            'keep:assassin,captain',
            'keep:assassin,duke',
            'keep:captain,duke'
        ])

class ResetStateTest(unittest.TestCase):
    def test_reset_state(self):
        for seed in range(RESET_STATE_RUNS):
            try:
                np_random = np.random.RandomState()
                np_random.seed(seed)
                game = Coup(4, np_random)
                game.init_game()
                while not game.is_game_over():
                    state_before = game.get_state()
                    # Block ties are broken at random, so we must save the random state
                    random_state = np_random.get_state()
                    # Cannot restore state to the middle of a challenge or block, so simulate until the state changes
                    state_after = state_before
                    actions = []
                    while (
                        state_after['game']['phase'] == state_before['game']['phase'] and
                        state_after['game']['whose_turn'] == state_before['game']['whose_turn']
                    ):
                        legal_actions = game.get_legal_actions()
                        action = np_random.choice(legal_actions)
                        game.play_action(action)
                        actions.append(action)
                        state_after = game.get_state()
                    # Reset everything and replay
                    game.reset_state(state_before)
                    np_random.set_state(random_state)
                    for action in actions:
                        # Advance the random state in the same way but do not use the result
                        legal_actions = game.get_legal_actions()
                        _ = np_random.choice(legal_actions)
                        game.play_action(action)
                    assert game.get_state() == state_after
            except Exception as e:
                raise Exception(f'Failed reset state test with seed {seed}', e)

class GameViewTest(unittest.TestCase):
    ''' Tests for the GameView class
    '''

    def setUp(self):
        self.maxDiff = None
        self.game = Coup(4, seeding.np_random(3)[0])
        self.game.init_game()
        self.game.players[0].cash = 8
        self.view = GameView(4)

    def assert_current_player(self, player_id):
        ''' Asserts that the current player is player_id
        '''
        state = self.game.get_state()
        self.assertEqual(state['game']['player_to_act'], player_id)

    def assert_state_view(self, expected):
        ''' Asserts that the current player's view of the game state is as expected
        '''
        state = self.game.get_state()
        view = self.view.view_of_state(state)
        self.assertEqual(view, expected)

    def assert_legal_actions(self, expected):
        ''' Asserts that the legal actions are correct from the current player's perspective
        '''
        state = self.game.get_state()
        actions = self.game.get_legal_actions()
        view = self.view.view_of_actions(actions, state['game']['player_to_act'])
        self.assertEqual(view, expected)

    def test_game(self):
        ''' Tests that the game state and legal actions are from the correct
        player's perspective throughout a turn
        '''
        # State is from perspective of player 0
        self.assert_current_player(0)
        self.assert_state_view({
            'game': {'phase': 'start_of_turn', 'player_to_act': 0, 'whose_turn': 0},
            'players': [
                {'cash': 8, 'hidden': ['assassin', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []}
            ]
        })
        self.assert_legal_actions([
            'assassinate:1',
            'assassinate:2',
            'assassinate:3',
            'coup:1',
            'coup:2',
            'coup:3',
            'exchange',
            'foreign_aid',
            'income',
            'steal:1',
            'steal:2',
            'steal:3',
            'tax'
        ])
        # Player 0 steals from player 1
        self.game.play_action(STEAL + ':1')
        # State is now from perspective of player 1
        self.assert_current_player(1)
        self.assert_state_view({
            'game': {'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 0, 'player_to_act': 0, 'whose_turn': 3},
            'players': [
                {'cash': 2, 'hidden': ['assassin', 'contessa'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 8, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': [('claim', 'captain')]}
            ]
        })
        # Player 1 passes
        self.game.play_action(PASS)
        # State is now from perspective of player 2
        self.assert_current_player(2)
        self.assert_state_view({
            'game': {'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 3, 'player_to_act': 0, 'whose_turn': 2},
            'players': [
                {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 8, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': [('claim', 'captain')]},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []}
            ]
        })
        # Player 2 challenges
        self.game.play_action(CHALLENGE)
        # State is now from perspective of player 3
        self.assert_current_player(3)
        self.assert_state_view({
            'game': {'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'player_to_act': 0, 'whose_turn': 1},
            'players': [
                {'cash': 2, 'hidden': ['captain', 'duke'], 'revealed': [], 'trace': []},
                {'cash': 8, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': [('claim', 'captain')]},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []}
            ]
        })
        # Player 2 passes, player 0 must reveal
        self.game.play_action(PASS)
        self.assert_current_player(0)
        self.assert_state_view({
            'game': {'phase': 'prove_challenge', 'action': 'steal', 'target_player': 1, 'player_to_act': 0, 'whose_turn': 0},
            'players': [
                {'cash': 8, 'hidden': ['assassin', 'captain'], 'revealed': [], 'trace': [('claim', 'captain')]},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []}
            ]
        })
        # Player 0 reveals captain, player 2 lost the challenge and must reveal
        self.game.play_action(reveal(CAPTAIN))
        self.assert_current_player(2)
        self.assert_state_view({
            'game': {'phase': 'incorrect_challenge', 'action': 'steal', 'target_player': 3, 'player_to_act': 0, 'whose_turn': 2},
            'players': [
                {'cash': 2, 'hidden': ['ambassador', 'captain'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 8, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': [('claim', 'captain'), ('reveal', 'captain')]},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []}
            ]
        })
        # Player 2 loses an ambassador, player 1 may now block
        self.game.play_action(reveal(AMBASSADOR))
        self.assert_current_player(1)
        self.assert_state_view({
            'game': {'phase': 'awaiting_block', 'action': 'steal', 'target_player': 0, 'player_to_act': 0, 'whose_turn': 3},
            'players': [
                {'cash': 2, 'hidden': ['assassin', 'contessa'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden'], 'revealed': ['ambassador'], 'trace': [('reveal', 'ambassador')]},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 8, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': [('claim', 'captain'), ('reveal', 'captain')]}
            ]
        })
        # Player 1 does not block, play passes to player 1
        self.game.play_action(PASS)
        self.assert_current_player(1)
        self.assert_state_view({
            'game': {'phase': 'start_of_turn', 'player_to_act': 0, 'whose_turn': 0},
            'players': [
                {'cash': 0, 'hidden': ['assassin', 'contessa'], 'revealed': [], 'trace': []},
                {'cash': 2, 'hidden': ['hidden'], 'revealed': ['ambassador'], 'trace': [('reveal', 'ambassador')]},
                {'cash': 2, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': []},
                {'cash': 10, 'hidden': ['hidden', 'hidden'], 'revealed': [], 'trace': [('claim', 'captain'), ('reveal', 'captain')]}
            ]
        })
        # Player 1 has fewer legal actions because he cannot afford to assassinate or coup.
        # The target player ids are from his perspective (with himself numbered as 0)
        self.assert_legal_actions([
            'exchange',
            'foreign_aid',
            'income',
            'steal:1',
            'steal:2',
            'steal:3',
            'tax'
        ])

if __name__ == '__main__':
    unittest.main()
