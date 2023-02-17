import unittest
import numpy as np

from rlcard.games.coup.coup import Coup, IllegalAction
from rlcard.games.coup.constants import *

class TestDealer:
    def __init__(self, deck):
        self.deck = deck

    def deal_cards(self, n):
        return [self.deck.pop(0) for _ in range(n)]
    
    def choose(self, items):
        return items[0]

class Helper(unittest.TestCase):
    def setUp(self):
        self.game = Coup(4, np.random)
        d = TestDealer(self.get_deck())
        self.game.init_game(d)

    def get_deck(self):
        # By default, every player has duke/captain, and the deck contains assassins
        return [DUKE, CAPTAIN] * 4 + [ASSASSIN, ASSASSIN]

    def assert_state(self, state):
        self.assertEqual(self.game.get_state(), state)

    def assert_legal_actions(self, actions):
        self.assertEqual(self.game.get_legal_actions(), actions)

    def assert_cash(self, player_id, cash):
        self.assertEqual(self.game.players[player_id].cash, cash)

    def assert_hidden(self, player_id, roles):
        self.assertEqual(set(self.game.players[player_id].hidden), set(roles))

    def assert_revealed(self, player_id, roles):
        self.assertEqual(set(self.game.players[player_id].revealed), set(roles))

class ForeignAidTest(Helper):
    def test_allowed(self):
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
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(FOREIGN_AID)
        # Player 0 played, the other three get a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(DUKE)
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block', 'action': 'foreign_aid', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(DUKE)
        # Players 1 and 3 blocked, one is chosen at random, then the other three players get a chance to challenge the block
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        # Player 2 challenged, player 1 must reveal whether he has a duke
        self.assert_state({'phase': 'challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'whose_turn': 0, 'player_to_act': 1})

    def test_blocked(self):
        self.setup_blocked()
        self.game.play_action(DUKE)
        # Player 1 did have the duke, player 2 must reveal
        self.assert_state({'phase': 'incorrect_challenge', 'action': 'foreign_aid', 'blocked_with': 'duke', 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CAPTAIN)
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

    def test_failed_block(self):
        self.setup_blocked()
        self.game.play_action(CAPTAIN)
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

class StealTest(Helper):
    def test_multiple_challenges(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(STEAL + '2')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(CHALLENGE)
        # Players 1 and 3 challenged, player 0 must reveal whether he has a captain
        self.assert_state({'phase': 'challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(CAPTAIN)
        # Player 0 had the role, so players 1 and 3 must reveal
        self.assert_state({'phase': 'incorrect_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(DUKE)
        self.assert_state({'phase': 'incorrect_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(DUKE)
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
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(STEAL + '2')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 2 now gets the chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'steal', 'target_player': 2, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(AMBASSADOR)
        # Player 2 blocks, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'steal', 'target_player': 2, 'blocked_with': 'ambassador', 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'steal', 'target_player': 2, 'blocked_with': 'ambassador', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'steal', 'target_player': 2, 'blocked_with': 'ambassador', 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        # Nobody challenged, the steal is blocked
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Money has not changed
        self.assert_cash(0, 2)
        self.assert_cash(2, 2)

class StealTest(Helper):
    def test_allowed(self):
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
    def setUp(self):
        super().setUp()
        self.game.players[0].cash = 3

    def test_allowed(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + '1')
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
        self.assert_state({'phase': 'lose_influence', 'action': 'assassinate', 'target_player': 1, 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(DUKE)
        # End of turn
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 pays 3
        self.assert_cash(0, 0)
        # Player 1 loses an influence
        self.assert_hidden(1, [CAPTAIN])
        self.assert_revealed(1, [DUKE])

    def test_challenged(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + '1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Player 2 challenged, player 0 must reveal whether he has an assassin
        self.assert_state({'phase': 'challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(DUKE)
        # Player 0 did not have an assassin, so the card is lost, turn is over
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 does not pay, since the action was challenged
        self.assert_cash(0, 3)
        # Player 0 loses an influence
        self.assert_hidden(0, [CAPTAIN])
        self.assert_revealed(0, [DUKE])

    def test_blocked(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + '1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 1 gets a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(CONTESSA)
        # Player 1 blocked, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        # Nobody challenged, the action does not take place
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 pays 3
        self.assert_cash(0, 0)

    def test_failed_block(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(ASSASSINATE + '1')
        # Player 0 played, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_challenge', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        # Nobody challenged, player 1 gets a chance to block
        self.assert_state({'phase': 'awaiting_block', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(CONTESSA)
        # Player 1 blocked, the other three get a chance to challenge
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 2})
        self.game.play_action(CHALLENGE)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(PASS)
        self.assert_state({'phase': 'awaiting_block_challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(PASS)
        # Player 0 challenged, player 1 must must reveal whether he has a contessa
        self.assert_state({'phase': 'challenge', 'action': 'assassinate', 'blocked_with': 'contessa', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(CAPTAIN)
        # Player 1 did not have a contessa, so the assassination continues
        self.assert_state({'phase': 'lose_influence', 'action': 'assassinate', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(DUKE)
        # Player 1 reveals his last card and is out of the game, play passes to player 2
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 2, 'player_to_act': 2})
        # Player 0 pays
        self.assert_cash(0, 0)
        # Player 1 has no influence
        self.assert_hidden(1, [])
        self.assert_revealed(1, [CAPTAIN, DUKE])

class ExchageTest(Helper):
    def test_exchange(self):
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
        self.assert_state({'phase': 'choose_new_roles', 'action': 'exchange', 'drawn_roles': [ASSASSIN, ASSASSIN], 'whose_turn': 0, 'player_to_act': 0})
        # Try some illegal choices: not enough roles
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(DUKE)
        self.assertEqual(str(cm.exception), 'Must choose 2 roles')
        # Wrong roles
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(DUKE + ',' + CONTESSA)
        self.assertEqual(str(cm.exception), 'Chosen roles are not available')
        # Too many of the same role
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(DUKE + ',' + DUKE)
        self.assertEqual(str(cm.exception), 'Chosen roles are not available')
        # Make a valid choice and end the turn
        self.game.play_action(DUKE + ',' + ASSASSIN)
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 0 should have new roles
        self.assert_hidden(0, [DUKE, ASSASSIN])

class CoupTest(Helper):
    def setUp(self):
        super().setUp()
        self.game.players[0].cash = 7
        self.game.reveal_role(1, DUKE)

    def test_coup(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(COUP + '3')
        # Player 3 must reveal
        self.assert_state({'phase': 'lose_influence', 'action': 'coup', 'target_player': 3, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(DUKE)
        # Turn ends
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 1, 'player_to_act': 1})
        # Player 3 has lost influence
        self.assert_revealed(3, [DUKE])
        self.assert_hidden(3, [CAPTAIN])
        # Player 0 pays 7
        self.assert_cash(0, 0)

    def test_coup_to_death(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(COUP + '1')
        # Player 1 must reveal his last card
        self.assert_state({'phase': 'lose_influence', 'action': 'coup', 'target_player': 1, 'whose_turn': 0, 'player_to_act': 1})
        self.game.play_action(CAPTAIN)
        # Turn skips over dead player to player 2
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 2, 'player_to_act': 2})
        # Player 1 has no influence
        self.assert_revealed(1, [DUKE, CAPTAIN])
        self.assert_hidden(1, [])

class TooPoorTest(Helper):
    def test_cannot_afford_coup(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        with self.assertRaises(IllegalAction) as cm:
            self.game.play_action(COUP + '3')
        self.assertEqual(str(cm.exception), 'Cannot afford to coup')

class LastPlayerDies(Helper):
    def setUp(self):
        super().setUp()
        self.game.players[0].cash = 7
        self.game.reveal_role(1, DUKE)
        self.game.reveal_role(1, CAPTAIN)
        self.game.reveal_role(2, DUKE)
        self.game.reveal_role(2, CAPTAIN)
        # Player 3 has one role left
        self.game.reveal_role(3, DUKE)

    def test_last_player_dies(self):
        self.assert_state({'phase': 'start_of_turn', 'whose_turn': 0, 'player_to_act': 0})
        self.game.play_action(COUP + '3')
        # Player 3 must reveal
        self.assert_state({'phase': 'lose_influence', 'action': 'coup', 'target_player': 3, 'whose_turn': 0, 'player_to_act': 3})
        self.game.play_action(CAPTAIN)
        # Game over
        self.assert_state({'phase': 'game_over', 'winning_player': 0})

class LegalActionsTest(Helper):
    def test_legal_actions_steal(self):
        self.assert_legal_actions([
            'assassinate1',
            'assassinate2',
            'assassinate3',
            'coup1',
            'coup2',
            'coup3',
            'exchange',
            'foreign_aid',
            'income',
            'steal1',
            'steal2',
            'steal3',
            'tax'
        ])
        # Player 0 steals
        self.game.play_action(STEAL + '1')
        # Player 3 challenges
        self.assert_legal_actions(['pass', 'challenge'])
        self.game.play_action(PASS)
        self.assert_legal_actions(['pass', 'challenge'])
        self.game.play_action(PASS)
        self.assert_legal_actions(['pass', 'challenge'])
        self.game.play_action(CHALLENGE)
        # Player 0 reveals captain
        self.assert_legal_actions([DUKE, CAPTAIN])
        self.game.play_action(CAPTAIN)
        # Player 3 reveals duke
        self.assert_legal_actions([DUKE, CAPTAIN])
        self.game.play_action(DUKE)
        # Player 1 blocks
        self.assert_legal_actions([PASS, AMBASSADOR, CAPTAIN])

    def test_legal_actions_exchange(self):
        # Player 0 exchanges
        self.game.play_action(EXCHANGE)
        # Nobody challenges
        self.game.play_action(PASS)
        self.game.play_action(PASS)
        self.game.play_action(PASS)
        # Player 0 can choose roles
        self.assert_legal_actions([
            'assassin,assassin',
            'assassin,captain',
            'assassin,duke',
            'captain,duke'
        ])

if __name__ == '__main__':
    unittest.main()
