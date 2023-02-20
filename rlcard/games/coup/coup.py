import re
from itertools import combinations

from .dealer import CoupDealer as Dealer
from .player import CoupPlayer as Player
from .constants import *

class IllegalAction(Exception):
    pass

class Coup:
    ''' Implementation of Coup that is agnostic to RLCard
    '''
    def __init__(self, num_players, np_random):
        self.num_players = num_players
        self.np_random = np_random

    def init_game(self, dealer=None):
        self.dealer = dealer if dealer else Dealer(self.np_random)
        self.players = [Player(i) for i in range(self.num_players)]
        for player in self.players:
            player.hidden = self.dealer.deal_cards(2)
        self.state = Turn(self, 0)

    def player_to_act(self):
        return self.state.player_to_act()

    def play_action(self, action):
        self.state.play_action(action)
        if not self.is_game_over():
            player_id = self.state.player_to_act()
            if not self.is_alive(player_id):
                raise RuntimeError(f'Unexpected state: player to act is dead')

    def get_legal_actions(self):
        return self.state.get_legal_actions()

    def get_state(self):
        return {
            'players': [p.get_state() for p in self.players],
            'game': self.state.get_state()
        }

    def get_next_player(self, player_id):
        next_player_id = player_id
        while True:
            next_player_id = (next_player_id + 1) % self.num_players
            if self.is_alive(next_player_id):
                break
            if next_player_id == player_id:
                raise RuntimeError(f'Unexpected state: only one player left alive')
        return next_player_id

    def end_turn(self):
        if not self.is_game_over():
            player_id = self.get_next_player(self.state.player_id)
            self.state = Turn(self, player_id)

    def is_game_over(self):
        return type(self.state) == GameOver

    def reveal_role(self, player_id, role):
        player = self.players[player_id]
        player.hidden.remove(role)
        player.revealed.append(role)
        live_players = [p for p in range(self.num_players) if self.is_alive(p)]
        if len(live_players) == 1:
            self.state = GameOver(live_players[0])

    def replace_role(self, player_id, role):
        player = self.players[player_id]
        player.hidden.remove(role)
        player.hidden += self.dealer.deal_cards(1)

    def replace_all_roles(self, player_id, roles):
        player = self.players[player_id]
        assert len(roles) == len(player.hidden)
        player.hidden = list(roles)

    def player_has_role(self, player_id, role):
        return role in self.players[player_id].hidden

    def is_alive(self, player_id):
        return len(self.players[player_id].hidden) > 0

    def get_influence(self, player_id):
        return list(self.players[player_id].hidden)

    def add_cash(self, player_id, cash):
        self.players[player_id].cash += cash

    def deduct_cash(self, player_id, cash):
        cash = min(cash, self.players[player_id].cash)
        self.players[player_id].cash -= cash
        return cash

    def can_afford(self, player_id, cost):
        return self.players[player_id].cash >= cost

    def choose(self, items):
        return self.dealer.choose(items)

class GameOver:
    def __init__(self, winning_player):
        self.winning_player = winning_player

    def get_state(self):
        return {'phase': 'game_over', 'winning_player': self.winning_player}

    def get_legal_actions(self):
        return []

ACTION_RE = re.compile(
    '(' + '|'.join(UNTARGETED_ACTIONS) + ')|' +
    '(' + '|'.join(TARGETED_ACTIONS) + r')(\d+)$'
)

class Turn:
    def __init__(self, game, player_id):
        self.game = game
        self.player_id = player_id
        self.action = None

    def play_action(self, action):
        if self.action:
            self.action.play_action(action)
        else:
            self.play_initial_action(action)

    def play_initial_action(self, action_str):
        m = ACTION_RE.fullmatch(action_str)
        if m is None:
            raise IllegalAction(f'Unknown action {action_str}')
        action_name = m.group(1) or m.group(2)
        if m.group(3) is not None:
            target_player = int(m.group(3))
            if target_player >= self.game.num_players:
                raise IllegalAction(f'Unknown target player {target_player}')
        if action_name == INCOME:
            self.game.add_cash(self.player_id, 1)
            self.game.end_turn()
        elif action_name == FOREIGN_AID:
            action = ForeignAid(self.game, self.player_id)
        elif action_name == TAX:
            action = Tax(self.game, self.player_id)
        elif action_name == EXCHANGE:
            action = ExchangeAction(self.game, self.player_id)
        elif action_name == STEAL:
            action = Steal(self.game, self.player_id, target_player)
        elif action_name == ASSASSINATE:
            action = AssassinateAction(self.game, self.player_id, target_player)
        elif action_name == COUP:
            action = CoupAction(self.game, self.player_id, target_player)
        else:
            raise NotImplementedError
        if not self.game.can_afford(self.player_id, action.cost):
            raise IllegalAction(f'Cannot afford to {action_name}')
        self.action = action
        action.init()

    def player_to_act(self):
        if self.action:
            return self.action.player_to_act()
        else:
            return self.player_id

    def get_legal_actions(self):
        if self.action:
            return self.action.get_legal_actions()
        else:
            return sorted(UNTARGETED_ACTIONS + [
                f'{a}{p}' for a in TARGETED_ACTIONS
                for p in range(self.game.num_players)
                if self.game.is_alive(p) and p != self.player_id
            ])

    def get_state(self):
        state = {
            'phase': 'start_of_turn',
            'whose_turn': self.player_id,
            'player_to_act': self.player_to_act()
        }
        if self.action:
            self.action.augment_state(state)
        return state

class Action:
    ''' An action requiring a response of some kind
    '''
    def __init__(self, game, name, player_id, target_player=None, cost=0):
        self.game = game
        self.name = name
        self.player_id = player_id
        self.target_player = target_player
        self.cost = cost
        self.challenge = None
        self.block = None
        self.final_action = None

    def init(self):
        if not self.challenge:
            self.game.deduct_cash(self.player_id, self.cost)

    def player_to_act(self):
        if self.challenge:
            return self.challenge.player_to_act()
        elif self.block:
            return self.block.player_to_act()
        elif self.final_action:
            return self.final_action.player_to_act()
        else:
            return self.player_id

    def play_action(self, action):
        if self.challenge:
            self.challenge.play_action(action)
        elif self.block:
            self.block.play_action(action)
        elif self.final_action:
            self.final_action.play_action(action)
        else:
            self.play_final_action(action)

    def get_legal_actions(self):
        if self.challenge:
            return self.challenge.get_legal_actions()
        elif self.block:
            return self.block.get_legal_actions()
        elif self.final_action:
            return self.final_action.get_legal_actions()
        else:
            raise RuntimeError(f'Unexpected state: could not determine legal actions')
            
    def play_final_action(self, action):
        ''' This is for actions like exchange, which require a response
        which is not a block, challenge or reveal
        '''
        raise RuntimeError(f'Unexpected state: could not resolve action {action}')

    def augment_state(self, state):
        state['action'] = self.name
        if self.target_player:
            state['target_player'] = self.target_player
        if self.challenge:
            state['phase'] = 'awaiting_challenge'
            self.challenge.augment_state(state)
        if self.block:
            state['phase'] = 'awaiting_block'
            self.block.augment_state(state)
        if self.final_action:
            self.final_action.augment_state(state)

    def resolve_challenge(self, action_allowed):
        self.challenge = None
        if action_allowed:
            self.game.deduct_cash(self.player_id, self.cost)
            self.action_accepted()
            if not self.block:
                self.do_action()
        else:
            self.game.end_turn()

    def resolve_block(self, action_allowed):
        self.block = None
        if action_allowed:
            self.do_action()
        else:
            self.game.end_turn()

    def action_accepted(self):
        ''' For actions that can be challenged, called when all
        players allow the action or a challenge failed

        If the action can be blocked, create the Block here
        If the action costs money, deduct it here
        '''
        pass

    def do_action(self):
        ''' Called when the action is allowed, after resolving challenges and blocks
        Subclasses are responsible for ending the turn after performing their action
        '''
        raise NotImplementedError(f'do_action in {self}')

class Challenge:
    def __init__(self, parent, challenged_player, role):
        # Parent could be an Action or a Block
        self.parent = parent
        self.game = parent.game
        self.challenged_player = challenged_player
        self.role = role
        self.player_id = self.game.get_next_player(challenged_player)
        self.responses = {}
        self.reveal = None
        # Correct means the challenger(s) were right
        # (None means we don't yet know if they were right)
        self.challenge_correct = None

    def player_to_act(self):
        if self.reveal:
            return self.reveal.player_to_act()
        else:
            return self.player_id

    def play_action(self, action):
        if self.reveal:
            self.reveal.play_action(action)
        else:
            if action != PASS and action != CHALLENGE:
                raise IllegalAction(f'Unknown action {action}')
            # Store the action
            self.responses[self.player_id] = action
            # Pass to next player until everyone has played
            next_player_id = self.game.get_next_player(self.player_id)
            if next_player_id == self.challenged_player:
                if CHALLENGE in self.responses.values():
                    # Challenged player must reveal whether they have the role
                    self.reveal = Reveal(self, self.challenged_player, 'challenge')
                else:
                    # Nobody challenged
                    self.parent.resolve_challenge(True)
            else:
                self.player_id = next_player_id

    def get_legal_actions(self):
        if self.reveal:
            return self.reveal.get_legal_actions()
        else:
            return [PASS, CHALLENGE]

    def augment_state(self, state):
        if self.reveal:
            self.reveal.augment_state(state)

    def after_reveal(self, revealed_role):
        if self.challenge_correct is None:
            # The first reveal, where the challenged player proves whether he has the role
            self.challenge_correct = (revealed_role != self.role)
            if self.challenge_correct:
                # Challenged player did not have the role, he loses the card,
                # and the challenge is resolved
                self.game.reveal_role(self.challenged_player, revealed_role)
                self.parent.resolve_challenge(False)
            else:
                # Challenged player had the role, he gets a new card
                self.game.replace_role(self.challenged_player, revealed_role)
                # The challenger(s) were not successful and they must now reveal in turn
                self.reveal_next_challenger(self.challenged_player)
        else:
            # A challenger has revealed. Move to the next challenger, if there is one
            assert not self.challenge_correct
            self.game.reveal_role(self.reveal.player_id, revealed_role)
            self.reveal_next_challenger(self.reveal.player_id)

    def reveal_next_challenger(self, player_id):
        # Go through all the players who challenged and make them reveal,
        # starting with the player to the left of player_id
        while True:
            player_id = self.game.get_next_player(player_id)
            if player_id == self.challenged_player:
                # All the players have been visited
                self.parent.resolve_challenge(True)
                break
            elif self.responses[player_id] == CHALLENGE:
                # Another challenger must reveal
                phase_name = 'correct_challenge' if self.challenge_correct else 'incorrect_challenge'
                self.reveal = Reveal(self, player_id, phase_name)
                break
            # Else this player allowed, so move on to the next

class Block:
    def __init__(self, action, roles, blocking_player_id=None):
        self.action = action
        self.game = action.game
        self.blocking_player_id = blocking_player_id
        if blocking_player_id:
            # A specific player may block
            self.player_id = blocking_player_id
        else:
            # Everyone may block in turn
            self.player_id = self.game.get_next_player(action.player_id)
        self.roles = roles if type(roles) == list else [roles]
        self.responses = {}
        self.challenge = None

    def player_to_act(self):
        if self.challenge:
            return self.challenge.player_to_act()
        else:
            return self.player_id

    def augment_state(self, state):
        if self.challenge:
            state['phase'] = 'awaiting_block_challenge'
            state['blocked_with'] = self.challenge.role
            state['blocking_player'] = self.challenge.challenged_player
            self.challenge.augment_state(state)

    def play_action(self, action):
        if self.challenge:
            self.challenge.play_action(action)
            return
        if action != PASS and action not in self.roles:
            raise IllegalAction(f'Unknown action {action}')
        # Store the action
        self.responses[self.player_id] = action
        if self.blocking_player_id:
            # Only the target player blocks
            self.execute_block()
        else:
            # Anyone can block, so go to the next player
            next_player_id = self.game.get_next_player(self.player_id)
            if next_player_id == self.action.player_id:
                self.execute_block()
            else:
                self.player_id = next_player_id

    def get_legal_actions(self):
        if self.challenge:
            return self.challenge.get_legal_actions()
        else:
            return [PASS] + self.roles

    def execute_block(self):
        blocks = [b for b in self.responses.items() if b[1] != PASS]
        if len(blocks) == 0:
            # No one blocked
            self.action.resolve_block(True)
        else:
            if len(blocks) == 1:
                # One person blocked
                block = blocks[0]
            else:
                # Several people blocked - choose a random one
                block = self.game.choose(blocks)
            self.challenge = Challenge(self, block[0], block[1])

    def resolve_challenge(self, block_allowed):
        action_allowed = not block_allowed
        self.action.resolve_block(action_allowed)

class Reveal:
    def __init__(self, parent, player_id, phase_name):
        # Parent could be an Action or Challenge
        self.parent = parent
        self.player_id = player_id
        self.phase_name = phase_name
        self.game = parent.game

    def play_action(self, action):
        if not self.game.player_has_role(self.player_id, action):
            raise IllegalAction(f'Player {self.player_id} does not have role {action}')
        self.parent.after_reveal(action)

    def get_legal_actions(self):
        return self.game.get_influence(self.player_id)

    def augment_state(self, state):
        state['phase'] = self.phase_name

    def player_to_act(self):
        return self.player_id

class ForeignAid(Action):
    def __init__(self, game, player_id):
        super().__init__(game, FOREIGN_AID, player_id)
        self.block = Block(self, DUKE)

    def do_action(self):
        self.game.add_cash(self.player_id, 2)
        self.game.end_turn()

class Steal(Action):
    def __init__(self, game, player_id, target_player):
        super().__init__(game, STEAL, player_id, target_player)
        self.challenge = Challenge(self, player_id, CAPTAIN)

    def action_accepted(self):
        self.block = Block(self, [AMBASSADOR, CAPTAIN], self.target_player)

    def do_action(self):
        cash = self.game.deduct_cash(self.target_player, 2)
        self.game.add_cash(self.player_id, cash)
        self.game.end_turn()

class Tax(Action):
    def __init__(self, game, player_id):
        super().__init__(game, TAX, player_id)
        self.challenge = Challenge(self, player_id, DUKE)

    def do_action(self):
        self.game.add_cash(self.player_id, 3)
        self.game.end_turn()

class AssassinateAction(Action):
    def __init__(self, game, player_id, target_player):
        super().__init__(game, ASSASSINATE, player_id, target_player, 3)
        self.challenge = Challenge(self, player_id, ASSASSIN)

    def action_accepted(self):
        self.block = Block(self, [CONTESSA], self.target_player)

    def do_action(self):
        self.final_action = Reveal(self, self.target_player, 'lose_influence')

    def after_reveal(self, revealed_role):
        self.game.reveal_role(self.target_player, revealed_role)
        self.game.end_turn()

class ExchangeAction(Action):
    def __init__(self, game, player_id):
        super().__init__(game, EXCHANGE, player_id)
        self.challenge = Challenge(self, player_id, AMBASSADOR)
        self.drawn_roles = None

    def do_action(self):
        # The action was not challenged, so player gets some cards to choose from
        assert self.drawn_roles is None
        self.drawn_roles = self.game.dealer.deal_cards(2)

    def get_legal_actions(self):
        if self.drawn_roles:
            existing_roles = self.game.get_influence(self.player_id)
            pool = existing_roles + self.drawn_roles
            tuples = combinations(pool, len(existing_roles))
            # Sort each tuple and then de-duplicate: (duke,captain) and (captiain,duke) are equivalent
            choices = list(set([tuple(sorted(t)) for t in tuples]))
            return sorted([','.join(c) for c in choices])

    def play_final_action(self, action):
        # Player has chosen the cards to keep
        new_roles = action.split(',')
        existing_roles = self.game.get_influence(self.player_id)
        if len(new_roles) != len(existing_roles):
            raise IllegalAction(f'Must choose {len(existing_roles)} roles')
        pool = existing_roles + self.drawn_roles
        for r in new_roles:
            if r not in pool:
                raise IllegalAction(f'Chosen roles are not available')
            # Remove r to make sure we handle duplicates correctly
            pool.remove(r)
        self.game.replace_all_roles(self.player_id, new_roles)
        self.game.end_turn()

    def augment_state(self, state):
        super().augment_state(state)
        if self.drawn_roles:
            state['phase'] = 'choose_new_roles'
            # TODO: drawn_roles should be private to the player who is exchanging
            state['drawn_roles'] = self.drawn_roles

class CoupAction(Action):
    def __init__(self, game, player_id, target_player):
        super().__init__(game, COUP, player_id, target_player, 7)
        self.final_action = Reveal(self, target_player, 'lose_influence')

    def after_reveal(self, revealed_role):
        self.game.reveal_role(self.target_player, revealed_role)
        self.game.end_turn()
