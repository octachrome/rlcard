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

''' Implementation of the card game Coup.

The rules can be found at https://boardgamegeek.com/boardgame/131357/coup.
A flowchart clarifying the game play can be found at
https://boardgamegeek.com/filepage/86105/action-resolution-order-flowchart.

This implementation is intended for machine learning applications and is
independent of any learning framework. All behaviour specific to RLCard
is restricted to the files game.py and envs/coup.py.

The original game contains elements of simultaneous play, but for simplicity
(and compatibility with RLCard) the turn order is serialized and deterministic
in this implementation. Particularly:

- After a player declares an action or reaction which is linked to a role,
  every other (living) player gets a chance to challenge the action, in turn
  order. Whether or not each player will challenge is kept secret until all
  players have responded. Ater this, the challenge is resolved. If the
  challenge is correct (the player declaring the action does not reveal the
  required role), then this player loses the influence, as usual. However, if
  the challenge is incorrect, ALL players who chose to challenge must reveal
  (and lose) an influence, in turn order. I feel this is in keeping with the
  spirit of the original game.

- There is one case where multiple players can choose to block an action:
  foreign aid (blocked by any player with a duke). As with a challenge, each
  player gets a chance to block (secretly). Once all players have decided,
  if more than one player wants to block, one of them is selected at random
  by the dealer. This is a significant deviation from the original rules, but
  it is needed to avoid too much complexity. I doubt this will much affect an
  agent's ability to learn a good strategy, since foreign aid is a minor action
  in the game. If anything, agents will learn to block foreign aid slightly
  more that they would otherwise, since there is a probability that its block
  will have no effect (when the dealer chooses a different player to block.)

Some other points where the rules sometimes vary from player to player:

- When a player is challenged, he may choose to reveal a role that is different
  from the one he claimed, i.e., intentionally losing the challenge as part of
  a longer bluff. (In accordance with the flowchart.)

- If a player assassinates, is challenged and loses the challenge, they do not
  pay the 3 credits. If they win the challenge (or no one challenges), they pay
  the 3 credits, regardless of whether the opponent blocks. (In accordance with
  the flowchart.)

- If a player assassinates and the opponent incorrectly challenges the
  assassin, the opponent loses an influence. The opponent then still has the
  chance to block with a contessa. (In accordance with the flowchart.)

- If a player steals and the opponent dies (by incorrectly challenging the
  captain), the stealing player gets no money. This is a convenient behaviour
  given the implementation. (The correct behaviour in this case is not clear in
  the official rules or flowchart.)

'''
import re

from .dealer import CoupDealer as Dealer
from .player import CoupPlayer as Player
from .constants import *
from .utils import get_keep_actions

class Coup:
    ''' Implementation of Coup that is agnostic to RLCard
    '''
    def __init__(self, num_players, np_random):
        ''' Constructs a Coup game

        Args:
            num_players (int): number of players in the game
            np_random (numpy.random.RandomState): used for randomized decisions
        '''
        self.num_players = num_players
        self.np_random = np_random

    def init_game(self, dealer=None):
        ''' Initializes a new game

        Must be called at the start of every game.

        Args:
            dealer: If provided, overrides the default dealer (for testing)
        '''
        self.dealer = dealer if dealer else Dealer(self.np_random)
        self.players = [Player(i) for i in range(self.num_players)]
        for player in self.players:
            player.hidden = self.dealer.deal_cards(2)
        self.state = Turn(self, 0)

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        Returns:
            (int): id of the player who must act in the current state

        Note that in Coup, the next player to act depends on the previous
        player's action.
        '''
        return self.state.player_to_act()

    def play_action(self, action):
        ''' Plays the given action

        Args:
            action (str): the action to play
        '''
        self.state.play_action(action)
        if not self.is_game_over():
            player_id = self.state.player_to_act()
            if not self.is_alive(player_id):
                raise RuntimeError(f'Unexpected state: player to act is dead')

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        Returns:
            (list of string): list of action names that are legal in the current state
        '''
        return self.state.get_legal_actions()

    def get_state(self):
        ''' Returns a dictionary describing the current state of the game

        See test_coup_game.py for examples of game states.

        Returns:
            (dict) that contains:
                game (dict): a dict which may contain the following entries, depending on phase
                    phase: current game phase (always present, one of constants.PHASES)
                    whose_turn: id of the player who played the initial action in this turn
                    player_to_act: id of the player who must now act
                    action: the initial action played in this turn, if any
                    target_player: the player targeted by the action, if any (assassinate, steal and coup only)
                    blocked_with: the role used to block the action, if any
                    blocking_player: the player who is blocking the action, if any
                    drawn_roles: for the exchange action, the role cards drawn from the deck
                    winning_player: id of the player who won, if the game is over
                players (list of dict): a dict for each player, containing:
                    cash (int): number of credits the player has
                    hidden (list of str): hidden role cards of the player
                    revealed (list of str): revealed role cards of the player
                    trace (list of tuple): selected actions the player has taken which hint at his roles
        '''
        return {
            'players': [p.get_state() for p in self.players],
            'game': self.state.get_state(),
            'dealer': self.dealer.get_state()
        }

    def get_next_player(self, player_id):
        ''' Returns the player whose turn will follow the given player

        Args:
            player_id (int): id of the player to query

        Returns:
            (int): id of the player who acts after player_id
        
        Used by Action subclasses to determine challenge order etc.

        Dead players are skipped.
        '''
        next_player_id = player_id
        while True:
            next_player_id = (next_player_id + 1) % self.num_players
            if self.is_alive(next_player_id):
                break
            if next_player_id == player_id:
                raise RuntimeError(f'Unexpected state: only one player left alive')
        return next_player_id

    def end_turn(self):
        ''' Finish the current turn and pass to the next player

        Called by Action subclasses when the action is complete.
        '''
        if not self.is_game_over():
            player_id = self.get_next_player(self.state.player_id)
            self.state = Turn(self, player_id)

    def reset_state(self, state):
        ''' Resets the game to a given state

        Args:
            state (dict): a state dictionary such as returned by get_state

        Currently only supports a limited number of states, such as the start
        of a player's turn.

        Used for testing and debugging only.
        '''
        self.state = Turn(self, state['game']['whose_turn'])
        for pid, p in enumerate(state['players']):
            self.players[pid].cash = p['cash']
            self.players[pid].hidden = p['hidden']
            self.players[pid].revealed = p['revealed']
        if state['game']['phase'] != START_OF_TURN:
            self.state.reset_state(state)
        assert state['game']['player_to_act'] == self.player_to_act()

    def is_game_over(self):
        ''' Returns whether the game is over

        Returns:
            (bool): True if the game is over
        '''
        return type(self.state) == GameOver

    def reveal_role(self, player_id, role):
        ''' Permanently reveals a player's influence

        Args:
            player_id (int): id of the player whose influence to reveal
            role (str): the role to reveal

        Called by Action subclasses when a player loses an influence.

        The given role will be removed from the player's hidden influence and
        added to the player's revealed influence.
        '''
        player = self.players[player_id]
        player.hidden.remove(role)
        player.revealed.append(role)
        live_players = [p for p in range(self.num_players) if self.is_alive(p)]
        if len(live_players) == 1:
            self.state = GameOver(live_players[0])

    def replace_role(self, player_id, role):
        ''' Replaces a player's influence with another from the deck

        Args:
            player_id (int): id of the player who gets a new influence
            role (str): the role to replace

        Called by Action subclasses when a player must reveal a card but does
        not lose one.
        '''
        player = self.players[player_id]
        player.hidden.remove(role)
        self.dealer.replace_cards([role])
        player.hidden += self.dealer.deal_cards(1)

    def replace_all_roles(self, player_id, roles):
        ''' Replaces all influence of a player

        Args:
            player_id (int): id of the player who gets new influence
            roles (list of str): the new roles

        Called at the end of an exchange action when the player has chosen the
        roles to keep.
        '''
        player = self.players[player_id]
        assert len(roles) == len(player.hidden)
        player.hidden = list(roles)

    def trace_claim(self, player_id, role):
        ''' Records a role claim in a player's history

        Args:
            player_id (int): id of the player making the claim
            role (str): the role claimed by the player

        Called whenever a player claims to have a role, by taking an action or
        blocking an opponent's action.

        Role claims can be used by an agent to infer hidden information about a
        player.
        '''
        self.players[player_id].trace.append(('claim', role))

    def trace_reveal(self, player_id, role):
        ''' Records a revealed role in a player's history

        Args:
            player_id (int): id of the player who revealed the role
            role (str): the role revealed by the player

        Called whenever a player reveals an influence due to a failed challenge,
        assassination or coup.

        Revealed roles can be used by an agent to learn how a player's prior
        actions are correlated with the hidden roles that the player has.
        '''
        self.players[player_id].trace.append(('reveal', role))

    def trace_lost_challenge(self, player_id, role):
        ''' Records in a player's history a role claim that was correctly challenged

        Args:
            player_id (int): id of the player who was successfully challenged
            role (str): the role that was challenged, which the player did not reveal

        Called whenever a player claims a role, is challenged, and loses the
        challenge.

        An agent can use this information to learn to stop bluffing a role that
        they have already been successfully challenged on.
        '''
        self.players[player_id].trace.append(('lost_challenge', role))

    def trace_exchange(self, player_id):
        ''' Records an exchange in a player's history

        Args:
            player_id (int): id of the player who exchanged his roles

        Called whenever a player successfully plays the the exchange action and
        replaces his cards.

        The player's previous roles and new roles are unknown to other players.

        An agent can use this information to modify their model of a player's
        hidden roles.
        '''
        self.players[player_id].trace.append(('exchange', None))

    def player_has_role(self, player_id, role):
        ''' Returns whether a player has the given hidden role

        Args:
            player_id (int): id of the player to check
            role (str): the role to check

        Returns 
        Used when determining the outcome of a challenge.
        '''
        return role in self.players[player_id].hidden

    def is_alive(self, player_id):
        ''' Returns whether a player is alive

        Args:
            player_id (int): id of the player to check

        Returns:
            (int): True of the player is alive
        '''
        return len(self.players[player_id].hidden) > 0

    def get_influence(self, player_id):
        ''' Returns a player's hidden roles

        Args:
            player_id (int): id of the player to check

        Returns:
            (list of str): the hidden roles of the player
        '''
        return list(self.players[player_id].hidden)

    def add_cash(self, player_id, cash):
        ''' Add some credits to a player's balance

        Args:
            player_id (int): id of the player to credit
            cash (int): number of credits to add
        '''
        self.players[player_id].cash += cash

    def deduct_cash(self, player_id, cash):
        ''' Subtract some credits from a player's balance

        Args:
            player_id (int): id of the player to debit
            cash (int): number of credits to subtract

        Returns:
            (int): the number of credits that was actually subtracted

        Ensures that a player never has negative credit, which is important
        when stealing from a player with 1 credit: in this case, the theif will
        only gain 1 credit.
        '''
        cash = min(cash, self.players[player_id].cash)
        self.players[player_id].cash -= cash
        return cash

    def can_afford(self, player_id, price):
        ''' Returns whether a player can afford a price

        Args:
            player_id (int): id of the player to check
            price (int): price to check

        Returns:
            (bool): True if the player can afford the price

        Used to check whether actions with a cost (coup and assassination) can
        be afforded by the current player.
        '''
        return self.players[player_id].cash >= price

    def choose(self, items):
        ''' Randomly choses one of the given items

        Args:
            items (list): items to choose from

        Returns:
            The chosen item

        Used to choose which player gets to a block foreign aid attempt when
        more than one player tries to block.
        '''
        return self.dealer.choose(items)

class GameOver:
    ''' State representing the end of the game
    '''
    def __init__(self, winning_player):
        ''' Constructs a game over state

        Args:
            winning_player (int): id of the player who won the game
        '''
        self.winning_player = winning_player

    def get_state(self):
        ''' Returns a dictionary describing the game state

        See Coup.get_state.
        '''
        return {'phase': GAME_OVER, 'winning_player': self.winning_player}

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        See Coup.get_legal_actions.
        '''
        return []

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        See Coup.player_to_act.
        '''
        return None

''' Regular expression describing all possible actions at the start of a player's turn
'''
ACTION_RE = re.compile(
    '(' + '|'.join(UNTARGETED_ACTIONS) + ')|' +
    '(' + '|'.join(TARGETED_ACTIONS) + r'):(\d+)$'
)

class Turn:
    ''' A turn in a game of Coup

    Includes the initial action, any block and any challenges
    '''
    def __init__(self, game, player_id):
        ''' Constructs a Turn

        Args:
            game (Coup): the game being played
            player_id (int): id of the player who starts the turn
        '''
        self.game = game
        self.player_id = player_id
        # Action is assigned when the player chooses their initial action
        self.action = None

    def play_action(self, action):
        ''' Plays the given action

        See Coup.play_action.
        '''
        if self.action:
            self.action.play_action(action)
        else:
            self._play_initial_action(action)

    def _play_initial_action(self, action_str):
        ''' Plays the given action at the start of a player's turn

        Args:
            action_str (str): the action to play

        Note: some actions include a target player, e.g., 'coup:3'
        '''
        m = ACTION_RE.fullmatch(action_str)
        if m is None:
            raise IllegalAction(f'Unknown action {action_str}')
        action_name = m.group(1) or m.group(2)
        if m.group(3) is not None:
            target_player = int(m.group(3))
            if target_player >= self.game.num_players:
                raise IllegalAction(f'Unknown target player {target_player}')
        if not self._can_afford_action(action_name):
            raise IllegalAction(f'Cannot afford to {action_name}')
        if self.game.can_afford(self.player_id, 10) and action_name != COUP:
            raise IllegalAction(f'Players with 10 or more credits must coup')
        action = None
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
            raise RuntimeError(f'Unexpected action {action_name}')
        if action:
            action.init()
            self.action = action

    def _can_afford_action(self, action_name):
        ''' Returns whether the player can afford the given action

        Args:
            action_name (str): name of the action (without target suffix)

        Returns:
            (bool): True if the player can afford the action
        '''
        return self.game.can_afford(self.player_id, ACTION_COSTS.get(action_name, 0))

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        See Coup.player_to_act.
        '''
        if self.action:
            return self.action.player_to_act()
        else:
            return self.player_id

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        See Coup.get_legal_actions.
        '''
        if self.action:
            return self.action.get_legal_actions()
        elif self.game.can_afford(self.player_id, 10):
            # A player with 10 or more credits must coup
            return sorted([
                f'coup:{p}' for p in range(self.game.num_players)
                if self.game.is_alive(p)
                and p != self.player_id
            ])
        else:
            return sorted(UNTARGETED_ACTIONS + [
                f'{a}:{p}' for a in TARGETED_ACTIONS
                for p in range(self.game.num_players)
                if self.game.is_alive(p)
                and p != self.player_id
                and self._can_afford_action(a)
            ])

    def get_state(self):
        ''' Returns a dictionary describing the game state

        See Coup.get_state.
        '''
        state = {
            'phase': START_OF_TURN,
            'whose_turn': self.player_id,
            'player_to_act': self.player_to_act()
        }
        if self.action:
            self.action.augment_state(state)
        return state

    def reset_state(self, state):
        ''' Resets the game to a given state

        Args:
            state (dict): a state dictionary such as returned by get_state

        Currently only supports a limited number of states, such as the start
        of a player's turn.

        Used for testing and debugging only.
        '''
        action = state['game']['action']
        target_player = state['game'].get('target_player')
        if target_player is not None:
            action = f'{action}:{target_player}'
        self._play_initial_action(action)
        self.action.reset_state(state)

class Action:
    ''' Base class for initial actions in the game, such as assassinate or tax

    This class implements the common logic for all actions which require a
    response from another player. This includes any costs that must be paid,
    blocks that are possible, and challenges.

    Subclass constructors should set up any initial response that is required:

    - If an action can be immediately challenged (all role-based actions), the
      subclass should assign a Challenge object to self.challenge.
    - If an action can be immediately blocked (only foreign aid), the subclass
      should assign a Block object to self.blocked.

    Subclasses must override the do_action function to perform the final
    behaviour of the action (steal, assassinate, etc.) and then end the turn
    by calling game.end_turn.
    '''
    def __init__(self, game, name, player_id, target_player=None):
        ''' Constructs an Action

        Args:
            game (Coup): the game being played
            name (str): name of the action
            player_id (int): id of the player whose turn it is
            target_player (int or None): id of the player who is targeted by the action
        '''
        self.game = game
        self.name = name
        self.player_id = player_id
        self.target_player = target_player
        self.cost = ACTION_COSTS.get(name, 0)
        # Assigned when a challenge is being processed
        self.challenge = None
        # Assigned when a block is being processed
        self.block = None
        # Set to a Reveal when the final action requires a reveal (assassinate, coup)
        self.final_action = None

    def init(self):
        ''' Initializes the action

        Must be called as soon as the player chooses the play this action.
        '''
        if not self.challenge:
            self.game.deduct_cash(self.player_id, self.cost)

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        Subclasses should not override this function.

        See Coup.player_to_act.
        '''
        if self.challenge:
            return self.challenge.player_to_act()
        elif self.block:
            return self.block.player_to_act()
        elif self.final_action:
            return self.final_action.player_to_act()
        else:
            return self.player_id

    def play_action(self, action):
        ''' Plays the given action

        Subclasses should not override this function.

        See Coup.play_action.
        '''
        if self.challenge:
            self.challenge.play_action(action)
        elif self.block:
            self.block.play_action(action)
        elif self.final_action:
            self.final_action.play_action(action)
        else:
            self.play_final_action(action)

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        See Coup.get_legal_actions.
        '''
        if self.challenge:
            return self.challenge.get_legal_actions()
        elif self.block:
            return self.block.get_legal_actions()
        elif self.final_action:
            return self.final_action.get_legal_actions()
        else:
            raise RuntimeError(f'Unexpected state: could not determine legal actions')

    def play_final_action(self, action):
        ''' Called when the player plays a final action after blocks/challenges

        Currently only applies to exchange, where the player must indicate which
        roles they want to keep.

        Subclasses do not normally need to implement this function.
        '''
        raise RuntimeError(f'Unexpected state: could not resolve action {action}')

    def augment_state(self, state):
        ''' Augments the game state to represent the state of this action

        Args:
            state (dict): the game state to be augmented

        Subclasses may override this function but they must call the superclass
        function before doing anything else.
        '''
        state['action'] = self.name
        if self.target_player is not None:
            state['target_player'] = self.target_player
        if self.challenge:
            state['phase'] = AWAITING_CHALLENGE
            self.challenge.augment_state(state)
        if self.block:
            state['phase'] = AWAITING_BLOCK
            self.block.augment_state(state)
        if self.final_action:
            self.final_action.augment_state(state)

    def resolve_challenge(self, action_allowed):
        ''' Called when the initial action challenge phase is completed

        Args:
            action_allowed (bool): whether the action can proceed

        If the action was not challenged, or if the action was challenged
        incorrectly (the player revealed the correct role), then
        resolve_challenge(True) will be called.

        If the action was challenged correctly (the player did not reveal the
        correct role), then resolve_challenge(False) will be called.

        Subclasses should not override this function and they should not call
        this function directly.
        '''
        self.challenge = None
        if action_allowed:
            self.game.deduct_cash(self.player_id, self.cost)
            # If the target player died due to challenge, end the turn
            if self.target_player is not None and not self.game.is_alive(self.target_player):
                self.game.end_turn()
            else:
                self.action_accepted()
                if not self.block:
                    self.do_action()
        else:
            self.game.end_turn()

    def resolve_block(self, action_allowed):
        ''' Called when the block phase is completed

        Args:
            action_allowed (bool): whether the action can proceed

        If the action was successfully blocked, then resolve_block(False)
        will be called. If the action was not blocked, or the block was
        challenged and the player did not reveal the correct role, then
        resolve_block(True) will be called.

        Subclasses should not override this function and they should not call
        this function directly.
        '''
        self.block = None
        if action_allowed:
            # If the target player died due to a challenged block, end the turn
            if self.target_player is not None and not self.game.is_alive(self.target_player):
                self.game.end_turn()
            else:
                self.do_action()
        else:
            self.game.end_turn()

    def action_accepted(self):
        ''' Called when the initial action is accepted

        This function is only called for actions that can be challenged. It is
        called when all players allow the action, or if it was challenged and
        the player revealed the correct role.

        Subclasses may override this function in the following cases:

        - If the action can be blocked, the Block should be assigned
        - If the action has a cost, the credits should be deducted here
        '''
        pass

    def do_action(self):
        ''' Called when the action can finally take place

        This function is only called once all challenges and blocks have been
        resolved.

        Subclasses must implement this function and they must end the turn
        after performing their action (or set up a final action which will
        end the turn later.)
        '''
        raise NotImplementedError(f'do_action in {self}')

    def reset_state(self, state):
        ''' Resets the game to a given state

        Args:
            state (dict): a state dictionary such as returned by get_state

        Currently only supports a limited number of states, such as the start
        of a player's turn.

        Used for testing and debugging only.
        '''
        if self.block:
            assert state['game']['phase'] == AWAITING_BLOCK
        elif self.challenge:
            assert state['game']['phase'] == AWAITING_CHALLENGE
        else:
            assert False, 'Cannot reset state'

class Challenge:
    ''' Handles all player actions relating to challenges.

    A Challenge is constructed whenever a player claims a role, either when
    taking his initial action or blocking another player.

    When the challenge is resolved, calls parent.challenge_resolved.
    '''
    def __init__(self, parent, challenged_player, role):
        ''' Constructs a Challenge

        Args:
            parent (Action or Block): the phase which initiated the challenge
            challenged_player (int): id of the player who claims the role
            role (str): the role that is claimed by the player
        '''
        self.parent = parent
        self.game = parent.game
        self.challenged_player = challenged_player
        self.role = role
        # The player who acts next (each player gets a chance to challenge)
        self.player_id = self.game.get_next_player(challenged_player)
        # The responses made so far to the challenge
        self.responses = {}
        # Assigned once all players have chosen whether to challenge
        self.reveal = None
        # True means the challenger(s) were right (challenged player does not
        # have the claimed role); None means we don't yet know if they were right
        self.challenge_correct = None
        # Record the role claim in the player's history
        self.game.trace_claim(challenged_player, role)

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        See Coup.player_to_act.
        '''
        if self.reveal:
            return self.reveal.player_to_act()
        else:
            return self.player_id

    def play_action(self, action):
        ''' Plays the given action

        See Coup.play_action.
        '''
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
                    self.reveal = Reveal(self, self.challenged_player, PROVE_CHALLENGE)
                else:
                    # Nobody challenged
                    self.parent.resolve_challenge(True)
            else:
                self.player_id = next_player_id

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        See Coup.get_legal_actions.
        '''
        if self.reveal:
            return self.reveal.get_legal_actions()
        else:
            return [PASS, CHALLENGE]

    def augment_state(self, state):
        ''' Augments the game state to represent the state of the challenge

        See Turn.get_state and Action.augment_state.
        '''
        if self.reveal:
            self.reveal.augment_state(state)

    def after_reveal(self, revealed_role):
        ''' Called when a player has revealed a role

        This can be when proving a challenge, when losing an incorrect
        challenge, or after being assassinated or couped.
        '''
        if self.challenge_correct is None:
            # The first reveal, where the challenged player proves whether he has the role
            self.challenge_correct = (revealed_role != self.role)
            if self.challenge_correct:
                # Challenged player did not have the role, he loses the card,
                # and the challenge is resolved
                self.game.reveal_role(self.challenged_player, revealed_role)
                self.game.trace_lost_challenge(self.challenged_player, self.role)
                self.parent.resolve_challenge(False)
            else:
                # Challenged player had the role, he gets a new card
                self.game.replace_role(self.challenged_player, revealed_role)
                # The challenger(s) were not successful and they must now reveal in turn
                self._reveal_next_challenger(self.challenged_player)
        else:
            # A challenger has revealed. Move to the next challenger, if there is one
            assert not self.challenge_correct
            self.game.reveal_role(self.reveal.player_id, revealed_role)
            self._reveal_next_challenger(self.reveal.player_id)

    def _reveal_next_challenger(self, player_id):
        ''' Called when the challenge failed and the challengers must reveal

        Args:
            player_id (int): the last player to act

        The first time this function is called, player_id is challenged_player,
        and the next player to act is the first challenger, who must reveal an
        influence. Each call after that, player_id is the previous challenger
        and we check if there is another challenger who must also reveal next.
        '''
        while True:
            player_id = self.game.get_next_player(player_id)
            if player_id == self.challenged_player:
                # All the players have been visited
                self.parent.resolve_challenge(True)
                break
            elif self.responses[player_id] == CHALLENGE:
                # Another challenger must reveal
                phase_name = CORRECT_CHALLENGE if self.challenge_correct else INCORRECT_CHALLENGE
                self.reveal = Reveal(self, player_id, phase_name)
                break
            # Else this player allowed, so move on to the next player

class Block:
    ''' Handles all player actions relating to blocks
    '''
    def __init__(self, action, roles, blocking_player_id=None):
        ''' Constructs a Block

        Args:
            action (Action): the action which initiated the block
            roles (list of str): the roles that may block the action
            blocking_player_id (int or None): the player who may block

        For most actions only the player who is being attacked may block, and
        blocking_player hods the id of this player.

        Foreign aid may be blocked by any player, so blocking_player_id is
        None.
        '''
        self.action = action
        self.game = action.game
        self.blocking_player_id = blocking_player_id
        if blocking_player_id is not None:
            # A specific player may block
            self.player_id = blocking_player_id
        else:
            # Everyone may block in turn
            self.player_id = self.game.get_next_player(action.player_id)
        self.roles = roles if type(roles) == list else [roles]
        self.responses = {}
        self.challenge = None

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        See Coup.player_to_act.
        '''
        if self.challenge:
            return self.challenge.player_to_act()
        else:
            return self.player_id

    def augment_state(self, state):
        ''' Augments the game state to represent the state of the block

        See Turn.get_state and Action.augment_state.
        '''
        if self.challenge:
            state['phase'] = AWAITING_BLOCK_CHALLENGE
            state['blocked_with'] = self.challenge.role
            state['blocking_player'] = self.challenge.challenged_player
            self.challenge.augment_state(state)

    def play_action(self, action):
        ''' Plays the given action

        See Coup.play_action.
        '''
        if self.challenge:
            self.challenge.play_action(action)
            return
        self.responses[self.player_id] = self._extract_response(action)
        if self.blocking_player_id is not None:
            # Only the target player blocks
            self._execute_block()
        else:
            # Anyone can block, so go to the next player
            next_player_id = self.game.get_next_player(self.player_id)
            if next_player_id == self.action.player_id:
                self._execute_block()
            else:
                self.player_id = next_player_id

    def _extract_response(self, action):
        ''' Extracts the player response from the given action

        Args:
            action (str): either 'pass' or e.g. 'block:duke'

        Returns:
            (str): the action without the 'block:' prefix: either 'pass' or a role name
        '''
        if action == PASS:
            return action
        elif action.startswith(BLOCK + ':'):
            role = action[len(BLOCK + ':'):]
            if role in self.roles:
                return role
        raise IllegalAction(f'Unknown action {action}')

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        See Coup.get_legal_actions.
        '''
        if self.challenge:
            return self.challenge.get_legal_actions()
        else:
            return [PASS] + [block(r) for r in self.roles]

    def _execute_block(self):
        ''' Called when all players have decided whether to block

        If no one blocked, the action can proceed. If more than one player
        wants to block, a player is picked at random by the dealer. A challenge
        then begins on the blocking player.
        '''
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
        ''' Called when the challenge is resolved

        Args:
            block_allowed (bool): True if the action was successfully blocked

        If the challenge was correct (the blocking player did not reveal the
        correct role), resolve_block(False) is called on the action. If the
        block was not challenged or it was incorrectly challenged,
        resolve_block(True) is called.
        '''
        action_allowed = not block_allowed
        self.action.resolve_block(action_allowed)

class Reveal:
    ''' Handles all player actions relating to revealing an influence
    '''
    def __init__(self, parent, player_id, phase_name):
        ''' Constructs a Block

        Args:
            parent (Action or Challenge): the phase which initiated the reveal
            player_id (int): the player who must reveal
            phase_name (str): name used to represent the phase in the game state
        '''
        # Parent could be an Action or Challenge
        self.parent = parent
        self.player_id = player_id
        self.phase_name = phase_name
        self.game = parent.game

    def play_action(self, action):
        ''' Plays the given action

        See Coup.play_action.
        '''
        if not action.startswith(REVEAL + ':'):
            raise IllegalAction(f'Unknown action {action}')
        role = action[len(REVEAL + ':'):]
        if not self.game.player_has_role(self.player_id, role):
            raise IllegalAction(f'Player {self.player_id} does not have role {role}')
        self.game.trace_reveal(self.player_id, role)
        self.parent.after_reveal(role)

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        See Coup.get_legal_actions.
        '''
        return [reveal(r) for r in self.game.get_influence(self.player_id)]

    def augment_state(self, state):
        ''' Augments the game state to represent the state of the reveal

        See Turn.get_state and Action.augment_state.
        '''
        state['phase'] = self.phase_name

    def player_to_act(self):
        ''' Returns the id of the player who acts next

        See Coup.player_to_act.
        '''
        return self.player_id

class ForeignAid(Action):
    ''' Implements the foreign aid action
    '''
    def __init__(self, game, player_id):
        super().__init__(game, FOREIGN_AID, player_id)
        # Foreign aid cannot be challenged but it can be blocked
        self.block = Block(self, DUKE)

    def do_action(self):
        # If not blocked, the player gets two credits and his turn ends
        self.game.add_cash(self.player_id, 2)
        self.game.end_turn()

class Steal(Action):
    ''' Implements the steal action
    '''
    def __init__(self, game, player_id, target_player):
        ''' First, there is a chance to challenge the player's captain claim
        '''
        super().__init__(game, STEAL, player_id, target_player)
        self.challenge = Challenge(self, player_id, CAPTAIN)

    def action_accepted(self):
        ''' After the challenge phase, the target player may block
        '''
        self.block = Block(self, [AMBASSADOR, CAPTAIN], self.target_player)

    def do_action(self):
        ''' If not blocked, the player steals 2 credits and his turn ends

        If the target player has only 1 credit, the theif only gains 1 credit.
        '''
        cash = self.game.deduct_cash(self.target_player, 2)
        self.game.add_cash(self.player_id, cash)
        self.game.end_turn()

class Tax(Action):
    ''' Implements the tax action
    '''
    def __init__(self, game, player_id):
        super().__init__(game, TAX, player_id)
        ''' First, there is a chance to challenge the player's duke claim
        '''
        self.challenge = Challenge(self, player_id, DUKE)

    def do_action(self):
        ''' If not challenged, the player gets 3 credits and his turn ends
        '''
        self.game.add_cash(self.player_id, 3)
        self.game.end_turn()

class AssassinateAction(Action):
    ''' Implements the assassinate action
    '''
    def __init__(self, game, player_id, target_player):
        super().__init__(game, ASSASSINATE, player_id, target_player)
        ''' First, there is a chance to challenge the player's assassin claim
        '''
        self.challenge = Challenge(self, player_id, ASSASSIN)

    def action_accepted(self):
        ''' After the challenge phase, the target player may block
        '''
        self.block = Block(self, [CONTESSA], self.target_player)

    def do_action(self):
        ''' If not blocked, the target player must reveal an influence
        '''
        self.final_action = Reveal(self, self.target_player, DIRECT_ATTACK)

    def after_reveal(self, revealed_role):
        ''' The target player loses the revealed influence, and the turn ends
        '''
        self.game.reveal_role(self.target_player, revealed_role)
        self.game.end_turn()

class ExchangeAction(Action):
    ''' Implements the exchange action
    '''
    def __init__(self, game, player_id):
        super().__init__(game, EXCHANGE, player_id)
        ''' First, there is chance to challenge the player's ambassador claim
        '''
        self.challenge = Challenge(self, player_id, AMBASSADOR)
        self.drawn_roles = None

    def do_action(self):
        ''' If not challenged, the player gets new cards to choose from
        '''
        assert self.drawn_roles is None
        self.drawn_roles = self.game.dealer.deal_cards(2)

    def get_legal_actions(self):
        ''' Returns a list of legal actions for the current player

        Overrides Action.get_legal_actions to return the legal choices that the
        player can make about which roles to keep.
        '''
        if self.drawn_roles:
            existing_roles = self.game.get_influence(self.player_id)
            pool = existing_roles + self.drawn_roles
            return get_keep_actions(pool, len(existing_roles))
        else:
            return super().get_legal_actions()

    def play_final_action(self, action):
        ''' Called when the player has chosen which roles to keep

        The player's hidden influence is updated with the chosen cards. The
        other cards are returned to the deck, and the turn ends.
        '''
        new_roles = keep_decode(action)
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
        self.game.dealer.replace_cards(pool)
        self.game.trace_exchange(self.player_id)
        self.game.end_turn()

    def augment_state(self, state):
        ''' Augments the game state to represent the state of the action

        Overrides Action.augment_state.
        '''
        super().augment_state(state)
        if self.drawn_roles:
            state['phase'] = CHOOSE_NEW_ROLES
            state['drawn_roles'] = list(self.drawn_roles)

class CoupAction(Action):
    ''' Implements the coup action
    '''
    def __init__(self, game, player_id, target_player):
        ''' There is no challenge or block, only a reveal
        '''
        super().__init__(game, COUP, player_id, target_player)
        self.final_action = Reveal(self, target_player, DIRECT_ATTACK)

    def after_reveal(self, revealed_role):
        ''' The target player loses the revealed influence, and the turn ends
        '''
        self.game.reveal_role(self.target_player, revealed_role)
        self.game.end_turn()
