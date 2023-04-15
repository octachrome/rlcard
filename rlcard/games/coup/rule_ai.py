# Copyright 2023 Chris Brown
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .constants import *

# Constants
NUM_ROLES = 3               # Number of each type of card in the deck
RANKED_ROLES = [            # Roles ordered by value
    'duke',
    'assassin',
    'captain',
    'inquisitor',
    'contessa',
    'ambassador'
]
ROLE_WEIGHTS = {            # The weights show how likely a role is to be revealed by AI
    'duke': 3,              # E.g. ambassador is 3 times more likely to be revealed than duke
    'assassin': 4,
    'captain': 5,
    'inquisitor': 6,
    'contessa': 6,
    'ambassador': 9
}

class CoupRuleAI:
    def __init__(
            self,
            state,
            np_random,
            search_horizon=7,           # How many moves the AI will search ahead for an end-game
            chance_to_bluff=0.5,        # Fraction of games in which the AI will bluff
            chance_to_challenge=0.1     # Fraction of turns in which the AI will challenge (not in the end-game)
        ):
        self.state = state
        self.np_random = np_random
        self.search_horizon = search_horizon
        self.chance_to_bluff = chance_to_bluff
        self.chance_to_challenge = chance_to_challenge
        self.game_state = state['game']
        self.players = state['players']
        self.phase = self.game_state['phase']
        self.whose_turn = self.game_state['whose_turn']
        self.action = self.game_state['action']
        self.target_id = self.game_state.get('target_player')
        self.blocking_role = self.game_state.get('blocked_with')
        self.blocking_player = self.game_state.get('blocking_player')
        self.ai_id = self.game_state['player-to-act']
        self.ai_player = state['players'][self.ai_id]

    def get_action(self):
        self.bluff_choice = self.np_random.rand() < self.chance_to_bluff
        if self.phase == START_OF_TURN:
            return self._play_our_turn()
        elif self.phase == AWAITING_CHALLENGE or self.phase == AWAITING_BLOCK:
            return self._respond_to_action()
        elif self.phase == AWAITING_BLOCK_CHALLENGE:
            return self._respond_to_block()
        elif self.phase in REVEAL_PHASES:
            return self._reveal_by_probability()
        elif self.phase == CHOOSE_NEW_ROLES:
            return self._exchange()

    def _respond_to_action(self):
        if (
            self.phase in [AWAITING_CHALLENGE, AWAITING_BLOCK] and
            self.action == STEAL and
            self.target_id == self.ai_id and
            self.ai_player['cash'] == 0
        ):
            # If someone wants to steal nothing from us, go ahead
            return PASS

        # Don't bluff in the final action response - it will just get challenged
        if self.phase == AWAITING_CHALLENGE:
            if self._should_challenge():
                return CHALLENGE

        if self.phase == AWAITING_BLOCK:
            blocking_role = self._get_blocking_role()
            if blocking_role:
                # If we have the role, may as well block
                return block(blocking_role)

            blocking_role = self._get_bluffed_blocking_role()
            if blocking_role:
                return block(blocking_role)

        return PASS

    def _respond_to_block(self):
        if self._should_challenge():
            return CHALLENGE
        else:
            return PASS

    def _bluff_was_called(self, role):
        called = False
        for tp, rl in self.ai_player['trace']:
            if tp == 'lost_challenge' and rl == role:
                called = True
            elif tp == 'exchange':
                called = False
        return called

    def _should_challenge(self):
        claimed_role = ACTION_ROLES.get(self.action, None) if self.phase == AWAITING_CHALLENGE else self.blocking_role
        challenged_player = self.whose_turn if self.phase == AWAITING_CHALLENGE else self.blocking_player

        # Challenge if somebody claims to a have role that was revealed 3 times or we have the rest of them
        used_roles = self._count_revealed_roles(claimed_role)
        for role in self.ai_player['hidden']:
            if role == claimed_role:
                used_roles += 1
        if used_roles == NUM_ROLES:
            return True
        if self._bluff_was_called(challenged_player, claimed_role):
            return True

        # Are we being assassinated on our last influence?
        if (
            self.phase == AWAITING_CHALLENGE and
            self.target_id == self.ai_id and
            self.action == ASSASSINATE and
            len(self.ai_player['hidden']) == 1
        ):
            # We will bluff contessa unless they are all revealed or claimed
            contessas = self._count_revealed_roles(CONTESSA) + self._count_role_claims(CONTESSA)
            if contessas >= NUM_ROLES:
                return True

            # Challenge if we already bluffed contessa and were caught
            if self._bluff_was_called(self.ai_id, CONTESSA):
                return True

            # Otherwise we will bluff contessa
            return False

        if not self._action_is_worth_challenging():
            return False

        if self._is_end_game():
            result = self._simulate()
            # Challenge if the opponent would otherwise win soon.
            if result < 0:
                return True
            # Don't bother challenging if we're going to win anyway.
            if result > 0:
                return False

        # Challenge at random.
        return self.np_random.rand() < self.chance_to_challenge

    def _action_is_worth_challenging(self):
        # Worth challenging anyone drawing tax.
        if self.action == TAX:
            return True
        # Worth challenging someone assassinating us or stealing from us,
        # Or someone trying to block us from assassinating or stealing.
        if (
            (self.action == STEAL or self.action == ASSASSINATE) and
            (self.whose_turn == self.ai_id or self.target_id == self.ai_id)
        ):
            return True

        return False

    def _count_revealed_roles(self, role):
        count = 0
        for p in self.players:
            for r in p['revealed']:
                if r == role:
                    count += 1
        return count

    def _is_end_game(self):
        opponents = self._players_by_strength()
        return len(opponents) == 1

    # This function adds randomness to AI decision making process
    # Even if some decision seem a good idea, sometimes AI will make a different call
    # Otherwise AIs are predictable and human opponents can predict their moves
    def _randomize_choice(self):
        # At the end AIs won't make random choices as it might make them lose
        if self._is_end_game() and len(self.ai_player['hidden']) == 1:
            return False
        return self.np_random.rand() < 0.1

    def _get_blocking_role(self):
        blocking_roles = ACTION_BLOCKS.get(self.action, [])
        for r in blocking_roles:
            if r in self.ai_player['hidden']:
                return r
        return None

    def _get_bluffed_blocking_role(self):
        blocking_roles = ACTION_BLOCKS.get(self.action, [])
        if len(blocking_roles) == 0:
            return None
        blocking_roles = blocking_roles[:]
        self.np_random.shuffle(blocking_roles)
        for r in blocking_roles:
            if self._should_bluff(r):
                return r
        # No bluffs are appropriate.
        return None

    def _play_our_turn(self):
        influence = self.ai_player['hidden']
        assassin_target = self._choose_target(ASSASSINATE)
        captain_target = self._choose_target(STEAL)
        if self.ai_player['cash'] >= 7:
            return coup(self._strongest_player())
        elif ASSASSIN in influence and self.ai_player['cash'] >= 3 and assassin_target is not None and not self._randomize_choice():
            return assassinate(assassin_target)
        elif CAPTAIN in influence and captain_target is not None and not self._randomize_choice():
            return steal(captain_target)
        elif DUKE in influence and not self._randomize_choice():
            return TAX
        elif self._count_revealed_roles(DUKE) == NUM_ROLES and CAPTAIN not in influence and not self._randomize_choice():
            return FOREIGN_AID
        else:
            # No good moves - check whether to bluff.
            possible_bluffs = []
            if self.ai_player['cash'] >= 3 and assassin_target is not None and self._should_bluff(ASSASSIN):
                possible_bluffs.append(assassinate(assassin_target))
            if captain_target is not None and self._should_bluff(CAPTAIN):
                possible_bluffs.append(steal(captain_target))
            if self._should_bluff(DUKE):
                possible_bluffs.append(TAX)
            if len(possible_bluffs) > 0 and not self._randomize_choice():
                # Randomly select one.
                return self.np_random.choice(possible_bluffs)
            else:
                # No bluffing.
                if ASSASSIN not in influence and not self._randomize_choice():
                    # If we don't have a captain, duke, or assassin, then exchange.
                    return EXCHANGE
                else:
                    # We have an assassin, but can't afford to assassinate.
                    if self._count_revealed_roles(DUKE) == NUM_ROLES:
                        return FOREIGN_AID
                    else:
                        return INCOME

    def _should_bluff(self, role):
        if self._bluff_was_called(role):
            # Don't bluff a role that we previously bluffed and got caught out on.
            return False
        if self._count_revealed_roles(role) == NUM_ROLES:
            # Don't bluff a role that has already been revealed three times.
            return False
        if role == CONTESSA and self.action == ASSASSINATE and len(self.ai_player['hidden']) == 1:
            # Bluff contessa if only 1 influence left as otherwise we lose
            return True
        if not self.bluff_choice and role not in self._get_claimed_roles(self.ai_id):
            # We shall not bluff (unless we already claimed this role earlier).
            return False
        if self._count_our_role_claims() > 2 and role not in self._get_claimed_roles(self.ai_id):
            # We have already bluffed a different role: don't bluff any more.
            return False
        # For now we can only simulate against a single opponent.
        if self._is_end_game() and self._simulate(role) > 0:
            # If bluffing would win us the game, we will probably be challenged, so don't bluff.
            return False
        else:
            # We will bluff.
            return True

    def _count_our_role_claims(self):
        return len(self._get_claimed_roles(self.ai_player))

    def _count_role_claims(self, role):
        count = 0
        for player_id in range(len(self.players)):
            if player_id == self.ai_id:
                continue
            if role in self._get_claimed_roles(player_id):
                count += 1
        return count

    def _reveal_by_probability(self):
        roles = [r for r in self.ai_player['hidden'] for _ in range(ROLE_WEIGHTS[r])]
        role = self.np_random.choice(roles)
        return reveal(role)

    def _choose_target(self, action):
        return [
            p for p in self._players_by_strength()
            if not self._player_can_block(p, action)
        ]

    def _player_can_block(self, player_id, action):
        claimed_roles = set(self._get_claimed_roles(player_id))
        for role in ACTION_BLOCKS.get(action, []):
            if role in claimed_roles:
                return True
        return False

    def _get_claimed_roles(self, player_id):
        claims = set()
        for tp, role in self.players[player_id]['trace']:
            if tp == 'claim':
                claims.add(role)
            elif tp == 'reveal' or tp == 'lost_challenge':
                claims.remove(role)
            elif tp == 'exchange':
                claims.clear()
        return list(claims)

    def _strongest_player(self):
        return self._players_by_strength()[0]

    # Rank opponents by influence first, and money second
    def _players_by_strength(self):
        # Start with live opponents who are not ourselves
        players = [
            p for p in self.players
            if p != self.ai_player and len(p['hidden']) > 0
        ]
        def strength(player):
            return len(player['hidden']) * 20 + player['cash']
        players.sort(key=strength, reverse=True)
        strong_players = [p for p in players if strength(p) == strength(players[0])]
        # Break ties at random
        return self.np_random.choice(strong_players)

    def _exchange(self):
        chosen = []
        needed = len(self.ai_player['hidden'])
        available = list(self.game_state['drawn_roles'])
        # Try to choose unique roles in order of preference
        for r in RANKED_ROLES:
            if r in chosen:
                continue
            if r in available:
                chosen.append(r)
                available.remove(r)
            if len(chosen) == needed:
                break
        if len(chosen) < needed:
            # We must choose duplicate roles
            for r in RANKED_ROLES:
                if r in available:
                    chosen.append(r)
                    available.remove(r)
                if len(chosen) == needed:
                    break
        assert len(chosen) == needed
        return keep(chosen)

    def _simulate(self, bluffed_role=None):
        opponent_id = self._strongest_player()
        return Simulator(self, opponent_id, bluffed_role).simulate()

class Simulator:
    ''' Simulates us and the remaining player playing their best moves to see who would win.
    If we win, return 1; if the opponent wins, -1; if no one wins within the search horizon, 0.
    Limitation: if a player loses an influence, it acts as if the player can still play either role.
    Limitation: doesn't take foreign aid.
    '''
    def __init__(self, ai, opponent_id, bluffed_role):
        self.ai = ai
        self.opponent_id = opponent_id
        self.opponent = ai.players[opponent_id]
        self.bluffed_role = bluffed_role
        self.cash = [
            self.opponent['cash'],
            ai.ai_player['cash']
        ]
        self.influence_count = [
            len(self.opponent['hidden']),
            len(ai.ai_player['hidden'])
        ]
        self.roles = [
            ai._get_claimed_roles(opponent_id),
            ai.ai_player['hidden']
        ]
        if bluffed_role:
            self.roles[1] += [bluffed_role]

    def _other_can_block(self, action):
        return len(set(self.roles[self.other]).intersection(set(ACTION_BLOCKS.get(action, [])))) > 0

    def _can_steal(self):
        return CAPTAIN in self.roles[self.turn] >= 0 and not self._other_can_block(STEAL)

    def _steal(self):
        if self.cash[self.other] < 2:
            self.cash[self.turn] += self.cash[self.other]
            self.cash[self.other] = 0
        else:
            self.cash[self.turn] += 2
            self.cash[self.other] -= 2

    def _can_assassinate(self):
        return ASSASSIN in self.roles[self.turn] >= 0 and not self._other_can_block(ASSASSINATE)

    def _assassinate(self):
        self.cash[self.turn] -= 3
        self.influence_count[self.other] -= 1

    def _can_tax(self):
        return DUKE in self.roles[self.turn]

    def _tax(self):
        self.cash[self.turn] += 3

    def _income(self):
        self.cash[self.turn] += 1

    def _coup(self):
        self.cash[self.turn] -= 7
        self.influence_count[self.other] -= 1

    def simulate(self):
        # Apply the pending move
        if self.ai.phase == AWAITING_CHALLENGE or self.ai.phase == AWAITING_BLOCK:
            # The opponent is playing an action; simulate it (unless we are blocking), then run from our turn
            i = 0
            self.turn = 0
            self.other = 1
            if not self.bluffed_role:
                action = self.ai.action
                if action == STEAL:
                    self._steal()
                elif action == ASSASSINATE:
                    self._assassinate();
                elif action == TAX:
                    self._tax()
                else:
                    raise Exception('unexpected initial action: ' + action)
        elif self.ai.phase == AWAITING_BLOCK_CHALLENGE:
            # The opponent is blocking our action; run from the opponent's turn
            i = 1
        elif self.ai.phase == START_OF_TURN:
            # It's our self.turn and we are considering a bluff; run from our turn
            i = 0
        while i < self.ai.search_horizon:
            i += 1
            self.turn = i % 2
            self.other = (i + 1) % 2
            if self.influence_count[0] == 0:
                return 1
            if self.influence_count[1] == 0:
                return -1
            if self._can_assassinate() and self.cash[self.turn] >= 3:
                self._assassinate()
            elif self.cash[self.turn] >= 7:
                self._coup()
            elif self._can_steal() and self.cash[self.other] > 0:
                # To do: only steal if cash >= 2, e.g., if they also have the duke?
                self._steal()
            elif self._can_tax():
                self._tax()
            else:
                self._income()
        # We don't know if we would win, but don't do anything rash
        return 0
