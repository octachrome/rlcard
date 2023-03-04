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

DUKE = 'duke'
CAPTAIN = 'captain'
ASSASSIN = 'assassin'
CONTESSA = 'contessa'
AMBASSADOR = 'ambassador'

ALL_ROLES = sorted([
    DUKE,
    CAPTAIN,
    ASSASSIN,
    CONTESSA,
    AMBASSADOR
])

BLOCKING_ROLES = sorted([
    DUKE,
    CAPTAIN,
    CONTESSA,
    AMBASSADOR
])

INCOME = 'income'
FOREIGN_AID = 'foreign_aid'
STEAL = 'steal'
TAX = 'tax'
ASSASSINATE = 'assassinate'
EXCHANGE = 'exchange'
COUP = 'coup'
BLOCK = 'block'
PASS = 'pass'
CHALLENGE = 'challenge'
REVEAL = 'reveal'
KEEP = 'keep'

UNTARGETED_ACTIONS = sorted([
    INCOME,
    FOREIGN_AID,
    TAX,
    EXCHANGE
])

TARGETED_ACTIONS = sorted([
    STEAL,
    ASSASSINATE,
    COUP
])

ACTION_COSTS = {
    ASSASSINATE: 3,
    COUP: 7,
}

START_OF_TURN = 'start_of_turn'
AWAITING_CHALLENGE = 'awaiting_challenge'
AWAITING_BLOCK = 'awaiting_block'
AWAITING_BLOCK_CHALLENGE = 'awaiting_block_challenge'
CHOOSE_NEW_ROLES = 'choose_new_roles'           # Ambassador exchange
GAME_OVER = 'game_over'
# The following phases all require the player to reveal a card
PROVE_CHALLENGE = 'prove_challenge'
CORRECT_CHALLENGE = 'correct_challenge'
INCORRECT_CHALLENGE = 'incorrect_challenge'
DIRECT_ATTACK = 'direct_attack'                 # Coup or assassination

PHASES = sorted([
    START_OF_TURN,
    AWAITING_CHALLENGE,
    AWAITING_BLOCK,
    AWAITING_BLOCK_CHALLENGE,
    CHOOSE_NEW_ROLES,
    PROVE_CHALLENGE,
    CORRECT_CHALLENGE,
    INCORRECT_CHALLENGE,
    DIRECT_ATTACK,
    GAME_OVER
])

class IllegalAction(Exception):
    pass

def block(role):
    if not role in BLOCKING_ROLES:
        raise IllegalAction(f'Cannot block with {role}')
    return BLOCK + ':' + role

def reveal(role):
    if not role in ALL_ROLES:
        raise IllegalAction(f'Unknown role {role}')
    return REVEAL + ':' + role

def keep(roles):
    return KEEP + ':' + ','.join(sorted(roles))

def keep_decode(action):
    if not action.startswith(KEEP + ':'):
        raise IllegalAction(f'Unknown action {action}')
    return action[len(KEEP + ':'):].split(',')
