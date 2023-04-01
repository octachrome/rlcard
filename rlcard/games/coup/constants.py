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

''' Constants relating to the game of Coup
'''

# Role cards
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

# Role cards that can block an action
BLOCKING_ROLES = sorted([
    DUKE,
    CAPTAIN,
    CONTESSA,
    AMBASSADOR
])

# Initial actions
INCOME = 'income'
FOREIGN_AID = 'foreign_aid'
STEAL = 'steal'
TAX = 'tax'
ASSASSINATE = 'assassinate'
EXCHANGE = 'exchange'
COUP = 'coup'

# Response actions
BLOCK = 'block'
PASS = 'pass'
CHALLENGE = 'challenge'
REVEAL = 'reveal'
KEEP = 'keep'

# Actions which do not target a specific player
UNTARGETED_ACTIONS = sorted([
    INCOME,
    FOREIGN_AID,
    TAX,
    EXCHANGE
])

# Actions which target a specific player
TARGETED_ACTIONS = sorted([
    STEAL,
    ASSASSINATE,
    COUP
])

# Costs of actions which require payment
ACTION_COSTS = {
    ASSASSINATE: 3,
    COUP: 7,
}

# Roles which can block each action
ACTION_BLOCKS = {
    ASSASSINATE: [CONTESSA],
    FOREIGN_AID: [DUKE],
    STEAL: [AMBASSADOR, CAPTAIN]
}

# Phases in the game
START_OF_TURN = 'start_of_turn'
AWAITING_CHALLENGE = 'awaiting_challenge'
AWAITING_BLOCK = 'awaiting_block'
AWAITING_BLOCK_CHALLENGE = 'awaiting_block_challenge'
CHOOSE_NEW_ROLES = 'choose_new_roles'           # Ambassador exchange
GAME_OVER = 'game_over'
# Phases which require the player to reveal a card
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

REVEAL_PHASES = sorted([
    PROVE_CHALLENGE,
    CORRECT_CHALLENGE,
    INCORRECT_CHALLENGE,
    DIRECT_ATTACK
])

class IllegalAction(Exception):
    ''' Thrown when a player (or agent) tries to play an illegal action
    '''
    pass

def block(role):
    ''' Encodes the block action

    Args:
        role (str): the role to block with

    Returns:
        (str) an action string
    '''
    if not role in BLOCKING_ROLES:
        raise IllegalAction(f'Cannot block with {role}')
    return BLOCK + ':' + role

def reveal(role):
    ''' Encodes the reveal action

    Args:
        role (str): the role to reveal

    Returns:
        (str) an action string
    '''
    if not role in ALL_ROLES:
        raise IllegalAction(f'Unknown role {role}')
    return REVEAL + ':' + role

def keep(roles):
    ''' Encodes the keep action, used at the end of an exchange

    Args:
        roles (list of str): the roles to keep

    Returns:
        (str) an action string
    '''
    return KEEP + ':' + ','.join(sorted(roles))

def keep_decode(action):
    ''' Decodes the keep action, used at the end of an exchange

    Args:
        action (str): an action string

    Returns:
        (list of str): the roles that the player wishes to keep
    '''
    if not action.startswith(KEEP + ':'):
        raise IllegalAction(f'Unknown action {action}')
    return action[len(KEEP + ':'):].split(',')

def assassinate(target):
    return ASSASSINATE + ':' + target

def coup(target):
    return COUP + ':' + target

def steal(target):
    return STEAL + ':' + target
