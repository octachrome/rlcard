DUKE = 'duke'
CAPTAIN = 'captain'
ASSASSIN = 'assassin'
CONTESSA = 'contessa'
AMBASSADOR = 'ambassador'

ALL_ROLES = set([
  DUKE,
  CAPTAIN,
  ASSASSIN,
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
PASS = 'pass'
CHALLENGE = 'challenge'

UNTARGETED_ACTIONS = set([
  INCOME,
  FOREIGN_AID,
  TAX,
  EXCHANGE
])

TARGETED_ACTIONS = set([
  STEAL,
  ASSASSINATE,
  COUP
])
