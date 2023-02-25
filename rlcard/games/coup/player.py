class CoupPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hidden = []
        self.revealed = []
        self.cash = 2
        self.trace = []

    def get_state(self):
        return {
            'cash': self.cash,
            'hidden': sorted(self.hidden),
            'revealed': sorted(self.revealed),
            'trace': list(self.trace)
        }
