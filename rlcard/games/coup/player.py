class CoupPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hidden = []
        self.revealed = []
        self.cash = 2