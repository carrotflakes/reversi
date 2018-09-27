import random

class AgentRandom:

    def __init__(self):
        pass

    def think(self, game):
        return random.choice(game.candidates())
