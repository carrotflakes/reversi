import random

class AgentAB:

    def __init__(self):
        pass

    def think(self, game):
        turn = game.turn
        def f(game, depth, alpha, beta):
            if game.end:
                return game.judge() * 100
            if depth == 0:
                # FIXME: too bad evaluation...
                b, w = game.count_black_white()
                return b - w
            if turn == game.turn:
                for pos in game.candidates():
                    g = game.copy()
                    g.step(*pos)
                    alpha = max(alpha, f(g, depth-1, alpha, beta))
                    if alpha >= beta:
                        break
                return alpha
            else:
                for pos in game.candidates():
                    g = game.copy()
                    g.step(*pos)
                    beta = min(beta, f(g, depth-1, alpha, beta))
                    if alpha >= beta:
                        break
                return beta

        alpha = -101
        best_pos = None
        for pos in game.candidates():
            g = game.copy()
            g.step(*pos)
            value = f(g, 3, alpha, 100) * turn
            if alpha < value:
                alpha = value
                best_pos = pos
        return pos
