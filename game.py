import numpy as np

class Game:

    def __init__(self):
        self.board = np.array([[0] * 8 for _ in range(8)], dtype=np.int64)
        self.board[3,3] = self.board[4,4] = -1
        self.board[3,4] = self.board[4,3] = 1
        self.turn = 1
        self.stone_count = 4
        self.end = False

    def step(self, x, y):
        if self.board[y,x] != 0:
            print(x, y)
            raise Exception()

        self.board[y,x] = self.turn

        # left
        flip = False
        for x_ in range(x-1, -1, -1):
            if self.board[y,x_] == 0:
                break
            elif self.board[y,x_] == self.turn:
                flip = True
                break
        if flip:
            for x_ in range(x-1, -1, -1):
                if self.board[y,x_] == self.turn:
                    break
                self.board[y,x_] = self.turn

        # left top
        flip = False
        for x_, y_ in zip(range(x-1, -1, -1), range(y-1, -1, -1)):
            if self.board[y_,x_] == 0:
                break
            elif self.board[y_,x_] == self.turn:
                flip = True
                break
        if flip:
            for x_, y_ in zip(range(x-1, -1, -1), range(y-1, -1, -1)):
                if self.board[y_,x_] == self.turn:
                    break
                self.board[y_,x_] = self.turn

        # top
        flip = False
        for y_ in range(y-1, -1, -1):
            if self.board[y_,x] == 0:
                break
            elif self.board[y_,x] == self.turn:
                flip = True
                break
        if flip:
            for y_ in range(y-1, -1, -1):
                if self.board[y_,x] == self.turn:
                    break
                self.board[y_,x] = self.turn

        # right top
        flip = False
        for x_, y_ in zip(range(x+1, 8), range(y-1, -1, -1)):
            if self.board[y_,x_] == 0:
                break
            elif self.board[y_,x_] == self.turn:
                flip = True
                break
        if flip:
            for x_, y_ in zip(range(x+1, 8), range(y-1, -1, -1)):
                if self.board[y_,x_] == self.turn:
                    break
                self.board[y_,x_] = self.turn

        # right
        flip = False
        for x_ in range(x+1, 8):
            if self.board[y,x_] == 0:
                break
            elif self.board[y,x_] == self.turn:
                flip = True
                break
        if flip:
            for x_ in range(x+1, 8):
                if self.board[y,x_] == self.turn:
                    break
                self.board[y,x_] = self.turn

        # right bottom
        flip = False
        for x_, y_ in zip(range(x+1, 8), range(y+1, 8)):
            if self.board[y_,x_] == 0:
                break
            elif self.board[y_,x_] == self.turn:
                flip = True
                break
        if flip:
            for x_, y_ in zip(range(x+1, 8), range(y+1, 8)):
                if self.board[y_,x_] == self.turn:
                    break
                self.board[y_,x_] = self.turn

        # bottom
        flip = False
        for y_ in range(y+1, 8):
            if self.board[y_,x] == 0:
                break
            elif self.board[y_,x] == self.turn:
                flip = True
                break
        if flip:
            for y_ in range(y+1, 8):
                if self.board[y_,x] == self.turn:
                    break
                self.board[y_,x] = self.turn

        # left bottom
        flip = False
        for x_, y_ in zip(range(x-1, -1, -1), range(y+1, 8)):
            if self.board[y_,x_] == 0:
                break
            elif self.board[y_,x_] == self.turn:
                flip = True
                break
        if flip:
            for x_, y_ in zip(range(x-1, -1, -1), range(y+1, 8)):
                if self.board[y_,x_] == self.turn:
                    break
                self.board[y_,x_] = self.turn

        def skip():
            for y in range(8):
                for x in range(8):
                    if self.can_put(x, y):
                        return False
            return True

        self.stone_count += 1
        self.turn *= -1
        if skip():
            # skip the turn
            self.turn *= -1
            if skip():
                self.end = True

    def candidates(self):
        return [
            (x, y)
            for x in range(8)
            for y in range(8)
            if self.can_put(x, y)
        ]

    def can_put(self, x, y):
        if self.board[y,x] != 0:
            return False

        for xy in [
                zip(range(x-1, -1, -1), [y] * 8),
                zip(range(x-1, -1, -1), range(y-1, -1, -1)),
                zip([x] * 8, range(y-1, -1, -1)),
                zip(range(x+1, 8), range(y-1, -1, -1)),
                zip(range(x+1, 8), [y] * 8),
                zip(range(x+1, 8), range(y+1, 8)),
                zip([x] * 8, range(y+1, 8)),
                zip(range(x-1, -1, -1), range(y+1, 8))
        ]:
            d = 0
            for x_, y_ in xy:
                if self.board[y_,x_] == 0:
                    break
                elif self.board[y_,x_] == self.turn:
                    if d > 0:
                        return True
                    break
                d += 1

        return False

    def judge(self):
        black = sum(self.board[y,x] == 1 for y in range(8) for x in range(8))
        white = sum(self.board[y,x] == -1 for y in range(8) for x in range(8))
        if black == white:
            return 0
        else:
            return (black > white) * 2 - 1


    def print(self):
        for y in range(8):
            for x in range(8):
                print('o_x'[self.board[y,x] + 1], end='')
            print('')

    def copy(self):
        game = Game()
        game.board = np.array(self.board, dtype=np.int64)
        game.turn = self.turn
        game.stone_count = self.stone_count
        game.end = self.end
        return game


if __name__ == '__main__':
    import random
    game = Game()
    print(game.candidates())
    game.print()
    while not game.end:
        game.step(*random.choice(game.candidates()))
        print('=' * 8)
        game.print()
