class AgentCUI:

    def __init__(self):
        pass

    def think(self, game):
        print('your turn({})'.format('o x'[game.turn+1]))
        candidates = game.candidates()

        for y in range(8):
            for x in range(8):
                if (x, y) in candidates:
                    print('{:2d}'.format(candidates.index((x, y))), end='')
                else:
                    print([' o', ' _', ' x'][game.board[y,x]+1], end='')
            print()

        i = -1
        while not (0 <= i < len(candidates)):
            print('> ', end='')
            try:
                i = int(input())
            except:
                pass
        return candidates[i]
