class AgentCUI:

    def __init__(self):
        pass

    def think(self, game):
        print('your turn({})'.format('o x'[game.turn+1]))
        game.print()
        x, y = -1, -1
        while (x, y) not in game.candidates():
            print('xy> ', end='')
            xy = input()
            x, y = map(int, xy)
            x, y = x-1, y-1
        return (x, y)
