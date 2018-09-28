from game import Game
from agent_ai import AgentAI
from agent_cui import AgentCUI
from agent_random import AgentRandom
import tensorflow as tf


def playout(game, agent1, agent2, show=False):
    def show_board():
        if show:
            print('=' * 8)
            game.print()
    show_board()
    poses = []
    while not game.end:
        pos = [agent2, None, agent1][game.turn+1].think(game)
        game.step(*pos)
        show_board()
        poses.append(pos)
    if show:
        print(game.judge())
    return poses


if __name__ == '__main__':
    sess = tf.Session()
    agent = AgentAI(sess)
    sess.run(tf.global_variables_initializer())
    '''
    for _ in range(100):
        for _ in range(100):
            game = Game()
            poses = playout(game, agent, agent)
        game.learn(TODO)
    '''

    count = 0
    agent2 = AgentRandom()
    for _ in range(1):
        game = Game()
        playout(game, agent, agent2, True)
        count += game.judge() == 1
    print('result: {}'.format(count))
    '''
    game = Game()
    agent1 = AgentCUI()
    agent2 = AgentRandom()
    playout(game, agent1, agent2)
    '''
