from game import Game
from agent_ai import AgentAI
from agent_cui import AgentCUI
from agent_ab import AgentAB
from agent_random import AgentRandom
import tensorflow as tf
import time


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

def eval(agent):
    count = 0
    agent2 = AgentRandom()
    epoch = 50
    for _ in range(epoch):
        game = Game()
        playout(game, agent, agent2)
        count += game.judge() == 1
    print('winning rate: {}/{}'.format(count, epoch))


if __name__ == '__main__':
    #'''
    sess = tf.Session()
    agent = AgentAI(sess, temperature=0.2)
    sess.run(tf.global_variables_initializer())

    for epoch in range(500):
        pl = []
        start_time = time.time()
        for _ in range(20):
            game = Game()
            poses = playout(game, agent, agent)
            pl.append(poses)
        print(time.time() - start_time)
        agent.learn(pl)

        eval(agent)

        game = Game()
        game.step(2, 3)
        [policy] = agent.policy([game])
        print(policy)

    '''
    sess = tf.Session()
    game = Game()
    agent1 = AgentRandom()
    agent2 = AgentAI(sess)
    agent3 = AgentAB()
    sess.run(tf.global_variables_initializer())
    #eval(agent3)
    playout(game, agent1, agent3, show=True)
    #'''
