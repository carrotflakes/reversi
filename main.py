from game import Game
from agent_ai import AgentAI
from agent_cui import AgentCUI
from agent_ab import AgentAB
from agent_random import AgentRandom
import tensorflow as tf
import time


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('infer', None, 'infer mode')
flags.DEFINE_integer('resume', None, 'resume training')


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

def eval(agent, epoch=50):
    agent2 = AgentRandom()
    count1 = 0
    count2 = 0
    for i in range(epoch):
        game = Game()
        if i % 2 == 0:
            playout(game, agent, agent2)
            count1 += game.judge() == 1
            count2 += game.judge() == -1
        else:
            playout(game, agent2, agent)
            count1 += game.judge() == -1
            count2 += game.judge() == 1
    print('win: {}, lose: {}, draw: {}'.format(count1, count2, epoch - count1 - count2))


if __name__ == '__main__':
    #'''
    sess = tf.Session()
    agent = AgentAI(sess, temperature=0.2)
    saver = tf.train.Saver()

    if FLAGS.resume is not None:
        saver.restore(sess, 'model-' + str(FLAGS.resume))
        global_step = FLAGS.resume + 1
    else:
        sess.run(tf.global_variables_initializer())
        global_step = 0

    for epoch in range(500):
        print(global_step)

        for _ in range(5):
            pl = []
            start_time = time.time()
            for _ in range(10):
                game = Game()
                poses = playout(game, agent, agent)
                pl.append(poses)
            print(time.time() - start_time)
            agent.learn(pl)

        saver.save(sess, './model', global_step=global_step)
        global_step += 1

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
