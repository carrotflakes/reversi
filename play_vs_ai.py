from game import Game
from agent_ai import AgentAI
from agent_cui import AgentCUI
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('model', None, 'model number')
flags.DEFINE_boolean('white', False, 'white')

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
    agent_ai = AgentAI(sess, temperature=0.0)
    saver = tf.train.Saver()

    if FLAGS.model is not None:
        saver.restore(sess, 'model-' + str(FLAGS.model))
    else:
        sess.run(tf.global_variables_initializer())

    agent_cui = AgentCUI()

    if FLAGS.white:
        agent2, agent1 =  agent_cui, agent_ai
    else:
        agent1, agent2 =  agent_cui, agent_ai

    game = Game()
    playout(game, agent1, agent2)
    print('judge: {}'.format(['white win', 'draw', 'black win'][game.judge() + 1]))
