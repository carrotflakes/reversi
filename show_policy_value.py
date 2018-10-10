import tensorflow as tf
from game import Game
from agent_ai import AgentAI, game_to_board
import sys

sess = tf.Session()
agent = AgentAI(sess, temperature=0.2)
saver = tf.train.Saver()
saver.restore(sess, 'model-{}'.format(sys.argv[1]))
game = Game()
for i in range(60):
    print('turn {}'.format(game.turn))
    game.print()
    [policy], [value] = sess.run([agent.policy_, agent.value_], {agent.board: [game_to_board(game)]})

    for y in range(8):
        for x in range(8):
            print('{:5f} '.format(policy[y, x]), end='')
        print()

    print('value: {}'.format(value))

    game.step(*agent.think(game))
