from slackclient import SlackClient
import time
import tensorflow as tf
from agent_ai import AgentAI
from game import Game
import sys

token = sys.argv[1]

client = SlackClient(token)

sess = tf.Session()
agent = AgentAI(sess, temperature=0.0)
saver = tf.train.Saver()
saver.restore(sess, 'model-' + sys.argv[2])
game = Game()

def render_game(game):
    numbers = ':one: :two: :three: :four: :five: :six: :seven: :eight:'.split()
    parts = []
    parts.append(['コンピュータの番です', '', 'あなたの番です(34 と入力すると左から3番目、上から4番目に打ちます)'][game.turn+1])
    parts.append(':white_small_square:' + ''.join(numbers))
    for y in range(8):
        parts.append(numbers[y])
        for x in range(8):
            parts[-1] += [':white_circle:', ':white_small_square:', ':black_circle:'][game.board[y,x]+1]
    return '\n'.join(parts)

def handle_message(data):
    global game
    if data.get('type') == 'message' and data.get('user') != self_user_id:
        channel = data.get('channel', None)
        if data.get('text', '') == 'オセロ':
            game = Game()
            message = '対局を開始します！'
            message += '\n' + render_game(game)
            client.rtm_send_message(channel, message)
        else:
            try:
                x, y = list(map(int, data['text']))
                x -= 1
                y -= 1
                game.step(x, y)
                client.rtm_send_message(channel, render_game(game))
                while not game.end and game.turn == -1:
                    game.step(*agent.think(game))
                    client.rtm_send_message(channel, render_game(game))
                if game.end:
                    message = ['引き分け', 'あなたの勝ち！', 'コンピュータの勝ち！'][game.judge()]
                    message += '\n\n`オセロ` と入力すると対局が始まります。'
                    client.rtm_send_message(channel, message)
            except:
                pass

if client.rtm_connect():
    self_user_id = client.server.login_data['self']['id']
    print('ready.')

    while True:
        data = client.rtm_read()

        for item in data:
            handle_message(item)

        time.sleep(0.2)
else:
    print("Connection failed, is token invalid?")
