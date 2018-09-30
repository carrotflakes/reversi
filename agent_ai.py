import tensorflow as tf
import numpy as np
from game import Game

class Node:

    def __init__(self, parent=None, policy=None):
        self.parent = parent
        self.children = None
        self.visit_count = 0
        self.score = 0
        self.policy = policy # -1 ~ 1
        self.value = None # -1 ~ 1

    def select(self, turn):
        C = 1.0
        best_node = None
        best_value = -1
        best_pos = None
        for pos, child in self.children.items():
            u = C * ((turn * child.policy + 1) / 2) * (self.visit_count ** 0.5 / (1 + child.visit_count))
            value = (child.score if turn == 1 else child.visit_count - child.score) / (child.visit_count + 0.001) + u
            if best_value < value:
                best_node = child
                best_value = value
                best_pos = pos
        return best_pos, best_node

    @property
    def expanded(self):
        return self.children is not None

    def print(self, indent=0):
        print('  ' * indent + '{:6f}/{:6f}, p:{}, v:{}'.format(self.score, self.visit_count, self.policy, self.value))
        if self.expanded:
            for child in self.children.values():
                child.print(indent=indent + 1)

class AgentAI:

    def __init__(self, sess, temperature=0):
        self.sess = sess
        self.expand_threshold = 3
        self.attempt = 100
        self.temperature = temperature

        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.board = tf.placeholder(tf.float32, [None, 8, 8, 2])
        self.true_policy = tf.placeholder(tf.float32, [None, 8, 8])
        self.true_value = tf.placeholder(tf.float32, [None])

        batch_size = tf.shape(self.board)[0]
        board_ = tf.concat([self.board, tf.ones([batch_size, 8, 8, 1])], axis=3)

        def layer(name, h):
            h = conv(name, 64, 3, 1, 'same', h)
            h = tf.layers.batch_normalization(h, training=self.is_training)
            return h

        c_h = layer('conv1', board_)
        c_h = layer('conv2', c_h)
        c_h = layer('conv3', c_h)
        c_h = layer('conv4', c_h)
        c_h = layer('conv5', c_h)
        c_h = layer('conv6', c_h)
        c_h = layer('conv7', c_h)

        with tf.variable_scope('policy'):
            h = conv('conv1', 1, 3, 1, 'same', c_h)
            h = tf.reshape(h, (-1, 8 * 8))

            self.policy_ = tf.reshape(tf.nn.softmax(h), (-1, 8, 8))

            self.policy_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.reshape(self.true_policy, (-1, 8 * 8)),
                logits=h)
            policy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            self.policy_train_op = policy_optimizer.minimize(self.policy_loss)

        with tf.variable_scope('value'):
            h = conv('conv1', 1, 3, 1, 'same', c_h)
            h = tf.reshape(h, (-1, 8 * 8))
            h = tf.layers.dense(h, 128, activation=tf.nn.relu)
            h = tf.layers.dense(h, 1)

            self.value_ = tf.nn.tanh(tf.reshape(h, (-1,)))

            self.value_loss = tf.losses.mean_squared_error(
                labels=self.true_value,
                predictions=self.value_)
            value_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            self.value_train_op = value_optimizer.minimize(self.value_loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([self.policy_train_op, self.value_train_op, update_ops])

    def policy(self, games):
        boards = list(map(game_to_board, games))
        policies = self.sess.run(self.policy_, {self.board: boards})
        #print(games[0].board)
        #print(policies[0])
        return [
            {
                (x, y): policy[y, x]
                for x, y in game.candidates()
            }
            for game, policy in zip(games, policies)
        ]

    def value(self, games):
        boards = list(map(game_to_board, games))
        values = self.sess.run(self.value_, {self.board: boards})
        return values

    def think(self, game):
        root = Node()
        for _ in range(self.attempt):
            g = game.copy()
            node = root
            while node.expanded:
                # select
                pos, node = node.select(g.turn)
                #print('selected: {}'.format(pos))
                g.step(*pos)
                if g.end:
                    node.value = g.judge()
                    break
            if node.visit_count >= self.expand_threshold:
                # expand
                #print('expand')
                #g.print()
                node.children = {
                    pos: Node(node, score)
                    for pos, score in self.policy([g])[0].items()
                }
                #print([(pos, node.policy) for pos, node in node.children.items()])
            # evaluate
            if node.value is None:
                node.value = self.value([g])[0]
                #g.print()
                #print('value: {}'.format(node.value))
            # backup
            value = (node.value + 1) / 2 + (np.random.rand() - 0.5) * self.temperature
            while node is not None:
                node.visit_count += 1
                node.score += value # ?
                node = node.parent

        pos, _ = max(root.children.items(), key=lambda x: x[1].visit_count)
        #root.print()
        #exit()
        return pos

    def learn(self, poses_list):
        boards = []
        policies = []
        values = []
        for poses in poses_list:
            for poses in poses_augment(poses):
                game = Game()
                for pos in poses:
                    boards.append(game_to_board(game))
                    policies.append(np.array([[pos == (x, y) for x in range(8)] for y in range(8)],
                                             dtype=np.float32))
                    game.step(*pos)
                values.extend([game.judge()] * len(poses))

        _, policy_loss, value_loss = self.sess.run([
            self.train_op,
            self.policy_loss,
            self.value_loss
        ], {
            self.board: boards,
            self.true_policy: policies,
            self.true_value: values,
            self.is_training: True
        })
        print('policy loss: {} value loss: {}'.format(policy_loss, value_loss))

def conv(name, channel, kernel_size, stride, padding, x):
    with tf.variable_scope(name) as scope:
        return tf.layers.conv2d(
            inputs=x,
            filters=channel,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=1,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(0.0001))

def game_to_board(game):
    return np.array(
        [
            [
                [
                    game.board[y,x] == i
                    for i in [1, -1]
                ]
                for x in range(8)
            ]
            for y in range(8)
        ],
        dtype=np.float32)

def poses_augment(poses):
    return [
        poses,
        [(y, x) for (x, y) in poses],
        [(7 - x, 7 - y) for (x, y) in poses],
        [(7 - y, 7 - x) for (x, y) in poses]
    ]
