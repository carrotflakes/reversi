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
            value = child.score / (child.visit_count + 0.001) + u
            if best_value < value:
                best_node = child
                best_value = value
                best_pos = pos
        return best_pos, best_node

    @property
    def expanded(self):
        return self.children is not None


class AgentAI:

    def __init__(self, sess):
        self.sess = sess
        self.expand_threshold = 3
        self.attempt = 200

        self.board = tf.placeholder(tf.float32, [None, 8, 8, 2])
        self.true_policy = tf.placeholder(tf.float32, [None, 8, 8])
        self.true_value = tf.placeholder(tf.float32, [None])

        batch_size = tf.shape(self.board)[0]
        board_ = tf.concat([self.board, tf.ones([batch_size, 8, 8, 1])], axis=3)

        with tf.variable_scope('policy'):
            h = conv('conv1', 128, 3, 1, 'same', board_)
            h = conv('conv2', 128, 3, 1, 'same', h)
            h = conv('conv3', 128, 3, 1, 'same', h)
            h = conv('conv4', 128, 3, 1, 'same', h)
            h = conv('conv5', 128, 3, 1, 'same', h)
            h = conv('conv6', 128, 3, 1, 'same', h)
            h = conv('conv7', 128, 3, 1, 'same', h)
            h = conv('conv8', 1, 3, 1, 'same', h)

            self.policy_ = tf.reshape(h, (-1, 8, 8))

            self.policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.true_policy, (-1, 8 * 8)),
                logits=tf.reshape(h, (-1, 8 * 8)))
            policy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            self.policy_train_op = policy_optimizer.minimize(self.policy_loss)

        with tf.variable_scope('value'):
            h = conv('conv1', 128, 3, 1, 'same', board_)
            h = conv('conv2', 128, 3, 1, 'same', h)
            h = conv('conv3', 128, 3, 1, 'same', h)
            h = conv('conv4', 128, 3, 1, 'same', h)
            h = conv('conv5', 128, 3, 1, 'same', h)
            h = conv('conv6', 128, 3, 1, 'same', h)
            h = conv('conv7', 128, 3, 1, 'same', h)
            h = conv('conv8', 1, 3, 1, 'same', h)
            h = tf.reshape(h, (-1, 8 * 8))
            h = tf.layers.dense(h, 128, activation=tf.nn.relu)
            h = tf.layers.dense(h, 1)

            self.value_ = tf.nn.tanh(tf.reshape(h, (-1,)))

            self.value_loss = (self.value_ - self.true_value) ** 2
            value_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            self.value_train_op = value_optimizer.minimize(self.value_loss)

    def policy(self, game):
        board = np.array(
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
        [policy] = self.sess.run(self.policy_, {self.board: [board]})
        return {
            (x, y): policy[y, x]
            for x, y in game.candidates()
        }

    def value(self, game):
        board = np.array(
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
        [value] = self.sess.run(self.value_, {self.board: [board]})
        return value

    def think(self, game):
        root = Node()
        for _ in range(self.attempt):
            g = game.copy()
            node = root
            while node.expanded:
                # select
                pos, node = node.select(g.turn)
                g.step(*pos)
                if g.end:
                    node.value = g.judge()
                    break
            if node.visit_count > self.expand_threshold:
                # expand
                node.children = {
                    pos: Node(node, score)
                    for pos, score in self.policy(g).items()
                }
            # evaluate
            if node.value is None:
                node.value = self.value(g)
            # backup
            value = (node.value + 1) / 2
            while node is not None:
                node.visit_count += 1
                node.score += value # ?
                node = node.parent

        pos, _ = max(root.children.items(), key=lambda x: x[1].visit_count)
        return pos

    def learn(self):
        pass

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
