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

    def search(self, game, n):
        leaves = [] # [(node, game, n)]
        def traverse(node, game, n):
            if not node.expanded:
                leaves.append((node, game, n))
                return

            C = 1.0
            turn = game.turn
            values = [
                (child.score if turn == 1 else child.visit_count - child.score) / (child.visit_count + 0.001) \
                + C * ((turn * child.policy + 1) / 2) * (node.visit_count ** 0.5 / (1 + child.visit_count))
                for child in node.children.values()
            ]
            vsum = sum(values)
            nums = [round(v * n / vsum) for v in values] # Note: sum(nums) is indefined.
            for (pos, child), num in zip(node.children.items(), nums):
                if num > 0:
                    g = game.copy()
                    g.step(*pos)
                    traverse(child, g, num)
        traverse(self, game, n)
        return leaves

    @property
    def expanded(self):
        return self.children is not None

    def print(self, path=''):
        print(path + ' {:6f}/{:6f}, p:{}, v:{}'.format(self.score, self.visit_count, self.policy, self.value))
        if self.expanded:
            for pos, child in self.children.items():
                child.print(path + '{}'.format(pos))

class AgentAI:

    def __init__(self, sess, temperature=0):
        self.sess = sess
        self.expand_threshold = 3
        self.steps = 10
        self.attempt = 100
        self.batch_size = 64
        self.temperature = temperature

        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.board = tf.placeholder(tf.float32, [None, 8, 8, 2])
        self.true_policy = tf.placeholder(tf.float32, [None, 8, 8])
        self.true_value = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])

        batch_size = tf.shape(self.board)[0]
        board_ = tf.concat([self.board, tf.ones([batch_size, 8, 8, 1])], axis=3)

        def layer(name, inp):
            h = conv(name + '_0', 64, 3, 1, 'same', inp)
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.nn.relu(h)
            h = conv(name + '_1', 64, 3, 1, 'same', h)
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = h + inp
            h = tf.nn.relu(h)
            return h

        c_h = conv('conv0', 64, 3, 1, 'same', board_)
        c_h = tf.layers.batch_normalization(c_h, training=self.is_training)
        c_h = tf.nn.relu(c_h)
        c_h = layer('conv1', c_h)
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

        with tf.variable_scope('value'):
            h = conv('conv1', 1, 1, 1, 'same', c_h)
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.nn.relu(h)
            h = tf.reshape(h, (-1, 8 * 8))
            h = tf.layers.dense(h, 64, activation=tf.nn.relu)
            h = tf.layers.dense(h, 1)

            self.value_ = tf.nn.tanh(tf.reshape(h, (-1,)))

            self.value_loss = tf.losses.mean_squared_error(
                labels=self.true_value,
                predictions=self.value_)

        loss = self.policy_loss + self.value_loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        minimize_op = optimizer.minimize(loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([minimize_op, update_ops])

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

    def infer(self, games):
        boards = list(map(game_to_board, games))
        policies, values = self.sess.run([self.policy_, self.value_], {self.board: boards})
        #policies, values = [np.array([[1/64 for _ in range(8)] for _ in range(8)])] * len(games), [0.] * len(games)
        return [
            (
                (lambda game, policy: lambda: [
                    ((x, y), policy[y, x])
                    for x, y in game.candidates()
                ])(game, policy),
                value
            )
            for game, policy, value in zip(games, policies, values)
        ]

    def think_v2(self, game_root):
        def backup(node, num):
            value = (node.value + 1) / 2 * num
            while node is not None:
                node.visit_count += num
                node.score += value
                node = node.parent

        root = Node()
        for _ in range(self.steps):
            leaves = root.search(game_root, self.attempt)
            for node, game, num in leaves:
                if game.end:
                    node.value = game.judge()
                    backup(node, num)
            leaves = list(filter(lambda x: not x[1].end, leaves))
            np.random.shuffle(leaves)
            leaves.sort(key=lambda x: -x[2])
            leaves = leaves[:self.batch_size]
            if len(leaves) == 0:
                continue
            policy_value_pairs = self.infer([game for _, game, _ in leaves])
            for (node, game, num), (policy_fn, value) in zip(leaves, policy_value_pairs):
                node.value = value
                if node.visit_count + num >= self.expand_threshold:
                    node.children = {
                        pos: Node(node, score)
                        for pos, score in policy_fn()
                    }
                backup(node, num)
        #game_root.print()
        #if game_root.stone_count >= 62:
        #    root.print()
        #print('========')
        pos, _ = max(root.children.items(), key=lambda x: x[1].visit_count)
        return pos

    def think(self, game):
        root = Node()
        for _ in range(self.attempt):
            g = game.copy()
            node = root
            while True:
                while node.expanded:
                    # select
                    pos, node = node.select(g.turn)
                    #print('selected: {}'.format(pos))
                    g.step(*pos)
                    if g.end:
                        node.value = g.judge()
                        break
                if not g.end and node.visit_count >= self.expand_threshold:
                    # expand
                    #print('expand')
                    #g.print()
                    node.children = {
                        pos: Node(node, score)
                        for pos, score in self.policy([g])[0].items()
                    }
                    #print([(pos, node.policy) for pos, node in node.children.items()])
                else:
                    break
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
                turns = []
                for pos in poses:
                    boards.append(game_to_board(game))
                    policies.append(np.array([[pos == (x, y) for x in range(8)] for y in range(8)],
                                             dtype=np.float32))
                    turns.append(game.turn)
                    game.step(*pos)
                judge = game.judge()
                values.extend(t * judge for t in turns)

        _, policy_loss, value_loss = self.sess.run([
            self.train_op,
            self.policy_loss,
            self.value_loss
        ], {
            self.board: boards,
            self.true_policy: policies,
            self.true_value: values,
            self.is_training: True,
            self.lr: 0.01,
        })
        print('policy loss: {} value loss: {}'.format(policy_loss, value_loss))

def conv(name, filters, kernel_size, stride, padding, x):
    with tf.variable_scope(name) as scope:
        return tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01))

def game_to_board(game):
    turn = game.turn
    return np.array(
        [
            [
                [
                    game.board[y,x] == i * turn
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
        [(7 - y, 7 - x) for (x, y) in poses],
        [(x, 7 - y) for (x, y) in poses],
        [(7 - x, y) for (x, y) in poses],
        [(y, 7 - x) for (x, y) in poses],
        [(7 - y, x) for (x, y) in poses]
    ]
