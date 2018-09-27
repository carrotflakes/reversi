import tensorflow as tf

class Node:

    def __init__(self, parent=None, policy=None):
        self.parent = parent
        self.children = None
        self.visit_count = 0
        self.score = 0
        self.policy = policy
        self.value = None

    def select(self):
        C = 1.0
        node = None
        value = -1
        for pos, child in self.children.items():
            u = C * child.policy * (self.visit_count ** 0.5 / (1 + child.visit_count))
            v = child.q + u
            if value < v:
                node = child
                value = v
        return node

    @property
    def q(self):
        return self.score / (self.visit_count + 0.001)

    @property
    def expanded(self):
        return self.children is not None


class AgentAI:

    def __init__(self):
        self.expand_threshold = 0.75
        self.attempt = 1000

        with tf.Graph().as_default():
            self.x = 

    def policy(game):
        pass

    def value(game):
        pass

    def think(self, game):
        root = Node()
        for _ in range(self.attempt):
            g = game.copy()
            node = root
            while node.expanded:
                # select
                pos, node = node.select()
                g.step(*pos)
                if g.end:
                    node.value = g.judge()
                    break
                if node.s() > self.expand_threshold: # ?
                    # expand
                    node.children = {
                        pos: Node(node, score)
                        for pos, score in self.policy(g)
                    }
            # evaluate
            if node.q is None:
                node.q = self.value(g)
            # backup
            value = node.value
            while node is not None:
                node.visit_count += 1
                node.score += value # ?
                node = node.parent

        pos, _ = max(node.children.items(), key=lambda x: x[1].visit_count)
        return pos

    def learn(self):
        pass
