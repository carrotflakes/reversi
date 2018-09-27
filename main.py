from game import Game
from agent_ai import AgentAI
from agent_cui import AgentCUI
from agent_random import AgentRandom
import tensorflow as tf


def playout(game, agent1, agent2):
    poses = []
    while not game.end:
        pos = [agent2, None, agent1][game.turn+1].think(game)
        game.step(*pos)
        poses.append(pos)
    return poses


if __name__ == '__main__':
    game = Game()
    agent1 = AgentCUI()
    agent2 = AgentRandom()
    playout(game, agent1, agent2)
