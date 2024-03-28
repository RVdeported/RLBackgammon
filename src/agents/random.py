import numpy as np
from src.env.env import Env

class AgentRandom:
    def make_move(self, env:Env) -> int:
        actions = env.get_moves().actions
        move = int(np.random.rand() * len(np.unique(actions)))
        return move
        