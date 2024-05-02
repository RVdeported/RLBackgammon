import sys
import os
from train import get_valid_algo
from src.env.env import *
from src.agents.policy import AgentPolicy
from src.agents.random import AgentRandom
import numpy as np
import pandas as pd
from train import train

ROUNDS = 100

def load_agent(name):
    agent = AgentPolicy()
    algo = get_valid_algo(name.split('.')[0], None, False)
    algo = algo.load(f"./models/{name}")

    agent.model = algo

    return agent

def sim(agents):
    env = TestEnv(eval_f = EvalPlace(),game_type=StartPos.LONGGAMMON)
    white_won = 0
    tests = ROUNDS
    for _ in range(tests):
        env.reset()
        while not env.is_terminate().terminate:
            agent   = agents[env.move_white]
            act_idx = agent.make_move(env)
            env.step(act_idx)
        white_won += int(env.is_terminate().won_side)
    return white_won / tests

if __name__ == "__main__":
    need_train = False
    config = "./configs/to_train.yaml"
    if len(sys.argv) > 1:
        need_train = sys.argv[1]
    if (len(sys.argv) > 2):
        config = sys.argv[2]

    if need_train:
        train(config)
    

    models = os.listdir("./models")
    agents = []
    for n in models:
        agents.append(load_agent(n)) 

    agents.append(AgentRandom())
    agents_num = len(agents)
    scores = np.zeros((agents_num, agents_num))
    for i in range(agents_num):
        for j in range(i + 1, agents_num):
            scores[i, j] = sim([agents[i], agents[j]])
            print(i, j)

    res = pd.DataFrame(scores, columns = [''.join(n.split('.')[1:]) for n in models] + ["random"])
    res.index = res.columns
    res.to_csv('results.csv')




           



