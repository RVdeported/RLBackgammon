import yaml
from stable_baselines3 import DQN, PPO, DDPG, TD3
from src.agents.policy import AgentPolicy
from src.env.env import TestEnv, EvalPlace, StartPos, ActionSpace
import sys



def get_start_pos(pos):
    if pos == "Short":
        return StartPos.SHORT
    elif pos == "LongGammon":
        return StartPos.LONGGAMMON
    else:
        raise Exception(f"Unidentified start pos: {pos}")
    
def get_valid_algo(algo:str, discr_space:bool, need_to_val=True):
    if   algo == "DQN":
        if need_to_val:
            assert(discr_space)
        return DQN
    
    elif algo == "DDPG":
        if need_to_val:
            assert(not discr_space)
        return DDPG
    
    elif algo == "PPO":
        return PPO
    
    elif algo == "TD3":
        if need_to_val:
            assert(not discr_space)
        return TD3
    else:
        raise Exception(f"Unidentified algo: {algo}")

def train(config_p):
    with open(config_p, "r") as f:
        d = yaml.safe_load(f)

    for item in d["models"]:
        name = list(item.keys())[0]
        v = item[name]
        env = TestEnv(
            eval_f      = EvalPlace(), 
            game_type   = get_start_pos(v["start_pos"]),
            action_type = ActionSpace.DISCRETE if v["discr_space"] else \
                        ActionSpace.CONTINIUS
        )

        agent = AgentPolicy(env, get_valid_algo(v["algo_name"], v["discr_space"]), v["algo_args"])
        agent.model.learn(total_timesteps = v["timesteps"])
        agent.model.config = v
        agent.model.save(f"./models/{name}.{v['algo_name']}")



if __name__ == "__main__":
    config = "./configs/to_train.yaml"
    if len(sys.argv) > 1:
        config = sys.argv[1]

    train(config)

        
