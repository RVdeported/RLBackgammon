# from stable_baselines.deepq.policies import MlpPolicy


class AgentPolicy:
    def __init__(self, env=None, algo=None, algoargs={}):
        if (env is not None  and 
            algo is not None
        ):
            self.model = algo("MlpPolicy", env, verbose=1, **algoargs)
        else:
            self.model = None

    def make_move(self, env) -> int:
        return self.model.predict(env.get_state())[0]
