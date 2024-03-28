import numpy as np
from env.env import DICE_IDX
from env.env import Env

class AgentHuman:
    def make_move(self, env:Env) -> int:
        state   = env.get_state()
        actions = env.get_moves().actions
        while True:
            print(f"Available moves with dice {state[DICE_IDX]}:")
            for n in actions:
                print(f"{n}\t", end = "")
            print("\nSelect the move\n")
            move = self.read_input()
            if move not in actions:
                print(f"Incorrect action {move}, try again")
                continue
            break
        
        for i, n in enumerate(actions):
            if n == move:
                return i
        assert(False)
        
    @staticmethod
    def read_input():
        a = input()
        try:
            a = int(a)
        except:
            print(f"Could not read {a}, try again")
            return None
        
        return a