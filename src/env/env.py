import numpy as np
from dataclasses import dataclass
from board import StartPos, Board, START_POS, Point
from copy import deepcopy

STATE_SIZE = 30
DICE_IDX   = 26

@dataclass
class SARS:
    state:  np.array
    action: np.array
    reward:float
    state_next: np.array
    color_switched:bool
    terminate:bool = False

@dataclass
class Actions:
    actions: list
    white:   bool

@dataclass
class Terminate:
    terminate: bool
    won_side:bool = None

class Env:
    def __init__(self):
        pass

    def step(self, action) -> SARS:
        raise Exception("Step is not implemented!")

    # def get_memory(self):
    #     raise Exception("get_memory is not implemented!")

    def evaluate(self):
        raise Exception("Evaluate is not implemented!")
    

class TestEnv(Env):
    def __init__(
            self, 
            game_type = StartPos.SHORT,
            gamma     = 0.99
    ):
        self.board = Board(game_type)
        self.dice = np.zeros(2, np.int8)
        self.update_dice()
        self.move_white = True
        self.curr_pl_done_moves = 0

    def update_dice(self):
        self.dice  = self.throw_dice()

    def throw_dice(self):
        dice  = np.array([self.board.throw_dice(), self.board.throw_dice()], np.int8) 
        if (dice[0] == dice[1]):
            dice = np.concatenate([dice, dice])

        return dice

    def get_moves(self) -> Actions:
        return Actions(
            self.board.get_available_moves(
                self.move_white, self.get_curr_dice()),
                self.move_white
        )
    
    def get_moves_from_state(self, state:np.array, white:bool) -> list:
        assert(state.shape[0] == STATE_SIZE)

        board = self.state2board(state)
        # side = not white if state[DICE_IDX+1] == 0 else white 
        side = white
        return Actions(board.get_available_moves(white, state[DICE_IDX]),  side)
    
    def get_curr_dice(self):
        return self.dice[self.curr_pl_done_moves]

    def step(self, action:int) -> SARS:
        moves = self.get_moves()
        state = self.get_state()
        if (len(moves.actions) == 0):
            action = -1
        assert(action < len(moves.actions))
        
        if action >=0:
            self.board.make_move(self.move_white, moves.actions[action], self.get_curr_dice())
        new_state = self.get_state()
        reward    = self.evaluate(new_state)
        switched  = False
        if self.curr_pl_done_moves +1  >= self.dice.shape[0]:
            self.curr_pl_done_moves = 0
            self.update_dice()
            self.move_white = not self.move_white
            switched = True
        else:
            self.curr_pl_done_moves += 1

        return SARS(state, action, reward, new_state, switched, self.is_terminate().terminate)
    
    def is_terminate(self):
        return self.is_terminate_board(self.board)
    
    @staticmethod
    def is_terminate_board(board):
        return Terminate(board.game_ended(), board.side_won(True))

    def get_state(self):
        if self.move_white:
            self.board.mirror()
        
        state = self.board2state(self.board, self.dice[self.curr_pl_done_moves:])

        if self.move_white:
            self.board.mirror()

        return state
    
    @staticmethod
    def board2state(board, dice:np.array):
        state = np.zeros(STATE_SIZE, np.int8) # 24 tiles + bar + dice
        for i in range(24):
            point = board.points[i]
            state[23-i] = -point.count if point.color else point.count

        state[24] = board.bar[0]
        state[25] = board.bar[1]

        state[DICE_IDX:DICE_IDX + dice.shape[0]] = dice

        return state


    @staticmethod
    def reverse_state(state):
        state_ = deepcopy(state)
        state_[:24] = np.flip(-state_[:24], 0)
        return state_

    @staticmethod
    def evaluate(state:np.array, white = True):
        assert(state.shape[0] == STATE_SIZE)

        state_ = state

        if not white:
            state_ = TestEnv.reverse_state(state)

        eval = 0.0
        for i in range(23):
            eval += state_[i] * i * 0.2 if state_[i] > 0 else 0.0
        eval += state_[0] * 10.0 if state_[0] > 0 else 0 # reward for getting own checker to the top

        eval -= state_[25] * 1.0 # negative reward for having own checkers on bar
        eval += state_[24] * 4.0 # positive reward for having enemy checkers on bar

        return eval
    
    def state2board(self, state) -> Board:
        points = [Point(n > 0, abs(n)) for n in state[:24]]
        bar    = state[24:26]

        res = Board()
        res.points = points
        res.bar = bar

        return res

    def peak(self, state, action, side:bool) -> SARS:
        board = self.state2board(state)
        assert(state[DICE_IDX] > 0)

        board.make_move(side, action, state[DICE_IDX])

        if state[DICE_IDX + 1] == 0:
            dice = self.throw_dice()
        else:
            dice = state[DICE_IDX+1:DICE_IDX + 3]

        new_state = self.board2state(board, dice)
        eval      = self.evaluate(new_state)
        term      = self.is_terminate_board(board)

        return SARS(state, action, eval, new_state, state[DICE_IDX + 1] == 0, term.terminate)
    
    def fwd_look(self, state, white:bool, start_white:bool, depth=1):
        res = [[[]]]

        actions = self.get_moves_from_state(state, start_white).actions
        sars = None
        for act in actions:
            sars = self.peak(state, act, start_white)
            res[0][0].append(self.evaluate(sars.state_next, white))
            if depth > 0 and sars is not None:
                side = not start_white if sars.color_switched else start_white
                new_res = self.fwd_look(sars.state_next, white, side, depth - 1)
                for i, n in enumerate(new_res):
                    if len(res) <= i + 1:
                        res.append([])
                    res[i+1] += new_res[i]
        
        assert(len(res) == depth + 1)

        return res
    
    
    




        






