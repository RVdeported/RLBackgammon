import numpy as np
from enum import Enum
from copy import deepcopy
from dataclasses import dataclass


class StartPos(Enum):
    LONGGAMMON = 0
    SHORT      = 1




class Point:
    def __init__(self, color: bool = False, count:int = 0):
        assert(count >= 0)
        self.count = int(count)
        self.color = color == True

    def is_empty(self) -> bool:
        return self.count == 0
    
    def can_place(self, color:bool) -> bool:
        return self.count <= 1 or self.color == color
    
    def place_checker(self, color:bool) -> bool: # or none..
        assert(self.can_place(color))
        if not self.is_empty() and color != self.color:
            assert(self.count == 1)        
            self.color = color
            return color
        else:
            assert(color == self.color or self.count == 0)
            self.color = color
            self.count += 1
            return None
        
    def remove_checker(self):
        assert(self.count > 0)
        self.count -= 1
        return
    

def START_POS(pos:StartPos):
    if (pos == StartPos.LONGGAMMON):
        return [
            Point(False, 15),Point(),Point(),Point(),Point(),Point(),
            Point(),Point(),Point(),Point(),Point(),Point(),
            Point(),Point(),Point(),Point(),Point(),Point(),
            Point(),Point(),Point(),Point(),Point(),Point(True, 15),
        ]
    if (pos == StartPos.SHORT):
        return [
            Point(False, 2),Point(),Point(),Point(),Point(),Point(True, 5),
            Point(),Point(True, 3),Point(),Point(),Point(),Point(False, 5),
            Point(True, 5),Point(),Point(),Point(),Point(False, 3),Point(),
            Point(False, 5),Point(),Point(),Point(),Point(),Point(True, 2),
        ]
    else: raise ValueError()

@dataclass
class BoardState:
    points:list
    bar   :np.array

class Board:
    def __init__(self, pos = StartPos.LONGGAMMON, dice_max = 6):
        
        self.pos = pos
        self.reset()
        self.dice_max = dice_max
        self.bar = np.zeros(2, dtype=np.int8)
        pass

    def fix_state(self):
        return BoardState(deepcopy(self.points, self.bar))
    
    def restore_state(self, state:BoardState):
        assert(len(state.points) == len(self.points))
        assert(state.bar.shape[0] == 2)

        self.points = state.points
        self.bar    = state.bar

    def can_move(self, color:bool, start:int, end:int) -> bool:
        if (end > 23 or end < 0):
            return False
        
        if (not self.points[end].can_place(color)):
            return False
        
        if (start == -1):
            return self.bar[int(color)] > 0
        
        return self.points[start].count > 0 and self.points[start].color == color
    
    def make_move(self, color:bool, start:int, dice:int):
        src  = self.points[start]
        dest = self.points[start - dice if color else start + dice]

        # assert(src.count > 0)
        # assert(src.color == color)

        # check we are dealing with bar checker
        if start == -1:
            assert(self.bar[int(color)] > 0)
            self.bar[int(color)] -= 1
            dest = self.points[24 - dice if color else dice - 1]
        else:
            src.remove_checker()
            
        to_bar = dest.place_checker(color)
        if to_bar is not None:
            self.bar[int(not to_bar)] += 1

    @staticmethod
    def throw_dice(dice_max = 6) -> int:
        return int(np.random.random() * dice_max) + 1
    
    def print_board(self):
        print_point = lambda x: print(f"{"+" if x.color else "-"}{x.count}\t", end="") 
        for i in range(6):
            point = self.points[12+i]
            print_point(point)
        print("||\t", end="")

        for i in range(6):
            point = self.points[18+i]
            print_point(point)
        print()

        for i in range(6):
            point = self.points[11-i]
            print_point(point)
        print("||\t", end="")
        for i in range(6):
            point = self.points[5-i]
            print_point(point)
        print()
        print(f"+{self.bar[1]} | -{self.bar[0]}")

    def get_available_checkers(self, color:bool):
        res = [i for i in range(24) if self.points[i].count > 0 and self.points[i].color == color]
        if (self.bar[int(color)] > 0):
            res.append(-1)
        return res
    
    def get_available_moves(self, color, dice):
        assert(dice <= self.dice_max and dice > 0)
        checks = self.get_available_checkers(color)
        
        res = []
        dice = -dice if color else dice
        if len(checks) == 0:
            return []
        if checks[-1] == -1:
            dest = dice - 1 if dice > 0 else 24 + dice
            return [-1] if self.can_move(color, -1, dest) else []

        for n in checks:
            if (self.can_move(color, n, n + dice)):
                res.append(n)
        
        return res
    
    def side_won(self, color):
        idx  = 0 if color else 23
        return self.points[idx].count == 15 and self.points[idx].color == color

    def game_ended(self):
        return self.side_won(True) or self.side_won(False)
    
    def reset(self):
        self.points = START_POS(self.pos)

    def mirror(self):
        for i, point in enumerate(self.points):
            self.points[i].color = not self.points[i].color

        self.points.reverse()
        buff = self.bar[0]
        self.bar[0] = self.bar[1]
        self.bar[1] = buff

            


