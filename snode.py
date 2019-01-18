from board import *
from copy import deepcopy
from numpy import pi, e, sqrt, log, argmax, argmin, inf, float64 as f64

class sNode():
    def __init__(self, b):
        self.b = b
        self.sons = []
        self.level = 96 - bin(self.b.linije).count('1')

    def expand(self):
        moves = self.b.getMoves()
        for move in moves:
            tmp = deepcopy(self.b)
            tmp.playMove(move)
            self.sons.append(sNode(tmp))

    def build_tree(self, depth):
        if self.level <= depth:
            return
        else:
            if len(self.sons) == 0:
                self.expand()
            for son in self.sons:
                son.build_tree(depth)

    def best_move(self, depth):
        evs = []
        if self.b.done() or (self.level <= depth):
            return [0, self.b.scr]
        else:
            evs = [son.best_move(depth)[1] for son in self.sons]

        if self.b.side == 1:
            return [argmax(evs),max(evs)]
        else:
            return [argmin(evs),min(evs)]

    def goodDepth(self):
        s = self.level
        d = 1
        while s*(s-1) < 20000 and s>2:
            d += 1
            s -= 1
        return d
