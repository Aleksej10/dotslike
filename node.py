from board import *
from copy import deepcopy
from numpy import pi, e, sqrt, log, argmax, argmin, inf, float64 as f64

from keras.models import load_model, Sequential
from keras.layers import Dense



class Node():
    try:
        model = load_model('model')
        print('model loaded from file')
    except OSError:
        model = Sequential()
        model.add(Dense(50, input_dim=100, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
        model.save('model')

    def __init__(self, b, father):
        self.b = b
        self.father = father
        self.sons = []
        self.visits = 0
        self.score = 0.0
        self.true_eval = 0
        self.done = 0
        self.estimate = 0.0
        self.inn = b.getNumpyArray()

    def expand(self):
        moves = self.b.getMoves()
        for move in moves:
            tmp = deepcopy(self.b)
            tmp.playMove(move)
            self.sons.append(Node(tmp, self))

    def show(self):
        print('visits: ', end='')
        print(self.visits, end=' ')
        print('chances: ', end='')
        print(f64(self.score)/self.visits, end=' ')
        print('true_eval: ', end='')
        print(self.true_eval, end=' ')
        print('done: ', end='')
        print(self.done)

    def showSons(self):
        for son in self.sons:
            son.show()

    def usb1(self):
        return (f64(self.score)/self.visits) + e*sqrt(f64(log(self.father.visits))/self.visits)

    def backprop(self, score, done):
        self.score += score
        self.visits += 1
        self.done = done
        if done:
            self.true_eval = score
        if self.father == None:
            return
        self.father.backprop(score, 0)

    def monte(self):
        if len(self.sons) == 0: #leaf node
            if self.b.done():
                ev = 1 if (self.b.scr > 0) else -1
                self.backprop(ev, 1)
                return
            if self.visits == 0:
                est = self.model.predict(self.inn)[0,0]
                self.estimate = est
                self.backprop(est, 0)
                return
            else:
                self.expand()
                return
        else:
            usbs = []
            k = 0
            for son in self.sons:
                if son.done:
                    usbs.append(-inf*self.b.side)
                    k+=1
                else:
                    usbs.append(son.usb1())
            if k>0:
                truz = [son.true_eval for son in self.sons]
                if self.b.side == 1:
                    if 1 in truz:
                        self.backprop(1,1)
                        return
                    else:
                        if len(usbs) == k:
                            self.backprop(-1,1)
                            return
                else:
                    if -1 in truz:
                        self.backprop(-1,1)
                        return
                    else:
                        if len(usbs) == k:
                            self.backprop(1,1)
                            return
            if self.b.side == 1:
                self.sons[argmax(usbs)].monte()
            else:
                self.sons[argmin(usbs)].monte()

    def chose(self):
        chances = []
        for son in self.sons:
            if son.visits == 0:
                son.monte()
            if son.done:
                chances.append(son.true_eval)
            else:
                chances.append(f64(son.score)/son.visits)
        if self.b.side == 1:
            return argmax(chances)
        else:
            return argmin(chances)

    def player_chose(self, n):
        for i in range(len(self.sons)):
            if (self.b.linije | n) == self.sons[i].b.linije:
                return i

    def mcts(self, n):
        while (not self.done) and (n>0):
            self.monte()
            n-=1

    def endlessMonte(self):
        try:
            while True:
                self.monte()
        except KeyboardInterrupt:
            pass

    def fitModel(self, game_score, est_next):
        y = 0
        if self.done:
            y = array(self.true_eval).reshape(1,1)
        else:
            y = f64(est_next + game_score)/2
            y = array(y).reshape(1,1)
        self.model.fit(self.inn, y, epochs=1, batch_size=1, verbose=0)
        if self.father == None:
            print('weights updated')
            return
        self.father.fitModel(game_score, self.estimate)


