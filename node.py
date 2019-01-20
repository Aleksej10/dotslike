from board import *
from copy import deepcopy
from numpy import pi, e, sqrt, log, argmax, argmin, inf, float64 as f64

from keras.models import load_model, Sequential
from keras.layers import Dense



class Node():
    try:
        model = load_model('model')             #tries to load static model of size [100, 50, 20, 1]
        print('model loaded from file')
    except OSError:
        model = Sequential()
        model.add(Dense(50, input_dim=100, activation='tanh'))
        model.add(Dense(20, activation='tanh'))
        model.add(Dense(1, activation='tanh'))  #output layer represents probability for winning, from -1 to 1
        model.compile(loss='mse', optimizer='sgd')
        model.save('model')

    #training data
    data_in = []
    exp_data = []

    def __init__(self, b, father):
        self.b = b              #board
        self.father = father    #father node (for backpropagation)
        self.sons = []          #list of sons (basicaly paths to all possible position from current one)
        self.visits = 0         #number of visits, used when calculating usb1
        self.score = 0.0        #summed probabilities of winning
        self.true_eval = 0      #true eval for leaf nodes or nodes fairly close to them
        self.done = 0           #node can be done without it's board being done, i just means it's true value is determined, no need to look at it any longer
        self.estimate = 0.0     #neural networks estimation, used for updating the weights later
        self.inn = array(self.b.getNumpyArray()).reshape(1,100)     #input to a neural network

    def expand(self):
        moves = self.b.getMoves()       #connects node to all possible positions (nodes) available from current state
        for move in moves:
            tmp = deepcopy(self.b)
            tmp.playMove(move)
            self.sons.append(Node(tmp, self))

    def show(self):                     #visual representation of a node
        print('visits: ', end='')
        print(self.visits, end=' ')
        print('chances: ', end='')
        print(f64(self.score)/self.visits, end=' ')
        print('done: ', end='')
        print(self.done, end=' ')
        print('estimate: ', end='')
        print(self.estimate)

    def showSons(self):                 #family display
        for son in self.sons:
            son.show()

    def usb1(self):                     #how you pick which path to pay a visit
        if self.visits == 0:            #state of art balance between exploration and exploitation
            return inf * self.father.b.side
        return (f64(self.score)/self.visits) + self.father.b.side*e*sqrt(2 * f64(log(self.father.visits))/self.visits)

    def backprop(self, score, done): #completely intuitive process of backpropagation
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
            if self.b.done():                           #game is finished
                ev = 1 if (self.b.scr > 0) else -1      #get eval and backpropagate
                self.estimate = self.model.predict(self.inn)[0,0]  #also save neural network's estimation
                self.backprop(ev, 1)                               #          for fitting the model later
                return
            if self.visits == 0:                        #get estimate and backpropagete, if the estimation is good
                self.estimate = self.model.predict(self.inn)[0,0]        #the node will be visited again soon,
                self.backprop(self.estimate, 0)                          #and then self.visits won't be zero, so
                return                                                   #the node is getting expanded
            else:
                self.expand()
                return
        else:
            usbs = []       #here we choose where to go from current node
            k = 0
            for son in self.sons:
                if son.done:
                    usbs.append(-inf*self.b.side)   #we don't want to explore nodes we already know true eval
                    k+=1
                else:
                    usbs.append(son.usb1())
            if k>0:
                truz = [son.true_eval for son in self.sons]    #true_eval can be either 1 or -1
                if self.b.side == 1:
                    if 1 in truz:               #if '1' is to play, and he can win a game with that move,
                        self.backprop(1,1)      #it's node is being declared done and it's true value is being updated accordingly
                        return
                    else:
                        if len(usbs) == k:      #if every node on 1's turn is done, and he is losing no matter what he plays,
                            self.backprop(-1,1) #it's node is being declared done and it's true value is being updated accordingly
                            return
                else:
                    if -1 in truz:              #same stuff, only if '-1' is to play
                        self.backprop(-1,1)
                        return
                    else:
                        if len(usbs) == k:
                            self.backprop(1,1)
                            return
            #finaly, if node is not done, we chose where to go
            if self.b.side == 1:
                self.sons[argmax(usbs)].monte()
            else:
                self.sons[argmin(usbs)].monte()

    def mcts(self, n):                  #just runs monte n times
        while (not self.done) and (n>0):
            self.monte()
            n-=1

    def chose(self):                #basicaly min/max
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


    def prob01(self):       #return probability of winning based on mcts
        return f64(self.score)/self.visits

    def player_chose(self, n):   #returns index of son player wants to go to based on index of line he intends to play
        for i in range(len(self.sons)):
            if (self.b.linije | n) == self.sons[i].b.linije:
                return i

    def endlessMonte(self):   #when playing against human, for pondering
        try:
            while True:
                self.monte()
        except KeyboardInterrupt:
            pass

    def fitModel(self, plus_one, plus_two):
        y = 0                                #weights are updated in regard to final game result
        if self.done:                        #and average of estimation two nodes after them in an
            y = self.true_eval               #attempt to smoothen the graph of estimated values
        else:                                #during the game
            y = f64(plus_one + plus_two)/2                    #weights are also updated based on true eval when possible

        self.data_in.append(self.b.getNumpyArray())
        self.exp_data.append([y])
        if self.father == None:
            self.model.fit(array(self.data_in), array(self.exp_data), epochs=1, verbose=0)
            self.exp_data = []
            self.data_in = []
            print('weights updated')
            return
        for brother in self.father.sons:        #also, weights are trained to minimize the difference between
            if brother is not self:             #estimated probabilities and probabilities calculated by mtcs
                self.data_in.append(brother.b.getNumpyArray())
                self.exp_data.append([brother.prob01()])
        self.father.fitModel(self.estimate, y)


