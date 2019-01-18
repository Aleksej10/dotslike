from board import *
from node import *
from snode import *

from datetime import datetime
from github import Github

token = input('enter token key: ')

def commitModel():
    g = Github(token)
    repo = g.get_repo('Aleksej10/dotslike')
    contents = repo.get_file_contents('model')
    ml = open('model', 'rb')
    f = ml.read()
    ml.close()
    tm = 'weights updated: ' + str(datetime.now())
    repo.update_file('model', tm, f, contents.sha)
    print('model uploaded')

sdie = 1

for i in range(1000):
    root  = Node(Board(0,0,1),None)
    root.mcts(333)

    print(i)
    print(sdie)
    while not root.b.done():
        if root.b.side == sdie:
            robot = sNode(root.b)
            d = robot.goodDepth()
            robot.build_tree(d-1)
            i = robot.best_move(d-1)[0]
            root.mcts(500)
            #if len(root.sons) == 0:
            #    root.expand()
            root = root.sons[i]
            print(root.estimate, end=' ')
            print(root.b.getNumpyArray()[:4])
        else:
            root.mcts(500)
            root = root.sons[root.chose()]
            print(root.estimate, end=' ')
            print(root.b.getNumpyArray()[:4])
    sdie *= -1

    root.mcts(10)
    game_eval = root.true_eval
    root.fitModel(game_eval, game_eval)
    Node.model.save('model')
    commitModel()






