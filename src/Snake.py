import queue

from numpy.random import randint

from Point import Point
from Action import Action
from Food import *
from Constants import *

class Snake:
    def __init__(self, gridSize, identity):
        self.head = Point(randint(10, gridSize - 10), randint(10, gridSize - 10)) # generate point with at least 10 units gap from any wall
        self.end = Point( self.head.x - 5, self.head.y )
        self.joints = queue.Queue()
        self.id = identity
        self.alive = True
        self.score = 0

    def didEatFood(self):
        if(self.head in foodList):
            self.score = self.score + 1
            self.growSnake()
            eatFood(self.head)

    def didHitWall(self):
        if(self.head.x == 0 or self.head.x == gridSize or self.head.y == 0 or self.head.y == gridSize):
            return True
        else:
            return False
