import queue

from numpy.random import randint

from Point import Point
from Action import Action

class Snake:

    def __init__(self, gridSize, identity):
        self.head = Point(randint(10, gridSize - 10), randint(10, gridSize - 10)) # generate point with at least 10 units gap from any wall
        self.end = Point( self.head.x - 5, self.head.y )
        self.joints = queue.Queue()
        self.id = identity
        self.alive = True