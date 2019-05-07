from numpy.random import randint

from Point import Point
from Constants import *

class Food:

    def __init__(self, snakes=[]):
        self.foodList = []
        self.createFood(maximumFood, snakes)

    def createFood(self, n, snakes=[]):
        occupiedPoints = []
        for snake in snakes:
            body = snake.getBodyList()
            bodyPoints = Point.returnBodyPoints(body)
            occupiedPoints.extend(bodyPoints)

        for i in range(n):
            while True:
                x = randint(1, gridSize-1)
                y = randint(1, gridSize-1)
                p = Point(x,y)
                if p not in occupiedPoints and p not in self.foodList:
                    self.foodList.append(p)
                    break

    def addFoodToList(self, pointList):
        for p in pointList:
            self.foodList.append(p)

    def eatFood(self, food, snakes=[]):
        for i, f in enumerate(self.foodList):
            if f == food:
                del self.foodList[i]

        if(len(self.foodList) < maximumFood):
            self.createFood(maximumFood - len(self.foodList), snakes)
