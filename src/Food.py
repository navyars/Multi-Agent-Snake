from numpy.random import randint

from Point import Point
from Constants import *

class Food:
    def __init__(self):
        self.foodList = []
        self.createFood(maximumFood)

    def returnFoodList(self):
        return self.foodList    

    def addFoodToList(self, pointList):
        for i in pointlist:
            self.foodList.append(i)

    def createFood(self, n):
        for i in range(n):
            x = randint(1, gridSize-1)
            y = randint(1, gridSize-1)
            self.foodList.append(Point(x, y))

    def eatFood(self, food):
        self.foodList.remove(food)
        if(len(self.foodList) < maximumFood):
            self.createFood(maximumFood - len(self.foodList))
