from numpy.random import randint

from Point import Point
from Constants import *

class Food:
    def __init__(self):
        self.foodList = []
        createFood(maximumFood)

    @staticmethod
    def addFoodToList(pointList):
        for i in pointlist:
            self.foodList.append(i)

    @staticmethod
    def createFood(n):
        for i in range(n):
            x = randint(1, gridSize-1)
            y = randint(1, gridSize-1)
            self.foodList.append(Point(x, y))

    @staticmethod
    def eatFood(food):
        self.foodList.remove(food)
        if(len(self.foodList) < maximumFood):
            createFood(maximumFood - len(self.foodList))
