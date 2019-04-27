from numpy.random import randint

from Point import Point
from Constants import *

foodList = []

def createFood(n):
    for i in range(n):
        x = randint(1, gridSize-1)
        y = randint(1, gridSize-1)
        foodList.append(Point(x, y))

def addFoodToList(pointList):
    for i in pointList:
        foodList.append(i)

def eatFood(food):
    for i in foodList:
        if Point.compare(i, food):
            foodList.remove(i)
    if(len(foodList) < maximumFood):
        createFood(maximumFood - len(foodList))
