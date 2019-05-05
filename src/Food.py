from numpy.random import randint

from Point import Point
from Constants import *

foodList = []

def createFood(n, snakes=[]):
    occupiedPoints = []
    for snake in snakes:
        body = snake.getBodyList()
        bodyPoints = Point.returnBodyPoints(body)
        occupiedPoints.extend(bodyPoints)
    occupiedPoints = set(occupiedPoints)


    for i in range(n):
        while True:
            x = randint(1, gridSize-1)
            y = randint(1, gridSize-1)
            p = Point(x,y)
            if p not in occupiedPoints and p not in foodList:
                foodList.append(p)
                break

def addFoodToList(pointList):
    for p in pointList:
        foodList.append(p)

def eatFood(food, snakes=[]):
    foodList.remove(food)
    if(len(foodList) < maximumFood):
        createFood(maximumFood - len(foodList), snakes)
