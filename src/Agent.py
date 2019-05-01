###   SINGLE SNAKE  ###
    # absolute
        # head
        # k nearest points - from head
        # direction - action enums
    # relative
        # k nearest points
        # direction - action enums

###  MULITPLE SNAKES  ###
    # absolute
        # head
        # k nearest points
        # direction - action enums
        # other snakes head
        # other snakes closest point
    # relative
        # k nearest points
        # direction - action enums
        # other snakes head
        # other snakes closest point

from math import *
import numpy as np

from Snake import Snake
from Point import Point
from Constants import *
from Food import *

def calculateDistance(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def findKNearestPoints(head, k):
    dist = []
    nearestPoints = []

    for food in foodList:
        dist.append(calculateDistance(food, head))

    if len(foodList) < k:
        k = len(foodList)

    argmin = np.argpartition(dist, k)

    for i in range(k):
        nearestPoints.append(foodList[argmin[i]])
    
    return nearestPoints

def findSnakeDirection(snake):
    if snake.joints == []:
        direction = snake.findDirection(snake.head, snake.end)
    else:
        direction = snake.findDirection(snake.head, snake.joints[0])

    return direction

def relativePoints(head, point):
    return Point( point.x - head.x, point.y - head.y )

def getAbsoluteStateForSingleAgent(snake, k = 3):
    state = []
    state.append(snake.head)  # head

    if(len(foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, k))

    state.append(findSnakeDirection(snake))   # direction

    return state

def getRelativeStateForSingleAgent(snake, k = 3):
    state = []

    if(len(foodList)):          # k nearest points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, k)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))
        state.extend(relativeFoodPoints)

    state.append(findSnakeDirection(snake))   # direction

    return state