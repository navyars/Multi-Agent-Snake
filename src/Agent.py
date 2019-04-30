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
from Action import Action
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

def findOtherAgentNearestPoint(snake, agent):
    body = [agent.head]
    body.extend(agent.joints)
    body.append(agent.end)
    points = Point.returnBodyPoints(body)

    dist = []
    for point in points:
        dist.append(calculateDistance(point, snake.head))
    dist = np.asarray(dist)

    minIndices = np.where(dist == dist.min())
    if len(minIndices[0]) == 1:
        return points[minIndices[0][0]]
    else:
        direction = findSnakeDirection(snake)
        for index in range(len(minIndices[0])):
            if direction == Action.TOP:
                if(points[minIndices[0][index]].x == snake.head.x and points[minIndices[0][index]].y >= snake.head.y):
                    return points[minIndices[0][index]]
            elif direction == Action.DOWN:
                if(points[minIndices[0][index]].x == snake.head.x and points[minIndices[0][index]].y <= snake.head.y):
                    return points[minIndices[0][index]]
            elif direction == Action.RIGHT:
                if(points[minIndices[0][index]].y == snake.head.y and points[minIndices[0][index]].x >= snake.head.x):
                    return points[minIndices[0][index]]
            elif direction == Action.LEFT:
                if(points[minIndices[0][index]].y == snake.head.y and points[minIndices[0][index]].x <= snake.head.x):
                    return points[minIndices[0][index]]
        return points[minIndices[0][0]]

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

def getAbsoluteStateForMultipleAgents(snake, agentList, k = 3):
    state = []
    state.append(snake.head)    # head

    if(len(foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, k))

    state.append(findSnakeDirection(snake))   # direction

    for agent in agentList:
        state.append(agent.head)    # other agent's head
        state.append(findOtherAgentNearestPoint(snake, agent)) # other agent's nearest  body point

    return state

def getRelativeStateForMultipleAgents(snake, agentList, k = 3):
    state = []

    if(len(foodList)):          # k nearest points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, k)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))
        state.extend(relativeFoodPoints)

    state.append(findSnakeDirection(snake))   # direction

    for agent in agentList:
        state.append(relativePoints(snake.head, agent.head))    # other agent's head
        state.append(relativePoints(snake.head,findOtherAgentNearestPoint(snake, agent))) # other agent's nearest  body point

    return state