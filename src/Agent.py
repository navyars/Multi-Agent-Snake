###   SINGLE SNAKE  ###
    # absolute
        # head
        # k nearest points - from head
        # direction - action enums
    # relative
        # k nearest points
        # direction - action enums
        # nearest wall

###  MULITPLE SNAKES  ###
    # absolute
        # head
        # k nearest points
        # direction - action enums
        # other snakes head
        # other snakes closest point
        # direction of the other agent
        # nearest point of the snake to the other agent
    # relative
        # k nearest points
        # direction - action enums
        # other snakes head
        # other snakes closest point
        # direction of the other agent
        # nearest point of the snake to the other agent
        # nearest wall

from math import *
import numpy as np

from Snake import Snake
from Point import Point
from Action import Action
from Constants import *

def calculateDistance(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def findKNearestPoints(head, food, k):
    dist = []
    nearestPoints = []

    for f in food.foodList:
        dist.append(calculateDistance(f, head))

    if len(food.foodList) < k:
        k = len(food.foodList)

    argmin = np.argpartition(dist, k)

    for i in range(k):
        nearestPoints.append(food.foodList[argmin[i]])

    return nearestPoints

def findSnakeDirection(snake):
    if snake.joints == []:
        direction = snake.findDirection(snake.head, snake.end)
    else:
        direction = snake.findDirection(snake.head, snake.joints[0])

    return direction

def relativePoints(head, point):
    return Point( point.x - head.x, point.y - head.y )

def calculateMinDistPoint(snake, points):
    dist = []
    for point in points:
        dist.append(calculateDistance(point, snake.head))
    dist = np.asarray(dist)

    minIndices = np.where(dist == dist.min())
    if minIndices[0].shape[0] == 1:
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

def findNearestWall(snake, gridSize = 50):   # checks the perpendicular distance from the
    points = []                             # snake's head to all the walls and returns
    points.append(Point(0, snake.head.y))         # minimum of those
    points.append(Point(snake.head.x, 0))
    points.append(Point(gridSize, snake.head.y))
    points.append(Point(snake.head.x, gridSize))

    minDistPoint = calculateMinDistPoint(snake, points)
    return minDistPoint

def findOtherSnakeNearestPoint(snake1, snake2): # snake2's nearest body point to the head of snake1
    body = [snake2.head]
    body.extend(snake2.joints)
    body.append(snake2.end)
    points = Point.returnBodyPoints(body)

    minDistPoint = calculateMinDistPoint(snake1, points)
    return minDistPoint

def getAbsoluteStateForSingleAgent(snake, food, k = 3):
    state = []
    state.append(snake.head)    # head

    if(len(food.foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, food, k))

    state.append(findSnakeDirection(snake))   # direction

    return state

def getRelativeStateForSingleAgent(snake, gridSize, food, k = 3):
    state = []

    if(len(food.foodList)):          # k nearest points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, food, k)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))
        state.extend(relativeFoodPoints)

    state.append(findSnakeDirection(snake))   # direction

    state.append(relativePoints(snake.head,findNearestWall(snake, gridSize)))  # nearest wall point

    return state

def getAbsoluteStateForMultipleAgents(snake, agentList, food, k = 3):
    state = []
    state.append(snake.head)    # head

    if(len(food.foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, food, k))

    state.append(findSnakeDirection(snake))   # direction

    for agent in agentList:
        if agent.alive == True:
            state.append(agent.head)    # other agent's head
            state.append(findOtherSnakeNearestPoint(snake, agent)) # other agent's nearest body point
            state.append(findSnakeDirection(agent))   # direction of the other agent
            state.append(findOtherSnakeNearestPoint(agent, snake))  # nearest body point of the snake to the other agent's head
        else:
            state.extend([Point(-1, -1), Point(-1, -1), -1, Point(-1, -1)])

    return state

def getRelativeStateForMultipleAgents(snake, agentList, gridSize, food, k = 3):
    state = []

    if(len(food.foodList)):          # k nearest points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, food, k)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))
        state.extend(relativeFoodPoints)

    state.append(findSnakeDirection(snake))   # direction

    for agent in agentList:
        if agent.alive == True:
            state.append(relativePoints(snake.head, agent.head))    # other agent's head
            state.append(relativePoints(snake.head,findOtherSnakeNearestPoint(snake, agent))) # other agent's nearest  body point
            state.append(findSnakeDirection(agent))   # direction of the other agent
            state.append(relativePoints(snake.head,findOtherSnakeNearestPoint(agent, snake)))  # nearest body point of the snake to the other agent's head
        else:
            state.extend([Point(-1, -1), Point(-1, -1), -1, Point(-1, -1)])

    state.append(relativePoints(snake.head,findNearestWall(snake, gridSize)))  # nearest wall point

    return state

def getStateLength(multipleAgents, k):
    if multipleAgents == False:
        return 9
    elif multipleAgents == True:
        return 3 + (k*2) + (numberOfSnakes-1)*7


def getState(snake, agentList, gridSize, relative, multipleAgents, food, k, normalize=False):
    state = []
    if snake.alive == False:
        return [-1] * getStateLength(multipleAgents, k)
    if relative == False and multipleAgents == False:
        state.extend(getAbsoluteStateForSingleAgent(snake, food, k))
    elif relative == False and multipleAgents == True:
        state.extend(getAbsoluteStateForMultipleAgents(snake, agentList, food, k))
    elif relative == True and multipleAgents == False:
        state.extend(getRelativeStateForSingleAgent(snake, gridSize, food, k))
    elif relative == True and multipleAgents == True:
        state.extend(getRelativeStateForMultipleAgents(snake, agentList, gridSize, food, k))

    flatState = []
    for entry in state:
        if isinstance(entry, Point):
            if normalize:
                flatState.append( entry.x * 1.0 / gridSize )
                flatState.append( entry.y * 1.0 / gridSize )
            else:
                flatState.append(entry.x)
                flatState.append(entry.y)
        elif isinstance(entry, Action):
            if normalize:
                flatState.append(entry.value * 1.0 / 3)
            else:
                flatState.append(entry.value)
        else:
            flatState.append(entry)

    return np.asarray(flatState)
