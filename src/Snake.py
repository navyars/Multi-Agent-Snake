from numpy.random import randint

from Point import Point
from Action import Action
from Constants import *

class Snake:
    snakeList = []

    def __init__(self, gridSize, identity):
        occupiedPoints = []
        for snake in Snake.snakeList:
            body = snake.getBodyList()
            bodyPoints = Point.returnBodyPoints(body)
            occupiedPoints.extend(bodyPoints)

        while True:
            self.head = Point(randint(10, gridSize - 10), randint(10, gridSize - 10)) # generate point with at least 10 units gap from any wall
            self.end = Point( self.head.x - 5, self.head.y )
            self.joints = []
            body = self.getBodyList()
            bodyPoints = Point.returnBodyPoints(body)
            if not bool(self.lists_overlap(bodyPoints, occupiedPoints)):
                break

        self.id = identity
        self.alive = True
        self.score = 0
        self.prev_head, self.prev_joints, self.prev_end = self._copy(self.head, self.joints, self.end)

        Snake.snakeList.append(self)
        return

    def __str__(self):
        if self.alive:
            body = self.getBodyList()
            body_str = str(map(str,body))
            print_message = "Snake {}\n\tPosition = {}\n\tScore = {}\n".format(self.id, body_str, self.score)
            return print_message
        else:
            print_message = "Snake {}\n\tDead. Score = {}\n".format(self.id, self.score)
            return print_message

    def lists_overlap(self, list1, list2):
        for point in list1:
            if point in list2:
                return True
        return False

    def getBodyList(self):
        body = [self.head]
        body.extend(self.joints)
        body.append(self.end)
        return body

    def didEatFood(self, food):
        return (self.head in food.foodList)

    def incrementScore(self, reward):
        self.score += reward
        return

    def didHitWall(self):
        return (self.head.x == 0 or self.head.x == gridSize or self.head.y == 0 or self.head.y == gridSize)

    def _update_point(self, p, direction):
        if direction == Action.TOP:
            p.y += 1
        elif direction == Action.DOWN:
            p.y -= 1
        elif direction == Action.RIGHT:
            p.x += 1
        elif direction == Action.LEFT:
            p.x -= 1
        return p

    def _copy(self, head, joints, end):
        _head = Point.fromPoint(head)
        _joints = [Point.fromPoint(p) for p in joints]
        _end = Point.fromPoint(end)
        return _head, _joints, _end

    def backtrack(self):
        self.head, self.joints, self.end = self._copy(self.prev_head, self.prev_joints, self.prev_end)
        return

    def moveInDirection(self, action):
        assert (action in self.permissible_actions()), "Action not allowed in this state."
        self.prev_head, self.prev_joints, self.prev_end = self._copy(self.head, self.joints, self.end)

        # move the snake in the direction specified
        if self.joints == []:
            direction = self.findDirection(self.head, self.end)
            if direction != action: # add joint when snake changes direction
                self.joints.append(Point.fromPoint(self.head))
            self.head = self._update_point(self.head, action)
            self.end = self._update_point(self.end, direction)
        else:
            direction = self.findDirection(self.head, self.joints[0])
            if direction != action: # add joint when snake changes direction
                self.joints.insert(0, Point.fromPoint(self.head))
            self.head = self._update_point(self.head, action)

            direction = self.findDirection(self.joints[-1], self.end)
            self.end = self._update_point(self.end, direction)
            if (self.end.x == self.joints[-1].x) and (self.end.y == self.joints[-1].y): # pop joint if end has reached it
                self.joints = self.joints[:-1]

        return

    def didHitSnake(self, opponent_snake):
        body = opponent_snake.getBodyList()

        p = self.head
        if p == opponent_snake.head :
            return (self.score <= opponent_snake.score) # if heads collide, the larger snake remains. If the scores are equal, then both snakes should die.

        for i in range(len(body)-1):
            p1 = body[i]
            p2 = body[i+1]
            if p1.x == p2.x: #vertical line
                if p.x == p1.x:
                    lim1, lim2 = tuple(sorted([p1.y, p2.y]))
                    if p.y in range( lim1, lim2 +1):
                        return True
            else: #horizontal line
                if p.y == p1.y:
                    lim1, lim2 = tuple(sorted([p1.x, p2.x]))
                    if p.x in range(lim1, lim2 + 1):
                        return True

        return False

    def findDirection(self, p1, p2):  # Direction of P1 with reference to P2
        if p1.x - p2.x == 0 and p1.y - p2.y < 0:
            return Action.DOWN
        elif p1.x - p2.x == 0 and p1.y - p2.y > 0:
            return Action.TOP
        elif p1.x - p2.x > 0 and p1.y - p2.y == 0:
            return Action.RIGHT
        elif p1.x - p2.x < 0 and p1.y - p2.y == 0:
            return Action.LEFT

    def growSnake(self): # Grows the snake in the direction of the tail. Should be called before moveInDirection
        if self.joints == []:
            direction = self.findDirection(self.end, self.head)
        # Finding direction from the last joint/head to tail as it is in this direction the increment should happen
        else:
            direction = self.findDirection(self.end, self.joints[-1])
        self.end = self._update_point(self.end, direction)

    def permissible_actions(self):  # Returns a list of permissible actions
        actions = []
        if self.joints == []:
            direction = self.findDirection(self.end, self.head)
        else:
            direction = self.findDirection(self.joints[0], self.head)
        for act in (Action):
            if act != direction:
                actions.append(act)
        return actions

    def killSnake(self, food):
        self.alive = False

        body = self.getBodyList()
        points = Point.returnBodyPoints(body)
        food.addFoodToList(points)

        Snake.snakeList.remove(self)
        del self.head
        del self.end
        del self.joints
