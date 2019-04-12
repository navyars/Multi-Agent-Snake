from numpy.random import randint

from Point import Point
from Action import Action

class Snake:

    def __init__(self, gridSize, identity):
        self.head = Point(randint(10, gridSize - 10), randint(10, gridSize - 10)) # generate point with at least 10 units gap from any wall
        self.end = Point( self.head.x - 5, self.head.y )
        self.joints = []
        self.id = identity
        self.alive = True

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

    def moveInDirection(self, action, snakes=[]):
        assert (action in self.permissible_actions()), "Action not allowed in this state."
        prev_head, prev_joints, prev_end = self._copy(self.head, self.joints, self.end)

        # move the snake in the direction specified
        if self.joints == []:
            direction = self.findDirection(self.head, self.end)
            if direction != action: # add joint when snake changes direction
                self.joints.append( Point.fromPoint(self.head))
            self.head = self._update_point(self.head, action)
            self.end = self._update_point(self.end, direction)
        else:
            direction = self.findDirection(self.head, self.joints[0])
            if direction != action: # add joint when snake changes direction
                self.joints.insert(Point.fromPoint(self.head), 0)
            self.head = self._update_point(self.head, action)

            direction = self.findDirection(self.joints[-1], self.end)
            self.end = self._update_point(self.end, direction)
            if self.end == self.joints[-1]: # pop joint if end has reached it
                self.joints = self.joints[:-1]

        # check if the snake has collided with wall or other snakes. If true, undo movement and kill it
        if self.didHitWall():
            self.head, self.joints, self.end = self._copy(prev_head, prev_joints, prev_end)
            self.kill()

        for s in snakes:
            if self.didHitSnake(s): # TODO: This should mostly be a simultaneous check across all snakes, after their movements
                self.head, self.joints, self.end = self._copy(prev_head, prev_joints, prev_end)
                self.kill()

        return

    def didHitSnake(self, opponent_snake):
            body = [opponent_snake.head]
            body.extend( opponent_snake.joints )
            body.append(opponent_snake.end)

            p = self.head
            if p == opponent_snake.head :
                return (self.score <= opponent_snake.score) # if heads collide, the larger snake remains. If the scores are equal, then both snakes should die.

            for i in xrange(len(body)-1):
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
