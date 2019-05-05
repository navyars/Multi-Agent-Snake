import copy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def fromPoint(cls, p):
        assert isinstance(p, Point), "Invalid parameter passed"
        return cls(p.x, p.y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    @staticmethod
    def returnBodyPoints(body):
        points = copy.deepcopy(body)
        for i in range(len(body) - 1):
            p1 = body[i]
            p2 = body[i + 1]
            if p1.x == p2.x:    # vertical line
                for y in range(min(p1.y, p2.y) + 1, max(p1.y, p2.y)):
                    points.append(Point(p1.x, y))
            else:   # horizontal line
                for x in range(min(p1.x, p2.x) + 1, max(p1.x, p2.x)):
                    points.append(Point(x, p1.y))

        return points
