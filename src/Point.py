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
