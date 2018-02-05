class MatShape(object):
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def get_size(self):
        return self.width * self.height * self.depth
