class Base:
    def __init__(self):
        self.WIDTH = 600
        self.HEIGHT = 600
        self.BLOCK_SIZE = 40
        self.MaxFoodIndex = (self.WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE