import json
from base import Base
import pygame as pg
import numpy as np
class Snake(Base):
    def __init__(self , length = 5):
        super().__init__()
        self.body = []
        self.length = length
        self.x = [self.BLOCK_SIZE]*self.length
        self.y = [self.BLOCK_SIZE]*self.length
        self.direction = 'right'

    def increase(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)
    def move_left(self):
        if self.direction != 'right':
            self.direction = 'left'
    def move_right(self):
        if self.direction != 'left':
            self.direction = 'right'
    def move_up(self):
        if self.direction != 'down':
            self.direction = 'up'
    def move_down(self):
        if self.direction != 'up':
            self.direction = 'down'

    def move(self):
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]
        if self.direction == 'left':
            self.x[0] -= self.BLOCK_SIZE
        elif self.direction == 'right':
            self.x[0] += self.BLOCK_SIZE
        elif self.direction == 'up':
            self.y[0] -= self.BLOCK_SIZE
        elif self.direction == 'down':
            self.y[0] += self.BLOCK_SIZE
    def checkCollision(self):
        headx = self.x[0]
        heady = self.y[0]
        for i in range(self.length-1 , 0 , -1):
            if self.x[i] == headx and self.y[i] == heady:
                return True
        if self.x[0] < 0 or self.x[0] > self.WIDTH-self.BLOCK_SIZE:
            return True
        if self.y[0] < 0 or self.y[0] > self.HEIGHT - self.BLOCK_SIZE:
            return True
        return False
class Food(Base):
    def __init__(self  ):
        super().__init__()
        self.x = self.BLOCK_SIZE*4
        self.y = self.BLOCK_SIZE*5
    def move(self , snake):
        import random
        x = random.randint(0 , self.MaxFoodIndex)*self.BLOCK_SIZE
        y = random.randint(0 , self.MaxFoodIndex)*self.BLOCK_SIZE
        clean = True
        for i in range(0 , snake.length):
            if snake.x[i] == x and snake.y[i] == y:
                clean = False
        if clean :
            self.x = x
            self.y = y
            return
        else :
            self.move(snake)



class Game(Base):
    def __init__(self):
        super().__init__()



        self.snake = Snake(length = 1)
        self.food = Food()
        self.score = 0
        self.record = 0
        self.game_over = False
        self.reward = 0
    def play(self):
        self.snake.move()
        self.reward = -0.1

        #snake eats food
        if self.snake.x[0] == self.food.x and self.snake.y[0] == self.food.y:
            self.snake.increase()
            self.food.move(self.snake)
            self.score+=10
            self.reward = 10
            #if self.record < self.score :
                #self.record = self.score
               # self.saveRecord()
        #snake eats its own body
        flag = self.snake.checkCollision()
        if flag :
            self.game_over = True
            self.reward = -100

    def saveRecord(self):
        import os
        if not os.path.exists('./resources'):
            os.mkdir('./resources')
        FileName = 'assets/resources/data.json'
        data = {"record":self.record}
        with open(FileName, 'w') as file:
            json.dump(data, file ,indent=4)
    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False

    def get_next_dir(self, move):
        new_dir = 'right'
        if np.array_equal(move, [1, 0, 0, 0]):
            new_dir = 'right'
        if np.array_equal(move, [0, 1, 0, 0]):
            new_dir = 'down'
        if np.array_equal(move, [0, 0, 1, 0]):
            new_dir = 'left'
        if np.array_equal(move, [0, 0, 0, 1]):
            new_dir = 'up'
        return new_dir
    def run(self , move):
        while True:
            new_dir = self.get_next_dir(move)
            if new_dir == 'right':
                self.snake.move_right()
            elif new_dir == 'down':
                self.snake.move_down()
            elif new_dir == 'left':
                self.snake.move_left()
            elif new_dir == 'up':
                self.snake.move_up()
            self.play()
            return self.reward, self.game_over, self.score

    def isDanger(self , point):
        point_x = point[0]
        point_y = point[1]

        for i in range(1 , self.snake.length):
            if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                return True
        if point_x < 0 or point_x > self.WIDTH-self.BLOCK_SIZE or point_y < 0 or point_y > self.HEIGHT-self.BLOCK_SIZE:
            return True
        return False