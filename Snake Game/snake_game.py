import pygame as pg
import random
import numpy as np
from base import Base
import json
import os

class Snake(Base):
    def __init__(self, screen, length=5):
        super().__init__()
        self.body = []
        self.length = length
        self.screen = screen
        self.snake_img = pg.image.load("block.jpg")
        # Initialize snake coordinates as a list of tuples (x,y)
        start_x = self.BLOCK_SIZE * 5
        start_y = self.BLOCK_SIZE * 5
        self.x = [start_x] * self.length
        self.y = [start_y] * self.length
        self.direction = 'right'

    def draw(self):
        for i in range(self.length):
            self.screen.blit(self.snake_img, (self.x[i], self.y[i]))

    def increase(self):
        self.length += 1
        # Append new segment at offscreen position, will be corrected in move()
        self.x.append(-self.BLOCK_SIZE)
        self.y.append(-self.BLOCK_SIZE)

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
        # Move body segments back
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]
        # Move head based on direction
        if self.direction == 'left':
            self.x[0] -= self.BLOCK_SIZE
        elif self.direction == 'right':
            self.x[0] += self.BLOCK_SIZE
        elif self.direction == 'up':
            self.y[0] -= self.BLOCK_SIZE
        elif self.direction == 'down':
            self.y[0] += self.BLOCK_SIZE
        self.draw()

    def check_collision(self):
        head_x = self.x[0]
        head_y = self.y[0]
        # Check collision with body
        for i in range(1, self.length):
            if self.x[i] == head_x and self.y[i] == head_y:
                return True
        # Check collision with boundaries
        if head_x < 0 or head_x >= self.WIDTH:
            return True
        if head_y < 0 or head_y >= self.HEIGHT:
            return True
        return False

class Food(Base):
    def __init__(self, screen):
        super().__init__()
        self.screen = screen
        self.food_img = pg.image.load("apple.jpg")
        self.x = self.BLOCK_SIZE * random.randint(1 , 14)
        self.y = self.BLOCK_SIZE * random.randint(1 , 14)

    def draw(self):
        self.screen.blit(self.food_img, (self.x, self.y))

    def move(self, snake):
        while True:
            x = random.randint(0, self.MaxFoodIndex) * self.BLOCK_SIZE
            y = random.randint(0, self.MaxFoodIndex) * self.BLOCK_SIZE
            collision = False
            for i in range(snake.length):
                if snake.x[i] == x and snake.y[i] == y:
                    collision = True
                    break
            if not collision:
                self.x = x
                self.y = y
                break

class Game(Base):
    def __init__(self):
        super().__init__()
        pg.init()
        pg.display.set_caption('Snake Game')
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pg.time.Clock()
        self.Screen_Update = pg.USEREVENT + 1
        self.timer = 1  # delay in ms between moves (lower is faster)
        pg.time.set_timer(self.Screen_Update, self.timer)
        self.snake = Snake(self.screen)
        self.food = Food(self.screen)
        self.score = 0
        self.game_over = False
        self.reward = 0

    def play(self):
        self.screen.fill((0, 0, 0))
        self.snake.move()
        self.food.draw()
        self.display_score()
        self.reward = -0.1

        # Check if snake eats food
        if self.snake.x[0] == self.food.x and self.snake.y[0] == self.food.y:
            self.snake.increase()
            self.food.move(self.snake)
            self.score += 10
            self.reward = 10

        # Check collision
        if self.snake.check_collision():
            self.game_over = True
            self.reward = -100

    def display_score(self):
        font = pg.font.Font('assets/fonts/PressStart2P-Regular.ttf', 20)
        msg = 'Score : ' + str(self.score)
        scores = font.render(msg, True, (255, 255, 255))
        self.screen.blit(scores, (10, 10))

    def show_restart_message(self):
        font = pg.font.Font('assets/fonts/PressStart2P-Regular.ttf', 20)
        message = f'The Score: {self.score}  -  Press R to restart'
        text = font.render(message, True, (255, 255, 255))
        # Correct x,y positioning: x uses WIDTH, y uses HEIGHT
        self.screen.blit(text, (self.WIDTH // 5, self.HEIGHT // 2))

    def reset(self):
        self.snake = Snake(self.screen)
        self.food = Food(self.screen)
        self.score = 0
        self.game_over = False

    def get_next_dir(self, move):
        if np.array_equal(move, [1, 0, 0, 0]):
            return 'right'
        if np.array_equal(move, [0, 1, 0, 0]):
            return 'down'
        if np.array_equal(move, [0, 0, 1, 0]):
            return 'left'
        if np.array_equal(move, [0, 0, 0, 1]):
            return 'up'
        return self.snake.direction  # default to current direction if no match

    def run(self, move):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            if event.type == self.Screen_Update:
                if not self.game_over:
                    self.play()
                else:
                    self.show_restart_message()
                pg.display.update()
                self.clock.tick(60)  # limit FPS to 60
                break

        new_dir = self.get_next_dir(move)
        if new_dir == 'right':
            self.snake.move_right()
        elif new_dir == 'down':
            self.snake.move_down()
        elif new_dir == 'left':
            self.snake.move_left()
        elif new_dir == 'up':
            self.snake.move_up()

        return self.reward, self.game_over, self.score

    def isDanger(self, point):
        point_x, point_y = point
        for i in range(1, self.snake.length):
            if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                return True
        if point_x < 0 or point_x >= self.WIDTH or point_y < 0 or point_y >= self.HEIGHT:
            return True
        return False

