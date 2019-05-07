from Snake import Snake
from Food import Food
from Action import Action
import numpy as np

class Game:
    def __init__(self, numOfSnakes=2, gridSize=10, maxEpisodeLength=300):
        self.snakes = []
        for idx in range(numOfSnakes):
            self.snakes.append(Snake(gridSize, idx) )

        self.food = Food(Snake.snakeList)
        self.gameLength = maxEpisodeLength
        self.time_step = 0
        return

    def __str__(self):
        print_message = "Time " + str(self.time_step) + "\n"
        print_message += "Food = " + str(map(str, self.food.foodList)) + "\n"
        for s in self.snakes:
            print_message += str(s) + "\n"
        return print_message

    def move(self, actionsList=[]):
        """
        Takes in a list of actions corresponding to each snake in the game.
        If a snake is dead, then its corresponding position in actionsList simply holds None.
        Returns boolean indicating whether the game has ended
        """
        assert len(actionsList)==len(self.snakes), "Deficiency of actions provided."

        action_validity_check = []
        for i in range(len(self.snakes)):
            s = self.snakes[i]
            if s.alive:
                permissible_actions = s.permissible_actions()
                action_validity_check.append( actionsList[i] in permissible_actions )
        assert all(action_validity_check), "At least one action is invalid"

        self.time_step += 1

        single_step_rewards = [0]*len(self.snakes)
        for i in range(len(actionsList)):
            snake = self.snakes[i]
            if snake.alive:
                snake.moveInDirection( actionsList[i] )
                if snake.didEatFood(self.food):
                    snake.incrementScore(1)
                    snake.growSnake()
                    self.food.eatFood(snake.head, Snake.snakeList)
                    single_step_rewards[i] = 1
                elif snake.didHitWall():
                    snake.backtrack()
                    snake.killSnake(self.food)
                    snake.incrementScore(-10)
                    single_step_rewards[i] = -10

        for i in range(len(self.snakes)):
            for j in range(i+1, len(self.snakes)):
                if not (self.snakes[i].alive and self.snakes[j].alive):
                    continue
                if self.snakes[i].didHitSnake(self.snakes[j]):
                    self.snakes[i].backtrack()
                    self.snakes[i].killSnake(self.food)

        return ( single_step_rewards, not (self.time_step == self.gameLength or all([not s.alive for s in self.snakes])) )

    def endGame(self):
        scoreboard = [s.score for s in self.snakes]
        del self.food
        del self.snakes
        return scoreboard
