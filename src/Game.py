from Snake import Snake
import Food
from Action import Action
import numpy as np

class Game:
    def __init__(self, numOfSnakes=2, gridSize=10, maxEpisodeLength=300):
        self.snakes = []
        for idx in range(numOfSnakes):
            self.snakes.append(Snake(gridSize, idx) )

        Food.createFood(10, self.snakes)
        self.gameLength = maxEpisodeLength
        self.time_step = 0
        return

    def __str__(self):
        print_message = "Time " + str(self.time_step) + "\n"
        print_message += "Food = " + str(map(str, Food.foodList)) + "\n"
        for s in self.snakes:
            print_message += str(s) + "\n"
        return print_message

    def move(self, actionsList=[]):
        """
        Takes in a list of actions corresponding to each snake in the game
        Returns boolean indicating whether the game has ended
        """
        assert len(actionsList)==len(self.snakes), "Deficiency of actions provided."

        action_validity_check = []
        for i in range(len(self.snakes)):
            s = self.snakes[i]
            permissible_actions = s.permissible_actions()
            action_validity_check.append( actionsList[i] in permissible_actions )
        assert all(action_validity_check), "At least one action is invalid"

        self.time_step += 1

        single_step_rewards = []
        for i in range(len(actionsList)):
            s = self.snakes[i]
            a = actionsList[i]

            s.moveInDirection(a)
            if s.didEatFood():
                single_step_rewards.append(1)
            else:
                single_step_rewards.append(0)

        for i in range(len(self.snakes)):
            for j in range(i+1, len(self.snakes)):
                if self.snakes[i].didHitSnake(self.snakes[j]):
                    self.snakes[i].backtrack()
                    self.snakes[i].killSnake()

        return ( single_step_rewards, not (self.time_step == self.gameLength or all([not s.alive for s in self.snakes])) )

    def endGame(self):
        Food.foodList = []
        return [s.score for s in self.snakes]
