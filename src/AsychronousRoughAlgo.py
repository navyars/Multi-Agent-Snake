import tensorflow as tf
import Agent
import FunctionApproximator
import Game
from  Constants import *
import numpy as np
import random


def epsilon_greedy_action(snake, sess, nn, input_data, epsilon):
        possible_actions = snake.permissible_actions()
        best_action, _ = nn.max_permissible_Q(sess, input_data, possible_actions)
        possible_actions.remove(best_action)
        prob = random.uniform(0, 1)
        if prob <= epsilon:
            return best_action
        else:
            return random.choice(possible_actions)
        #choose action according to epsilon-greedy
        #return the action to be chosen

def best_q(snake, sess, nn, input_data):
    return nn.max_permissible_Q(sess, input_data, snake.permissible_actions())[1]

def mainAlgorithm (max_time_steps=1000, reward=1, penalty=-10, asyncUpdate=30, globalUpdate=120):
    policyNetwork = []
    targetNetwork = []
    policySess = []
    targetSess = []
    epsilon = []

    #Initializing the 2*n neural nets
    for idx in range(numberOfSnakes):
        policyNetwork.append(FunctionApproximator.NeuralNetwork(3))
        targetNetwork.append(FunctionApproximator.NeuralNetwork(3))
        policySess.append(tf.Session(graph=policyNetwork[idx].graph))
        targetSess.append(tf.Session(graph=targetNetwork[idx].graph))
        policyNetwork[idx].init(policySess[idx])
        targetNetwork[idx].init(targetSess[idx])
        epsilon.append(np.random.normal(0.8, 0.1))

    time_steps = 0

    while time_steps <= max_time_steps:
        g = Game.Game(numberOfSnakes, gridSize, globalEpisodeLength)
        #Start a Game
        snake_list = g.snakes
        episodeRunning = True
        pastStateAlive = [True for i in range(numberOfSnakes)]
        actions_taken = [0 for j in range(numberOfSnakes)]
        initial_state = [0]*numberOfSnakes

        while episodeRunning: #Meaning in an episode
            for idx in range(numberOfSnakes):
                pruned_snake_list = [ snake for snake in snake_list if snake != snake_list[idx] ]
                #initial_state[idx] = Agent.getRelativeStateForMultipleAgents(g.snakes[idx],pruned_snake_list)
                initial_state[idx] = Agent.getRelativeStateForSingleAgent(g.snakes[idx]) #Can either be this or the above line
                if g.snakes[idx].alive:
                    actions_taken[idx] = epsilon_greedy_action(g.snakes[idx], policySess[idx], policyNetwork[idx], initial_state[idx], epsilon[idx])
                    pastStateAlive[idx] = True
                else:
                    actions_taken[idx] = None
                    pastStateAlive[idx] = False

            try:
                episodeRunning = g.Move(actions_taken)
            except AssertionError:
                print("Error making moves {} in game :\n{}".format(actionsList, g))

            #Now we transition to the next state
            time_steps += 1

            for idx in range(numberOfSnakes):
                if (pastStateAlive[idx]): # To check if snake was already dead or just died
                    pruned_snake_list = [ snake for snake in snake_list if snake != snake_list[idx] ]
                    # final_state[idx] = Agent.getRelativeStateForMultipleAgents(g.snakes[idx],pruned_snake_list)
                    final_state = Agent.getRelativeStateForSingleAgent(g.snakes[idx])
                    single_step_reward = reward if snake_list[idx].alive else penalty # if it was alive in the past state but it isn't after the move, then it just died and deserves a penalty

                    if (not episodeRunning or not(g.snakes[idx].Alive)) or (time_steps % asyncUpdate == 0): # Training is done on the snake only on terminal state
                        policyNetwork[idx].update_gradient(policySess[idx], initial_state[idx], actions_taken[idx], single_step_reward)
                        policyNetwork[idx].train(policySess[idx], initial_state[idx], actions_taken[idx], reward[idx])
                        policyNetwork.reset_accumulator(policySess[idx])
                    else: #Else only an update is done
                        next_state_best_Q = best_q(g.snakes[idx], targetSess[idx], targetNetwork[idx], final_state)
                        policyNetwork[idx].update_gradient(policySess[idx], initial_state[idx], actions_taken[idx], single_step_reward, next_state_best_Q)

                    if time_steps % globalUpdate == 0:
                        checkpoint_path = "transfer_{}.ckpt".format(idx)
                        policyNetwork[idx].save_model(policySess, checkpoint_path)
                        targetNetwork[idx].restore_model(targetSess, checkpoint_path)
