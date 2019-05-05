import tensorflow as tf
import Agent
import FunctionApproximator
import Game
from Constants import *
import numpy as np
import random

from threading import Lock, Thread
from queue import Queue
from multiprocessing import cpu_count

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

def async_Q(max_time_steps, reward, penalty, asyncUpdate, globalUpdate, policyNetwork, policySess, targetNetwork, targetSess, lock, queue):
    time_steps = 0
    epsilon = []
    for idx in range(numberOfSnakes):
        while True:
            e = np.random.normal(0.8, 0.1)
            if e < 1:
                epsilon.append(e)
                break
    while True:
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
                if g.snakes[idx].alive:
                    initial_state[idx] = Agent.getRelativeStateForSingleAgent(g.snakes[idx], 50) #Can either be this or the above line
                    actions_taken[idx] = epsilon_greedy_action(g.snakes[idx], policySess[idx], policyNetwork[idx], initial_state[idx], epsilon[idx])
                    pastStateAlive[idx] = True
                else:
                    actions_taken[idx] = None
                    pastStateAlive[idx] = False

            try:
                episodeRunning = g.move(actions_taken)
            except AssertionError:
                print("Error making moves {} in game :\n{}".format(actionsList, g))

            #Now we transition to the next state
            time_steps += 1
            lock.acquire()
            T = queue.get()
            T += 1
            queue.put(T)
            lock.release()


            for idx in range(numberOfSnakes):
                if (pastStateAlive[idx]): # To check if snake was already dead or just died
                    pruned_snake_list = [ snake for snake in snake_list if snake != snake_list[idx] ]
                    # final_state[idx] = Agent.getRelativeStateForMultipleAgents(g.snakes[idx],pruned_snake_list)
                    final_state = Agent.getRelativeStateForSingleAgent(g.snakes[idx], 50)
                    single_step_reward = 0
                    if not snake_list[idx].alive: # if it was alive in the past state but it isn't after the move, then it just died and deserves a penalty
                        single_step_reward = penalty
                    elif snake_list[idx].didEatFood(): # it ate food, boost its reward!
                        single_step_reward = reward

                    if not episodeRunning or not g.snakes[idx].alive: # Training is done on the snake only on terminal state
                        lock.acquire()
                        policyNetwork[idx].update_gradient(policySess[idx], initial_state[idx], actions_taken[idx], single_step_reward)
                        policyNetwork[idx].train(policySess[idx], initial_state[idx], actions_taken[idx], single_step_reward)
                        policyNetwork.reset_accumulator(policySess[idx])
                        lock.release()
                    else: #Else only an update is done
                        next_state_best_Q = best_q(g.snakes[idx], targetSess[idx], targetNetwork[idx], final_state)
                        lock.acquire()
                        policyNetwork[idx].update_gradient(policySess[idx], initial_state[idx], actions_taken[idx],
                                                           single_step_reward, next_state_best_Q)
                        if time_steps % asyncUpdate == 0:
                            policyNetwork[idx].train(policySess[idx], initial_state[idx], actions_taken[idx],
                                                     single_step_reward, next_state_best_Q)
                            policyNetwork.reset_accumulator(policySess[idx])
                        lock.release()

                    T = queue.get()
                    queue.put(T)
                    if T % globalUpdate == 0:
                        checkpoint_path = "transfer_{}.ckpt".format(idx)
                        policyNetwork[idx].save_model(policySess[idx], checkpoint_path)
                        targetNetwork[idx].restore_model(targetSess[idx], checkpoint_path)
        print("Episode done on thread")
        T = queue.get()
        queue.put(T)
        if T >= max_time_steps:
            break

def mainAlgorithm(max_time_steps=1000, reward=1, penalty=-10, asyncUpdate=30, globalUpdate=120):
    policyNetwork = []
    targetNetwork = []
    policySess = []
    targetSess = []

    #Initializing the 2*n neural nets
    for idx in range(numberOfSnakes):
        policyNetwork.append(FunctionApproximator.NeuralNetwork(9))
        targetNetwork.append(FunctionApproximator.NeuralNetwork(9))
        policySess.append(tf.Session(graph=policyNetwork[idx].graph))
        targetSess.append(tf.Session(graph=targetNetwork[idx].graph))
        policyNetwork[idx].init(policySess[idx])
        targetNetwork[idx].init(targetSess[idx])
        checkpoint_path = "transfer_{}.ckpt".format(idx)
        policyNetwork[idx].save_model(policySess[idx], checkpoint_path)
        targetNetwork[idx].restore_model(targetSess[idx], checkpoint_path)
    T = 0
    q = Queue()
    q.put(T)
    lock = Lock()
    threads = [Thread(target=async_Q, args=(max_time_steps, reward, penalty, asyncUpdate, globalUpdate,
                                                                        policyNetwork, policySess, targetNetwork, targetSess,
                                                                        lock, q)) for _ in range(4)]
    #map(lambda t: t.start(), threads)
    for t in threads:
        t.start()

    print(threads)
    print("main complete")

if __name__ == '__main__':
    mainAlgorithm(max_time_steps=100)
