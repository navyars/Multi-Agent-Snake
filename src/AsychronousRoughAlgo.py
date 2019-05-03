import tensorflow as tf
import Agent
import FunctionApproximator
import Game
from  Constants import *
import numpy as np
import random


def epsilon_greedy_action(s, sess, nn, input_data, ep):
        possible_actions = s.permissible_actions()
        action_dict = {}
        for act in possible_actions:
            input_action = [[act]]
            action_dict[act] = nn.Q(sess, input_data, input_action)[0][0]
        best_action = max(action_dict, key=action_dict.get)
        possible_actions.remove(best_action)
        prob = random.uniform(0, 1)
        if prob <= ep:
            return best_action
        else:
            return random.choice(possible_actions)
        #choose action according to epsilon-greedy
        #return the action to be chosen


def best_q(s, sess, nn, input_data):
    q = []
    possible_actions = s.permissible_actions()
    for act in possible_actions:
        input_action = [[act]]
        q.append(nn.Q(sess, input_data, input_action)[0][0])
    return max(q)


def getReward(s, r, p):
    if s.didEatFood():
        return r
    elif s.didHitWall() or s.didHitSnake():
        return p
    else:
        return 0


def mainAlgorithm (max_time_steps=1000, reward=1, penalty=-10, asyncUpdate=30, globalUpdate=120):
    nn1 = []
    nn2 = []
    sess1 = []
    sess2 = []
    epsilon = []
    done = False
    #Initializing the 2*n neural nets
    for idx in range(numberOfSnakes):
        nn1.append(FunctionApproximator.NeuralNetwork(3)) #Global neural network
        nn2.append(FunctionApproximator.NeuralNetwork(3))  #Local neural net
        sess1.append(tf.Session(graph=nn1[idx].graph))
        sess2.append(tf.Session(graph=nn2[idx].graph))
        nn1[idx].init(sess1[idx])
        nn2[idx].init(sess2[idx])
        epsilon.append(np.random.normal(0.8, 0.1))

    time_steps = 0

    while done == False:
        g = Game.Game(numberOfSnakes, gridSize, globalEpisodeLength)
        #Start a Game
        snake_list = g.snakes
        flag = 0
        pastStateAlive = [True for i in range(numberOfSnakes)]
        actions_taken = [0 for j in range(numberOfSnakes)]
        while flag == 0: #Meaning in an episode
            for idx in range(numberOfSnakes):
                pruned_snake_list = snake_list
                del pruned_snake_list[idx]
                #initial_state[idx] = Agent.getRelativeStateForMultipleAgents(g.snakes[idx],pruned_snake_list)
                initial_state[idx] = Agent.getRelativeStateForSingleAgent(g.snakes[idx]) #Can either be this or the above line
                if g.snakes[idx].alive:
                    actions_taken[idx] = epsilon_greedy_action(g.snakes[idx], sess1[idx], nn1[idx],initial_state[idx], epsilon[idx])
                    pastStateAlive[idx] = True
                else:
                    actions_taken[idx] = -1
                    pastStateAlive[idx] = False
            
            flag = g.Move(actions_taken)                
            #Now we transition to the next state
            time_steps += 1
    
            for idx in range(numberOfSnakes):
                if (pastStateAlive[idx]): # To check if snake was already dead or just died
                    pruned_snake_list = snake_list
                    del pruned_snake_list[idx]
                    # final_state[idx] = Agent.getRelativeStateForMultipleAgents(g.snakes[idx],pruned_snake_list)
                    final_state[idx] = Agent.getRelativeStateForSingleAgent(g.snakes[idx])
                    reward[idx] = getReward(g.snakes[idx], reward, penalty)
                    if flag == 1 or not(g.snakes[idx].Alive): #Training is done on the snake only on terminal state
                        update_g = \
                        nn1[idx].update_gradient(sess1[idx], final_state[idx], actions_taken[idx], reward[idx], None)[
                            0][0]
                        train_g = nn1[idx].train(sess1[idx], final_state[idx], actions_taken[idx], reward[idx], None)[0][0]
                    else: #Else only an update is done
                        best_q[idx] = best_q(g.snakes[idx], sess2[idx], nn2[idx], final_state[idx])
                        #Can instead use TargetNetwork.max_Q
                        update_g = \
                        nn1[idx].update_gradient(sess1[idx], final_state[idx], actions_taken[idx], reward[idx],
                                                 best_q[idx])[0][0]
                
                if time_steps % asyncUpdate == 0:
                    print("work_in_progress")
                    # Train the model (nn1) with all the update_gradients and resets it
                    #train_g = nn1[idx].train(sess1[idx], final_state[idx], actions_taken[idx], reward[idx], None)[0][0]
                    #reset_g = nn1[idx].reset_accumulator(sess1[idx])

                if time_steps % globalUpdate == 0:
                    print("work_in_progress")
                    #Copy the weights to nn2

    if time_steps > max_time_steps:
        done=True

