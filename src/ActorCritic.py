import numpy as np
import os
import shutil

from Agent import *
from Action import Action
from Point import Point
from Game import Game
from Constants import *

def getFeatureVector(state, action):
    featureVector = []  # s*a, s^2*a^2
    actionValue = action.value + 1
    for feature in state:
        # feature is already normalized. Multiplying with actionValue/4 ensures it stays normalized
        featureVector.append(feature * actionValue / 4)
        featureVector.append(feature**2 * actionValue**2 / 16)
    return np.asarray(featureVector)

def getNumericalPreferences(snake, state, theta):
    numericalPreferenceList = []
    for action in snake.permissible_actions():
        featureVector = getFeatureVector(state, action)
        numericalPreference = np.dot(theta.T, featureVector)
        numericalPreferenceList.append(numericalPreference)

    return numericalPreferenceList

def getPolicy(snake, state, theta, action):
    featureVector = getFeatureVector(state, action)
    numericalPreference = np.dot(theta.T, featureVector)    # h(s, a, theta)
    numericalPreferenceList = getNumericalPreferences(snake, state, theta)

    return (np.exp(numericalPreference) / np.sum( np.exp(numericalPreferenceList) ))    # e^h(s, a, theta)/ Sum over b(e^h(s, b, theta))

def getValueFunction(state, w):
    if np.all(np.asarray(state) == -1):
        return 0

    featureVector = np.asarray(state)
    return np.dot(w.T, featureVector)

def getAction(snake, state, theta):
    actionProbability = []
    actions = []
    for action in snake.permissible_actions():
        actionProbability.append(getPolicy(snake, state, theta, action))
        actions.append(action)
    return Action(np.random.choice(actions, p=actionProbability))

def getGradientForPolicy(snake, state, action, theta):
    featureVector = getFeatureVector(state, action)
    exps = np.exp(getNumericalPreferences(snake, state, theta))
    feature_exps = np.asarray([ getFeatureVector(state, action) * exps[i] for i, action in enumerate(snake.permissible_actions()) ] )
    numr = np.sum(feature_exps, axis=0)
    denr = np.sum(exps)
    return featureVector - (numr / denr)

def train(gridSize, relative, multipleAgents, k, alphaTheta, alphaW, gamma, maxTimeSteps,
                                        checkpointFrequency=500, checkpoint_dir="checkpoints", load=False, load_dir="checkpoints", load_time_step=500):
    length = getStateLength(multipleAgents)
    theta = np.zeros((numberOfSnakes, length * 2))
    w = np.zeros((numberOfSnakes, length))

    if load: # resume training from old checkpoints
        w = np.load("{}/w_{}.npy".format(load_dir, load_time_step))
        theta = np.load("{}/theta_{}.npy".format(load_dir, load_time_step))

    if os.path.isdir(checkpoint_dir):
        # if directory exists, delete it and its contents
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(checkpoint_dir)

    timeSteps = 0
    counter = 0
    while timeSteps <= maxTimeSteps:
        g = Game(numberOfSnakes, gridSize, globalEpisodeLength)
        episodeRunning = True

        while episodeRunning:
            I = 1
            actionList = []
            stateList = []
            for i, snake in enumerate(g.snakes):
                if not snake.alive:
                    actionList.append(None)
                    stateList.append([-1] * getStateLength(multipleAgents))
                    continue
                opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
                stateList.append(getState(snake, opponentSnakes, gridSize, relative, multipleAgents, g.food, k, normalize=True))
                action = getAction(snake, stateList[i], theta[i])
                actionList.append(action)

            singleStepRewards, episodeRunning = g.move(actionList)
            timeSteps += 1
            print "t = " + str(timeSteps)

            if timeSteps % checkpointFrequency == 0:
                np.save("{}/theta_{}.npy".format(checkpoint_dir, timeSteps), theta)
                np.save("{}/w_{}.npy".format(checkpoint_dir, timeSteps), w)

            for i, snake in enumerate(g.snakes):
                if not snake.alive:
                    continue
                opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
                state = stateList[i]
                action = actionList[i]
                nextState = getState(snake, opponentSnakes, gridSize, relative, multipleAgents, g.food, k, normalize=True)
                reward = singleStepRewards[i]
                delta = reward + gamma * getValueFunction(nextState, w[i]) - getValueFunction(state, w[i])
                w[i] = np.add(w[i], (alphaW * delta) * np.asarray(state))
                theta[i] += alphaTheta * I * delta * getGradientForPolicy(snake, state, action, theta[i])
                I *= gamma

            if timeSteps > maxTimeSteps:
                break

        g.endGame()

    np.save("{}/theta_{}.npy".format(checkpoint_dir, timeSteps), theta)
    np.save("{}/w_{}.npy".format(checkpoint_dir, timeSteps), w)


def inference(gridSize, relative, multipleAgents, k, load_dir="checkpoints", load_time_step=500):
    w = np.load("{}/w_{}.npy".format(load_dir, load_time_step))
    theta = np.load("{}/theta_{}.npy".format(load_dir, load_time_step))
    g = Game(numberOfSnakes, gridSize, globalEpisodeLength)
    episodeRunning = True
    while episodeRunning:
        actionList = []
        for i, snake in enumerate(g.snakes):
            if not snake.alive:
                actionList.append(None)
                stateList.append([-1] * getStateLength(multipleAgents))
                continue
            opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
            state = getState(snake, opponentSnakes, gridSize, relative, multipleAgents, g.food, k, normalize=True)
            action = getAction(snake, state, theta[i])
            actionList.append(action)

        singleStepRewards, episodeRunning = g.move(actionList)
        print g

def graphical_inference(gridSize, relative, multipleAgents, k, load_dir="checkpoints", load_time_step=500, play=False, scalingFactor=9):
    import pygame
    import GraphicsEnv

    numSnakes = numberOfSnakes
    if play:
        numSnakes += 1
    colors = np.random.randint(0, 256, size=[numSnakes, 3])
    if play: # user interacts with the agents
        colors[0] = (0, 0, 0) # player's snake is always black
    win = pygame.display.set_mode((scalingFactor * gridSize, scalingFactor * gridSize))  # Game Window
    screen = pygame.Surface((gridSize+1, gridSize+1))  # Grid Screen
    pygame.display.set_caption("Snake Game")
    crashed = False

    w = np.load("{}/w_{}.npy".format(load_dir, load_time_step))
    theta = np.load("{}/theta_{}.npy".format(load_dir, load_time_step))
    g = Game(numSnakes, gridSize, globalEpisodeLength)
    episodeRunning = True

    while episodeRunning and not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        actionList = []
        if play:
            actionList.append( GraphicsEnv.manual_action(g.snakes[0], event) )
        for i in range(int(play), numSnakes):
            snake = g.snakes[i]
            if not snake.alive:
                actionList.append(None)
                continue
            opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
            state = getState(snake, opponentSnakes, gridSize, relative, multipleAgents, g.food, k, normalize=True)
            action = getAction(snake, state, theta[i - int(play) ])
            actionList.append(action)

        singleStepRewards, episodeRunning = g.move(actionList)
        GraphicsEnv.displayGame(g, win, screen, colors)

if __name__=='__main__':
    graphical_inference(30, False, False, 3, load_dir="old_checkpoints", load_time_step=10000, play=False)
