import numpy as np
from Agent import *
from Action import Action
from Point import Point
from Game import Game
from Constants import *

def getFeatureVector(state, action):
    featureVector = []  # s*a, s^2*a^2
    actionValue = action.value + 1
    for feature in state:
        featureVector.append(feature * actionValue)
        featureVector.append(feature**2 * actionValue**2)

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
    numr = np.sum([ getFeatureVector(state, action) * exps[i] for i, action in enumerate(snake.permissible_actions()) ])
    denr = np.sum(exps)
    return featureVector - (numr / denr)

def actorCritic(gridSize, relative, multipleAgents, k, alphaTheta, alphaW, gamma, maxTimeSteps):    # TODO: should the game be instantiated here?
    length = getStateLength(multipleAgents = False)
    theta = np.zeros((numberOfSnakes, length * 2))
    w = np.zeros((numberOfSnakes, length))

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
                stateList.append(getState(snake, opponentSnakes, gridSize, relative, multipleAgents, k))
                action = getAction(snake, stateList[i], theta[i])
                actionList.append(action)
            
            singleStepRewards, episodeRunning = g.move(actionList)
            timeSteps += 1

            for i, snake in enumerate(g.snakes):
                if not snake.alive:
                    continue
                opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
                state = stateList[i]
                action = actionList[i]
                nextState = getState(snake, opponentSnakes, gridSize, relative, multipleAgents, k)
                reward = singleStepRewards[i]
                delta = reward + gamma * getValueFunction(nextState, w[i]) - getValueFunction(state, w[i])
                w[i] = np.add(w[i], (alphaW * delta) * np.asarray(state))
                theta[i] += alphaTheta * I * delta * getGradientForPolicy(snake, state, action, theta[i])
                I *= gamma
            
            if timeSteps > maxTimeSteps:
                break

        g.endGame()
        np.save("theta.npy", theta)
        np.save("w.npy", w)

def inference(gridSize, relative, multipleAgents, k):
    w = np.load("w.npy")
    theta = np.load("theta.npy")
    g = Game(numberOfSnakes, gridSize, globalEpisodeLength)
    episodeRunning = True
    while episodeRunning:
        actionList = []
        stateList = []
        for i, snake in enumerate(g.snakes):
            if not snake.alive:
                actionList.append(None)
                stateList.append([-1] * getStateLength(multipleAgents))
                continue
            opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
            stateList.append(getState(snake, opponentSnakes, gridSize, relative, multipleAgents, k))
            action = getAction(snake, stateList[i], theta[i])
            actionList.append(action)
        
        singleStepRewards, episodeRunning = g.move(actionList)