# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = {}

    def getQValue(self, state, action):
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        return 0.0

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        value = float('-inf')
        for action in legalActions:
            new_value = self.getQValue(state, action)
            value = max(value, new_value)
        return value

    def computeActionFromQValues(self, state):
        best_action = []
        legal_actions = self.getLegalActions(state)
        max_value = self.computeValueFromQValues(state)
        best_action = [action for action in legal_actions if self.getQValue(state, action) == max_value]
        return random.choice(best_action)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        new_q = (1 - self.alpha) * self.getQValue(state, action) + \
                self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
        self.q_values[(state, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        q_value = 0.0
        f_vector = self.featExtractor.getFeatures(state, action)
        for feature in f_vector:
            q_value += self.weights[feature] * f_vector[feature]
        return q_value

    def update(self, state, action, nextState, reward):
        difference = (reward + self.discount * self.getValue(nextState)) -self.getQValue(state, action)
        f_vector = self.featExtractor.getFeatures(state, action)
        for feature in f_vector:
            self.weights[feature] += self.alpha * difference * f_vector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
