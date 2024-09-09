# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        i = 0
        while i < self.iterations:
            new_values = util.Counter()
            for state in states:
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    max_val = float("-inf")
                    for action in actions:
                        sum = self.computeQValueFromValues(state, action)
                        max_val = max(sum, max_val)
                    new_values[state] = max_val
            i += 1
            for s in states:
                self.values[s] = new_values[s]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        q_value = 0
        result = self.mdp.getTransitionStatesAndProbs(state, action)
        for res in result:
            reward = self.mdp.getReward(state, action, res[0])
            q_value += res[1] * (reward + self.discount * self.values[res[0]])
        return q_value

    def computeActionFromValues(self, state):
        actions = self.mdp.getPossibleActions(state)
        max_value = float('-inf')
        best_action = None
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        i = 0
        states_num = len(states)
        while i < self.iterations:
            state = states[i % states_num]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                max_val = float("-inf")
                for action in actions:
                    sum_p = self.computeQValueFromValues(state, action)
                    max_val = max(sum_p, max_val)
                self.values[state] = max_val
            i += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        states = self.mdp.getStates()
        predecessors = {}
        priority_queue = util.PriorityQueue()

        for state in states: predecessors[state] = set()
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    max_value = float('-inf')
                    for nextState, prob in transitions: predecessors[nextState].add(state)
                    q_value = self.computeQValueFromValues(state, action)
                    max_value = max(q_value, max_value)
                diff = abs(max_value - self.values[state])
                priority_queue.push(state, - diff)
        i = 0
        while i < self.iterations:
            if priority_queue.isEmpty():
                break
            state = priority_queue.pop()
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                max_val = float("-inf")
                for action in actions:
                    q_value = self.computeQValueFromValues(state, action)
                    max_val = max(q_value, max_val)
                self.values[state] = max_val
                for p in predecessors[state]:
                    if not self.mdp.isTerminal(p):
                        max_value = float('-inf')
                        actions = self.mdp.getPossibleActions(p)
                        for action in actions:
                            q_value = self.computeQValueFromValues(p, action)
                            max_value = max(q_value, max_value)
                        diff = abs(max_value - self.values[p])
                        if diff > self.theta:
                            priority_queue.update(p, -diff)
            i += 1
