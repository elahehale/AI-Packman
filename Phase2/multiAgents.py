# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        result = 0
        foodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if len(foodDistances) > 0:
            result += float(9 / min(foodDistances))
        index = 0
        scared_indexes = [i for i in newScaredTimes if i > 1]
        minimum_scared_ghost_distance = 10000
        for ghost in newGhostStates:
            distance = abs(newPos[0] - ghost.getPosition()[0]) + abs(newPos[1] - ghost.getPosition()[1])
            if index in scared_indexes and distance < minimum_scared_ghost_distance:
                minimum_scared_ghost_distance = distance
            elif distance > 1:
                if distance > 3:
                    result += 3
                else:
                    result += distance
            elif distance == 1:
                return -2000
            index += 1
        if len(scared_indexes) > 0:
            result += float(1 / minimum_scared_ghost_distance)
        return result + successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):

        ghosts_num = gameState.getNumAgents()

        def maximumValue(gameState, depth):
            max_value = float('-inf')
            depth = depth - 1
            legal_actions = gameState.getLegalActions(0)
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), 'stop'
            for action in legal_actions:
                value = minimumValue(gameState.generateSuccessor(0, action), depth, 1)
                if value > max_value:
                    max_value = value
                    next_action = action

            return max_value, next_action

        def minimumValue(gameState, depth, ghost_index):
            min_value = float('inf')
            legal_actions = gameState.getLegalActions(ghost_index)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in legal_actions:
                if ghost_index < (ghosts_num - 1):
                    min_value = min(min_value,
                                    minimumValue(gameState.generateSuccessor(ghost_index, action), depth, ghost_index+1))
                else:
                    min_value = min(min_value, maximumValue(gameState.generateSuccessor(ghost_index, action), depth)[0])
            return min_value

        return maximumValue(gameState, self.depth + 1)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        ghosts_num = gameState.getNumAgents()

        def maximumValue(gameState, depth, alpha, beta):
            max_value = float('-inf')
            depth = depth - 1
            legal_actions = gameState.getLegalActions(0)
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), 'stop'
            for action in legal_actions:
                value = minimumValue(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
                if value > max_value:
                    max_value = value
                    next_action = action
                if value> beta: return value,next_action
                alpha = max(alpha, max_value)

            return max_value, next_action

        def minimumValue(gameState, depth, ghost_index , alpha, beta):
            min_value = float('inf')
            legal_actions = gameState.getLegalActions(ghost_index)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in legal_actions:
                if ghost_index < (ghosts_num - 1):
                    min_value = min(min_value,
                                    minimumValue(gameState.generateSuccessor(ghost_index, action), depth,
                                                 ghost_index + 1 , alpha, beta))
                else:
                    min_value = min(min_value, maximumValue(gameState.generateSuccessor(ghost_index, action), depth, alpha, beta)[0])
                if min_value < alpha: return min_value
                beta = min(beta, min_value)
            return min_value

        return maximumValue(gameState, self.depth + 1, float('-inf'), float('inf') )[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):

        ghosts_num = gameState.getNumAgents()

        def maximumValue(gameState, depth):
            max_value = float('-inf')
            depth = depth - 1
            legal_actions = gameState.getLegalActions(0)
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), 'stop'
            for action in legal_actions:
                value = chanceValue(gameState.generateSuccessor(0, action), depth, 1)
                if value > max_value:
                    max_value = value
                    next_action = action

            return max_value, next_action

        def chanceValue(gameState, depth, ghost_index):
            sum =0
            legal_actions = gameState.getLegalActions(ghost_index)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in legal_actions:
                if ghost_index < (ghosts_num - 1):
                    sum = sum + chanceValue(gameState.generateSuccessor(ghost_index, action), depth, ghost_index+1)
                else:
                    sum = sum +  maximumValue(gameState.generateSuccessor(ghost_index, action), depth)[0]
            return float(sum/len(legal_actions))

        return maximumValue(gameState, self.depth + 1)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()
    result = 0
    foodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(foodDistances) > 0 : result -=1/100* (max(foodDistances) + min(foodDistances))
    index = 0
    scared_indexes = [i for i in newScaredTimes if i > 0]
    minimum_scared_ghost_distance = 10000
    for ghost in newGhostStates:
        distance = abs(newPos[0] - ghost.getPosition()[0]) + abs(newPos[1] - ghost.getPosition()[1])
        if index in scared_indexes and distance < minimum_scared_ghost_distance:
            minimum_scared_ghost_distance = distance
        elif distance > 1:
            if distance > 3:
                result += 3
            else:
                result += distance
        elif distance == 1:
            return -2000
        index += 1
    if len(scared_indexes) > 0:
        result += float(1 / minimum_scared_ghost_distance)
    return result + currentGameState.getScore()- 20 * len(capsules)


# Abbreviation
better = betterEvaluationFunction
