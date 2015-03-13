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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in
                newGhostStates]

        "*** YOUR CODE HERE ***"
        curPos = currentGameState.getPacmanPosition()
        score = 0
        if curPos == newPos:
            score -= 100
        if len(newFood.asList()) != 0:
            score += 10 * max([1.0 / manhattanDistance(newPos, x) for x in
                newFood.asList()])
        score += sum([manhattanDistance(newPos, x.getPosition()) for x in
            newGhostStates])
        return score + successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        class treeNode:
            def __init__(self, state = None, father = None, action = None,
                    targetdep = 0, evalfn = None):
                self.state = state
                self.father = father
                self.action = action
                self.value = 0
                self.evalfn = evalfn
                self.num = state.getNumAgents()
                self.targetdep = targetdep
                if self.father != None:
                    self.agentIndex = (father.agentIndex + 1) % self.num
                    if self.agentIndex == 0:
                        self.dep = father.dep + 1
                    else:
                        self.dep = father.dep
                else:
                    self.agentIndex = 0
                    self.dep = 0
                self.child = []
                if ((not self.state.isWin()) and (not self.state.isLose()) and
                    self.dep != self.targetdep):
                        acts = self.state.getLegalActions(self.agentIndex)
                        for act in acts:
                            t = treeNode(self.state.generateSuccessor(
                                self.agentIndex, act), self, act, 
                                self.targetdep, evalfn)
                            self.child.append(t)
                        
            def updatevalue(self):
                if len(self.child) == 0:
                    self.value = self.evalfn(self.state)
                    return (self.value, self.action)
                else:
                    if self.agentIndex == 0:
                        v = max([c.updatevalue()[0] for c in self.child])
                        self.value = v
                        return (v, self.action)
                    else:
                        v = min([c.updatevalue()[0] for c in self.child])
                        self.value = v
                        return (v, self.action)

        mtree = treeNode(state = gameState, targetdep = self.depth, evalfn =
                self.evaluationFunction)
        mtree.updatevalue()
        return max([(c.value, c.action) for c in mtree.child], key = lambda
                x:x[0])[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        class treeNode:
            def __init__(self, state = None, father = None, action = None,
                    targetdep = 0, evalfn = None):
                self.state = state
                self.father = father
                self.action = action
                self.value = 0
                self.evalfn = evalfn
                self.num = state.getNumAgents()
                self.targetdep = targetdep
                if self.father != None:
                    self.agentIndex = (father.agentIndex + 1) % self.num
                    if self.agentIndex == 0:
                        self.dep = father.dep + 1
                    else:
                        self.dep = father.dep
                else:
                    self.agentIndex = 0
                    self.dep = 0
                        
            def updatevalue(self, alpha = -float('Inf'), beta = float('Inf')):
                if (self.state.isWin() or self.state.isLose() or self.dep ==
                    self.targetdep):
                    self.value = self.evalfn(self.state)
                    return (self.value, self.action)
                else:
                    if self.agentIndex == 0:
                        v = -float('Inf')
                        togo = None
                        acts = self.state.getLegalActions(self.agentIndex)
                        for act in acts:
                            t = treeNode(self.state.generateSuccessor(
                                self.agentIndex, act), self, act, 
                                self.targetdep, self.evalfn)
                            (v, togo) = max([(v, togo), 
                                t.updatevalue(alpha, beta)], key = lambda
                                x:x[0])
                            if v > beta:
                                self.value = v
                                if self.dep == 0:
                                    return (v, togo)
                                else:
                                    return (v, self.action)
                            alpha = max(alpha, v)
                        self.value = v
                        if self.dep == 0:
                            return (v, togo)
                        else:
                            return (v, self.action)
                    else:
                        v = float('Inf')
                        togo = None
                        acts = self.state.getLegalActions(self.agentIndex)
                        for act in acts:
                            t = treeNode(self.state.generateSuccessor(
                                self.agentIndex, act), self, act, 
                                self.targetdep, self.evalfn)
                            (v, togo) = min([(v, togo), 
                                t.updatevalue(alpha, beta)], key = lambda
                                x:x[0])
                            if v < alpha:
                                self.value = v
                                return (v, self.action)
                            beta = min(beta, v)
                        self.value = v
                        return (v, self.action)
 

        mtree = treeNode(state = gameState, targetdep = self.depth, evalfn =
                self.evaluationFunction)
        return mtree.updatevalue()[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        class treeNode:
            def __init__(self, state = None, father = None, action = None,
                    targetdep = 0, evalfn = None):
                self.state = state
                self.father = father
                self.action = action
                self.value = 0
                self.evalfn = evalfn
                self.num = state.getNumAgents()
                self.targetdep = targetdep
                if self.father != None:
                    self.agentIndex = (father.agentIndex + 1) % self.num
                    if self.agentIndex == 0:
                        self.dep = father.dep + 1
                    else:
                        self.dep = father.dep
                else:
                    self.agentIndex = 0
                    self.dep = 0
                self.child = []
                if ((not self.state.isWin()) and (not self.state.isLose()) and
                    self.dep != self.targetdep):
                        acts = self.state.getLegalActions(self.agentIndex)
                        for act in acts:
                            t = treeNode(self.state.generateSuccessor(
                                self.agentIndex, act), self, act, 
                                self.targetdep, evalfn)
                            self.child.append(t)
                        
            def updatevalue(self):
                if len(self.child) == 0:
                    self.value = self.evalfn(self.state)
                    return (self.value, self.action)
                else:
                    if self.agentIndex == 0:
                        v = max([c.updatevalue()[0] for c in self.child])
                        self.value = v
                        return (v, self.action)
                    else:
                        v = 1.0 * sum([c.updatevalue()[0] for c in
                            self.child]) / len(self.child)
                        self.value = v
                        return (v, self.action)

        mtree = treeNode(state = gameState, targetdep = self.depth, evalfn =
                self.evaluationFunction)
        mtree.updatevalue()
        return max([(c.value, c.action) for c in mtree.child], key = lambda
                x:x[0])[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''
    successorGameState = [currentGameState.generatePacmanSuccessor(action) for
            action in actions]
    newPos = successorGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in
            newGhostStates]
    '''
    curPos = currentGameState.getPacmanPosition()
    curGhostStates = currentGameState.getGhostStates()
    score = 0
    curFood = currentGameState.getFood()
    if len(curFood.asList()) != 0:
        temp = [1.0 / manhattanDistance(curPos, x) for x in curFood.asList()]
        temp.sort()
        score += 10 * sum(temp[-1:])
    score += sum([manhattanDistance(curPos, x.getPosition()) for x in
        curGhostStates])
    score -= len(currentGameState.getLegalActions())
    return score + currentGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

