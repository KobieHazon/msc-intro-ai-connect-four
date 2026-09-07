"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Author: Kobie Hazon

"""

# multiAgents.py
# --------------
# Framework attribution: John DeNero and Dan Klein.
# http://ai.berkeley.edu.

import math
import random

import util
from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


def get_action_child_state(gameState, actions):
    """
    Utility generator used to iterate easily over the child states of a gameState.
    Receives the state to get the child states of and the appropriate actions to do.
    """
    for column_action in actions:
        child_state = gameState.generateSuccessor(gameState.turn, column_action)
        child_state.switch_turn(gameState.turn)
        yield column_action, child_state


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal
        """
        "*** YOUR CODE HERE ***"
        best_col, minimax_value = self.get_minimax_action_value(gameState, self.depth, True)
        return best_col

    def get_max_val(self, gameState, depth):
        """
        Returns the maximum minimax value of a gameState's sons and the column that leads to it.
        Receives the gameState and the depth to calculate the minimax values with.
        """
        valid_actions = gameState.getLegalActions()
        max_value = -math.inf
        max_value_column = random.choice(valid_actions)
        for child_column, child_state in get_action_child_state(gameState, valid_actions):
            child_value = self.get_minimax_action_value(child_state, depth - 1, False)[1]
            if child_value > max_value:
                max_value = child_value
                max_value_column = child_column
        return max_value_column, max_value

    def get_min_val(self, gameState, depth):
        """
        Returns the minimum minimax value of a gameState's sons and the column that leads to it.
        Receives the gameState and the depth to calculate the minimax values with.
        """
        valid_actions = gameState.getLegalActions()
        min_value = math.inf
        min_value_column = random.choice(valid_actions)
        for child_column, child_state in get_action_child_state(gameState, valid_actions):
            child_value = self.get_minimax_action_value(child_state, depth - 1, True)[1]
            if child_value < min_value:
                min_value = child_value
                min_value_column = child_column
        return min_value_column, min_value

    def get_minimax_action_value(self, gameState, depth, is_max_player):
        """
        Returns best action according to minimax algorithm starting from state (gameState), depth to calculate the
        values on (depth) and if we are on a maximum node (we start as a maximum node).
        """
        if depth == 0 or gameState.is_terminal():
            return None, self.evaluationFunction(gameState)

        if is_max_player:
            return self.get_max_val(gameState, depth)
        else:
            return self.get_min_val(gameState, depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Your minimax agent with alpha-beta pruning (question 2)
        """
        "*** YOUR CODE HERE ***"
        best_col, minimax_value = self.get_alphabeta_minimax_action_value(
            gameState, self.depth, -math.inf, math.inf, True
        )
        return best_col

    def get_max_val(self, gameState, depth, alpha, beta):
        """
        Returns the maximum minimax value of a gameState's sons and the column that leads to it.
        Receives the gameState and the depth to calculate the minimax values with.
        This function uses the parameters alpha, beta to prune branches that aren't necessary.
        """
        valid_actions = gameState.getLegalActions()
        max_value = -math.inf
        max_value_column = random.choice(valid_actions)
        for child_column, child_state in get_action_child_state(gameState, valid_actions):
            child_value = self.get_alphabeta_minimax_action_value(
                child_state, depth - 1, alpha, beta, False
            )[1]
            if child_value > max_value:
                max_value = child_value
                max_value_column = child_column
            if max_value > beta:
                return max_value_column, max_value
            alpha = max(alpha, max_value)
        return max_value_column, max_value

    def get_min_val(self, gameState, depth, alpha, beta):
        """
        Returns the minimum minimax value of a gameState's sons and the column that leads to it.
        Receives the gameState and the depth to calculate the minimax values with.
        This function uses the parameters alpha, beta to prune branches that aren't necessary.
        """
        valid_actions = gameState.getLegalActions()
        min_value = math.inf
        min_value_column = random.choice(valid_actions)
        for child_column, child_state in get_action_child_state(gameState, valid_actions):
            child_value = self.get_alphabeta_minimax_action_value(
                child_state, depth - 1, alpha, beta, True
            )[1]
            if child_value < min_value:
                min_value = child_value
                min_value_column = child_column
            if min_value < alpha:
                return min_value_column, min_value
            beta = min(beta, min_value)
        return min_value_column, min_value

    def get_alphabeta_minimax_action_value(self, gameState, depth, alpha, beta, is_max_player):
        """
        Returns best action according to minimax algorithm starting from state (gameState), depth to calculate the
        values on (depth) and if we are on a maximum node (we start as a maximum node).
        Utilizes parameters alpha, beta passed to helper functions to prune unnecessary branches.
        """
        if depth == 0 or gameState.is_terminal():
            return None, self.evaluationFunction(gameState)

        if is_max_player:
            return self.get_max_val(gameState, depth, alpha, beta)
        else:
            return self.get_min_val(gameState, depth, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_col, expectimax_val = self.get_expectimax_action_value(gameState, self.depth, True)
        return best_col

    def get_max_val(self, gameState, depth):
        """
        Returns the maximum minimax value of a gameState's sons and the column that leads to it.
        Receives the gameState and the depth to calculate the minimax values with.
        As part of the expectimax algorithm, it calls the expectimax function for its sons.
        """
        valid_actions = gameState.getLegalActions()
        max_value = -math.inf
        max_value_column = random.choice(valid_actions)
        for child_column, child_state in get_action_child_state(gameState, valid_actions):
            child_value = self.get_expectimax_action_value(child_state, depth - 1, False)[1]
            if child_value > max_value:
                max_value = child_value
                max_value_column = child_column
        return max_value_column, max_value

    def get_exp_val(self, gameState, depth):
        """
        Returns the expectimax value of a gameState's sons and the column that leads to it.
        Receives the gameState and the depth to calculate the expectimax values with.
        It assumes uniform distribution on all the sons probabilities.
        """
        valid_actions = gameState.getLegalActions()
        exp_value = 0
        child_state_probability = 1 / len(valid_actions)
        for _child_column, child_state in get_action_child_state(gameState, valid_actions):
            exp_value += (
                child_state_probability
                * self.get_expectimax_action_value(child_state, depth - 1, True)[1]
            )
        return None, exp_value

    def get_expectimax_action_value(self, gameState, depth, is_max_player):
        """
        Returns best action according to expectimax algorithm starting from state (gameState), depth to calculate the
        values on (depth) and if we are on a maximum node (we start as a maximum node).
        The returned action may not actually be the optimal.
        """
        if depth == 0 or gameState.is_terminal():
            return None, self.evaluationFunction(gameState)

        if is_max_player:
            return self.get_max_val(gameState, depth)
        else:
            return self.get_exp_val(gameState, depth)
