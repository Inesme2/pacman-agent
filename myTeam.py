# Authors: Iker Benito (u217895), Inés Montoya (u235592)
# This project forms part of the Berkley Pacman "Capture the Flag" contest.

# COMMENTS:
# The only thing that must be improved is that one the Pacman eats a
# capsule, and improve the offensive strategy by implementing a 
# Minimax algorithm.

# Import the required libraries
import random
import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    # Initialize the agent
    def __init__(self, index, time_for_computing=.1):
        # Calls the base class constructor to set up the agent
        super().__init__(index, time_for_computing)
        # Placeholder for the start position, to be set in register_initial_state
        self.start = None

    # Called at the beginning of each game to set the agent's starting position
    def register_initial_state(self, game_state):
        # Record the agent's starting position
        self.start = game_state.get_agent_position(self.index)
        # Calls the base class version of this function as well
        CaptureAgent.register_initial_state(self, game_state)

    # Decides on the best action from the possible legal actions
    def choose_action(self, game_state):
        # Get all legal actions for the agent
        actions = game_state.get_legal_actions(self.index)
        # Evaluate the worth of each action
        values = [self.evaluate(game_state, a) for a in actions]
        # Find the maximum value among all action values
        max_value = max(values)
        # Filter out the best actions that have the max value
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        # Check if there is only a small amount of food left, which may trigger special behavior
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            # If there are less than 2 foods left, prioritize going home
            best_action = self.go_home(game_state, actions)
            return best_action
        # If there are no special conditions, return one of the best actions
        return random.choice(best_actions)

    # A method to direct the agent back to its start position
    def go_home(self, game_state, actions):
        # Initialize the shortest distance to a very high value
        best_dist = float('inf')
        # Placeholder for the best action that leads home
        best_action = None
        # Check the distance to start position from the successor of each action
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.start, pos2)
            # Update the best action if this one is closer to the start
            if dist < best_dist:
                best_action = action
                best_dist = dist
        # Return the action that gets the agent closest to home
        return best_action

    # Computes the next state given an action
    def get_successor(self, game_state, action):
        # Generate the successor game state after taking the action
        successor = game_state.generate_successor(self.index, action)
        # Get the new position from the state
        pos = successor.get_agent_state(self.index).get_position()
        # Check if the position is halfway between two grid points
        if pos != nearestPoint(pos):
            # If so, return the successor of taking the same action again
            # This makes sure the agent moves a full grid square
            return successor.generate_successor(self.index, action)
        else:
            # Otherwise, return the successor as is
            return successor

    # Evaluates the value of an action by considering various features
    def evaluate(self, game_state, action):
        # Extract features for the given state and action
        features = self.get_features(game_state, action)
        # Get the weights for each feature
        weights = self.get_weights(game_state, action)
        # Compute the dot product of features and weights
        return features * weights

    # Abstract method to get features
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    # Abstract method to get feature weights
    def get_weights(self, game_state, action):
        # Return a dictionary of feature names to weights
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    # Initialization function, sets up necessary variables for the agent
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)  # Initialize the base class
        self.lastAction = None  # Stores the last action taken by the agent
        self.lastPositions = []  # Stores a history of the agent's previous positions

    # Chooses the best action to take in the current game state
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)  # Get all legal actions for the agent
        actions.remove(Directions.STOP)  # Remove 'Stop' from the possible actions to keep moving

        # Avoid reversing the last action if possible to prevent oscillating
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if reverse in actions and len(actions) > 1:
            actions.remove(reverse)

        future_positions = []  # Stores future positions associated with the possible actions
        best_action = None  # Best action to take
        best_score = float('-inf')  # Initialize best score to negative infinity

        # Evaluate each action and choose the one with the highest score
        for action in actions:
            successor = self.get_successor(game_state, action)  # Get the successor state from taking action
            score = self.evaluate(game_state, action)  # Evaluate the action's score
            future_position = successor.get_agent_state(self.index).get_position()
            
            # Discourage staying in the same small area by penalizing repeated positions
            if len(self.lastPositions) > 3 and future_position in self.lastPositions[-3:]:
                score -= 100  # Penalize returning to recent positions to encourage exploration

            # Update the best action if this action's score is better and the position is new
            if score > best_score and future_position not in future_positions:
                best_score = score
                best_action = action
                future_positions.append(future_position)

        # If all actions were bad (e.g., trapped), pick the least bad or allow reversal
        if not best_action:
            best_action = reverse if reverse in actions else random.choice(actions)

        # Update the history of positions to include the current position
        self.lastPositions.append(game_state.get_agent_state(self.index).get_position())
        if len(self.lastPositions) > 6:  # Limit the history to the last 6 positions
            self.lastPositions.pop(0)

        return best_action  # Return the chosen action

    # Computes features used for evaluating actions
    def get_features(self, game_state, action):
        features = super().get_features(game_state, action)  # Get base features
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        
        # Compute distance to the nearest food for the successor state
        food_list = self.get_food(successor).as_list()
        features['food_remain'] = len(food_list)  # Feature for the number of remaining food
        if food_list:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance  # Feature for distance to the nearest food
        
        return features  # Return the computed features

    # Returns the weights for each feature used in the evaluation
    def get_weights(self, game_state, action):
        # Define the weights for each feature
        return {
            'successor_score': 100,  # Weight for the game score
            'distance_to_food': -2,  # Make reaching food a high priority
            'food_remain': -100,     # Prioritize reducing the count of remaining food
            # Additional weights can be added here
        }

class DefensiveReflexAgent(ReflexCaptureAgent):
    # This method selects an action for the agent to execute
    def choose_action(self, game_state):
        # Obtain all legal actions for the current state
        actions = game_state.get_legal_actions(self.index)
        # Evaluate the value of each action
        values = [self.evaluate(game_state, a) for a in actions]
        # Find the maximum value among all evaluated actions
        max_value = max(values)
        # Compile a list of actions that yield the maximum value
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        # Randomly choose among the best actions
        return random.choice(best_actions)
    
    # Generates a set of features used for evaluating actions
    def get_features(self, game_state, action):
        features = util.Counter()  # Initialize feature vector as a counter
        successor = self.get_successor(game_state, action)  # Get the game state after the action is taken
        my_state = successor.get_agent_state(self.index)  # Get the agent's state after the action is taken
        my_pos = my_state.get_position()  # Get the agent's position after the action is taken

        # Adjust the position to ensure it's within the legal playing grid
        my_pos = self._ensure_grid_position(my_pos, game_state)

        # Determine if the agent is on defense (not a Pacman)
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Compute the distance to invading Pacmen
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]  # Get enemy agent states
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]  # Filter for invaders visible
        features['num_invaders'] = len(invaders)  # Count visible invaders
        if len(invaders) > 0:  # If there are invaders
            dists = [self.get_maze_distance(my_pos, self._ensure_grid_position(a.get_position(), game_state)) for a in invaders]
            features['invader_distance'] = min(dists)  # Store the minimum distance to an invader

        # Punish the agent for stopping and for reversing direction
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features  # Return the compiled features

    # Assigns weights to each feature
    def get_weights(self, game_state, action):
        # Set the weights for each feature
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    # Ensures the agent's position is within the game grid
    def _ensure_grid_position(self, position, game_state):
        grid_width = game_state.get_walls().width  # Width of the game grid
        grid_height = game_state.get_walls().height  # Height of the game grid
        x, y = position  # Unpack the position tuple
        # Adjust the position to ensure it's within bounds
        x = min(max(1, int(x)), grid_width - 1)
        y = min(max(1, int(y)), grid_height - 1)
        return (x, y)  # Return the adjusted position
