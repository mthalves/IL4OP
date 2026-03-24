from isaaclab_experiments.src.planning_algorithms.node import CANode, CONode
from isaaclab_experiments.src.planning_algorithms.node import find_new_CPO_root, particle_revigoration

import random

class POMCPDPW(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it
        
        self.discount_factor        = kwargs.get('discount_factor',0.95) # discount factor (historical weight)
        self.particle_revigoration  = kwargs.get('particle_revigoration',True) # enable particle revigoration (silver2010pomcp)
        self.k                      = kwargs.get('k', 100) # particle filter size

        ###
        # Progressive Widening parameters
        ###
        # - action widening
        self.ka = kwargs.get('ka', 0.5) 
        self.alpha_a = kwargs.get('alpha_a', 0.5)

        # - state/observation widening
        self.ko = kwargs.get('ko', 0.5)        
        self.alpha_o = kwargs.get('alpha_o', 0.5)
        
    def simulate_action(self, node, action):
        # 1. Acting
        next_state, reward, _, _ = node.state.step(action)
        next_node = CANode(action, next_state, node.depth+1, node)

        # 2. Returning the next node and the reward
        observation = next_node.state.get_observation()
        return next_node, observation, reward

    def rollout_policy(self,node):
        return node.state.sample_new_action()

    def rollout(self,node,problem):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Choosing an action
        action = self.rollout_policy(node)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        node.depth += 2

        # 4. Rolling out
        R = reward + (self.discount_factor * self.rollout(node, problem))
        return R

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return CONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.is_final_state()

    def simulate(self, node, problem):
        # 1. Adding state to particle filter and checking the stop conditions
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Selecting action and simulating it
        action = node.action_prog_widen(mode='max',
                        coef={'ka':self.ka,'alpha_a':self.alpha_a, 'c':0.5}) 
        (action_node, observation, reward) = self.simulate_action(node, action)

        # - checking if this action is already tried
        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break

        # 3. If the observation widenning criteria is met, do it
        if len(action_node.children) <= self.ko*action_node.visits**self.alpha_o:
            # - adding the action child on the tree
            if action_node.action not in [c.action for c in node.children]:
                node.children.append(action_node)
            
            # - checking if the observation is in the tree
            observation_node = None
            for child in action_node.children:
                if action_node.state.observation_is_equal(child.observation, observation):
                    observation_node = child
                    observation_node.state = action_node.state.copy()
                    break
            
            if observation_node is None:
                observation_node = action_node.add_child(observation)

            # - adding current state to the particle filter/belief set
            observation_node.particle_filter.append(observation_node.state)
            
            # - expanding the tree
            if observation_node.visits == 0:
                future_reward = self.rollout(observation_node, problem)
                observation_node.visits += 1
            else:
                future_reward = self.simulate(observation_node, problem)

        # - otherwise, sample an existing observation and state for simulation
        else:
            observation_node = action_node.sample_child()
            if observation_node is None:
                observation_node = action_node.add_child(observation)

            if observation_node.particle_filter is not None and \
               len(observation_node.particle_filter) > 0:
                observation_node.state = random.sample(
                    observation_node.particle_filter,1)[0]

            future_reward = self.simulate(observation_node, problem)

        # - node update
        node.visits += 1
        action_node.visits += 1
        R = reward + (self.discount_factor * future_reward)
        node.update(action, R)
        return R

    def search(self, root, problem):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        while it < self.max_it:
            
            # a. Sampling the belief state for simulation
            if len(root.particle_filter) == 0:
                beliefState = problem.sample_state(root.state)
            else:
                beliefState = random.sample(root.particle_filter,1)[0]
            root.state = beliefState

            # b. simulating
            self.simulate(root, problem)
            it += 1
            
        return root.get_best_action()

    def plan(self, agent, problem):
        # 1. Getting the current state and previous action-observation pair
        state = problem.get_current_state(agent['pos'])
        observation = state.get_observation()
        previous_action = None if len(agent['action_history']) == 0 \
                            else agent['action_history'][-1]

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = CONode(observation=observation,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            self.root = find_new_CPO_root(
                state, previous_action, observation, self.root
            )

        # 3. Performing particle revigoration
        if self.particle_revigoration:
            particle_revigoration(state, problem, self.root, self.k)

        # 4. Searching for the best action within the tree
        best_action = self.search(self.root, problem)
        self.root.show_qtable()
        return [best_action]