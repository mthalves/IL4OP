from isaaclab_experiments.src.planning_algorithms.node import RhoANode, RhoONode
from isaaclab_experiments.src.planning_algorithms.node import particle_revigoration

import numpy as np
import random
import time

class TBRhoPOMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.episode = 0
        self.max_depth = max_depth
        self.max_it = max_it
        self.c = 0.5
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        ###
        # POMCP enhancements
        ###
        # particle Revigoration (silver2010pomcp)
        particle_revigoration = kwargs.get('particle_revigoration')
        if particle_revigoration is not None:
            self.pr = particle_revigoration
        else: #default
            self.pr = True

        k = kwargs.get('k') # particle filter size
        self.k = k if k is not None else 100
        
        smallbag_size = kwargs.get('smallbag_size') # smallbag size
        self.smallbag_size = smallbag_size if smallbag_size is not None else 10
        
        time_budget = kwargs.get('time_budget') # time budget in seconds
        self.time_budget = time_budget if time_budget is not None else 2.0
        self.start_time_budget = time.time()
        

    def rhofunction(self,particles,action):
        belief_reward = 0.0

        norm = 0
        if isinstance(particles,dict):
            # calculating belief 
            for key in particles:
                tmp_state = particles[key][0].copy()
                state,reward, _, _ = tmp_state.step(action)
                
                trans_p = (state.get_trans_p(action))[1]
                obs_p = (state.get_obs_p(action))[1]
                belief_reward += reward*trans_p*obs_p*particles[key][1]
                norm += particles[key][1]
        else:
            # calculating belief 
            for particle in particles:
                tmp_state = particle.copy()
                state,reward, _, _ = tmp_state.step(action)
                
                trans_p = (state.get_trans_p(action))[1]
                obs_p = (state.get_obs_p(action))[1]
                belief_reward += reward*trans_p*obs_p
                norm += 1
        return belief_reward/self.smallbag_size

    def importance_sampling(self,smallbag,action,next_state):
        next_smallbag = []
        next_smallbag.append(next_state)

        while len(next_smallbag) < self.smallbag_size:
            # (1) sampling the particle from smallbag
            particle = random.choice(smallbag)

            # (2) generating particle' from particle using G
            tmp_state = particle.copy()
            state,reward, _, _ = tmp_state.step(action)

            # (3) storing the generated particle particle' in the new smallbag
            next_smallbag.append(state)
        
        return next_smallbag

    def simulate_action(self, node, action):
        # 1. Acting
        next_state, reward, _, _ = node.state.step(action)
        next_node = RhoANode(action, next_state, node.depth+1, node)

        # 2. Returning the next node and the reward
        return next_node, reward
    
    
    def rollout_policy(self,state):
        return random.choice(state.actions)

    def rollout(self,node,problem,smallbag):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        node.depth += 2

        next_smallbag = self.importance_sampling(smallbag,action,node.state)

        # 4. Rolling out
        R = self.rhofunction(smallbag, action) +\
            self.discount_factor*self.rollout(node,problem,next_smallbag)
        return R


    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return RhoONode(observation=obs,state=tmp_state,depth=depth,parent=None)


    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        if (time.time() - self.start_time_budget) > self.time_budget:
            return True
        return node.state.is_final_state()

    
    def simulate(self, node, problem, smallbag):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node, problem, smallbag)

        # 3. Selecting the best action
        action = node.select_action(mode='ucb') 

        # 4. Simulating the action
        (action_node, reward) = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 6. Getting the observation and adding the observation child on the tree
        observation_node = None
        observation = action_node.state.get_observation()

        for child in action_node.children:
            if action_node.state.observation_is_equal(child.observation, observation):
                observation_node = child
                observation_node.state = action_node.state.copy()
                break
        
        if observation_node is None:
            observation_node = action_node.add_child(observation)
        observation_node.visits += 1

        # 7. Generating the new smallbag
        next_smallbag = self.importance_sampling(smallbag,action,observation_node.state)
        for particle in smallbag:
            node.particle_filter.append(particle)
            node.add_to_cummulative_bag(particle,action)
        node.particle_filter.append(node.state)
        node.add_to_cummulative_bag(node.state,action)

        # 8. Calculating the reward, quality and updating the node
        R = self.rhofunction(node.cummulative_bag, action) + \
            float(self.discount_factor * self.simulate(observation_node,problem,next_smallbag))
        
        # - node update
        node.particle_filter.append(node.state)
        node.update(action, R)
        return R
    

    def search(self, root, problem):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        self.start_time_budget = time.time()
        while it < self.max_it:
            
            # a. Sampling the belief state for simulation
            if len(root.particle_filter) < 1 + self.smallbag_size:
                beliefState = problem.sample_nstate(root.state, 1 + self.smallbag_size)
                beliefState, smallbag = sampled_states[0], sampled_states[1:]
            else:
                sampled_states = random.sample(root.particle_filter, 1 + self.smallbag_size)
                beliefState, smallbag = sampled_states[0], sampled_states[1:]
            root.state = beliefState

            # b. simulating
            self.simulate(root, problem, smallbag)
            it += 1

            if (time.time() - self.start_time_budget) > self.time_budget:
                return root.get_best_action()
            
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
            print('<!> No previous root found, creating a new one')
            self.root = RhoONode(observation=observation,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            print('<!> Advancing within the existent tree')
            self.root = find_new_PO_root(
                state, previous_action, observation, self.root
            )

        # 3. Performing particle revigoration
        if self.pr:
            particle_revigoration(state, problem, self.root, self.k)

        # 4. Searching for the best action within the tree
        print('P> Starting the search for the best action')
        best_action = self.search(self.root, problem)
        self.root.show_qtable()
        return [best_action]

###
# RhoPOMCP find new root
###
def find_new_PO_root(
 current_state, previous_action, current_observation, previous_root
) -> RhoONode:
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no previous root found')
        return new_root

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no action node found')
        print('<!> Previous action:', previous_action)
        return new_root

    # b. walking over observation nodes
    for child in action_node.children:
        obs = child.observation
        if child.state.observation_is_equal(obs, current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no observation node found')
        return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    print('<y> Walking on the tree to find the new root node')
    return new_root