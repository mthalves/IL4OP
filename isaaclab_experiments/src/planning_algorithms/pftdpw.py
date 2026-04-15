from isaaclab_experiments.src.planning_algorithms.node import CBNode, \
                                                find_new_belief_root, belief_particle_revigoration

import random

class PFTDPW(object):

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

        self.c = kwargs.get('c', 50) # exploration constant for action progressive widening

        ###
        # Progressive Widening parameters
        ###
        # - action widening
        self.ka = kwargs.get('ka', 15.0) 
        self.alpha_a = kwargs.get('alpha_a', 0.03)

        # - state/observation widening
        self.ko = kwargs.get('ko', 4.0)        
        self.alpha_o = kwargs.get('alpha_o', 0.01)

        # - belief bag size
        self.m = 20
        
    def simulate_action(self, node, action):
        new_particle_filter = []
        total_reward = 0
        for _ in range(self.m):
            particle = random.sample(node.particle_filter,1)[0]
            beliefState = particle[0].copy()
            # 1. Acting
            next_state, reward, _, _ = beliefState.step(action)
            new_particle_filter.append((next_state,reward))
            total_reward += reward
        return new_particle_filter, float(total_reward/self.m)

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
        node.depth += 1

        # 4. Rolling out
        R = reward + (self.discount_factor * self.rollout(node, problem))
        return R

    def get_rollout_node(self,node):
        tmp_state = node.state.copy()
        depth = node.depth
        rollout_node = CBNode(action=None,state=tmp_state,depth=depth,parent=None)
        rollout_node._set_default_actions()
        return rollout_node

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
                    coef={'ka':self.ka,'alpha_a':self.alpha_a, 'c':self.c}) 

        # - checking if this action is already tried
        next_node = node.get_child(action)
        if not next_node:
            next_state, reward, _, _ = node.state.step(action)
            next_node = node.add_child(next_state, action)

        # 3. If the observation widenning criteria is met, do it
        reward, future_reward = 0, 0
        if len(next_node.children) <= self.ko*next_node.visits**self.alpha_o:            
            # - calculating current reward
            new_particle_filter, reward = self.simulate_action(node, action)
            particle = random.sample(new_particle_filter,1)[0]
            next_node.state = particle[0].copy()

            # - expanding the tree
            for p in new_particle_filter:
                next_node.particle_filter.append(p)
            rollout_node = self.get_rollout_node(next_node)
            future_reward = self.rollout(rollout_node, problem)

        # - otherwise, sample an existing observation and state for simulation
        else:
            # - calculating current reward
            if len(next_node.particle_filter) > 0:
                particle = random.sample(
                    next_node.particle_filter,1)[0]
            
                next_node.state = particle[0].copy()
                reward = particle[1]
            else:
                new_particle_filter, reward = self.simulate_action(node, action)
                particle = random.sample(new_particle_filter,1)[0]
                next_node.state = particle[0].copy()

                for p in new_particle_filter:
                    next_node.particle_filter.append(p)
            
            # - expanding the tree
            future_reward = self.simulate(next_node, problem)

        # 4. Node update
        node.visits += 1
        next_node.visits += 1
        R = reward + (self.discount_factor * future_reward)
        node.update(action, R)
        return R

    def search(self, root, problem):
        # 1. Performing the PFT-DPW search
        it = 0
        while it < self.max_it:
            self.simulate(root, problem)
            it += 1
        return root.get_best_action()

    def plan(self, agent, problem):
        # 1. Getting the current state and previous action-observation pair
        state = problem.get_current_state(agent['pos'])
        previous_action = None if len(agent['action_history']) == 0 \
                            else agent['action_history'][-1]

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = CBNode(action=None,state=state,depth=0,parent=None)
            self.root._set_default_actions()
        # or advancing within the existent tree
        else:
            self.root = find_new_belief_root(
                state, previous_action, self.root
            )

        # 3. Performing particle revigoration
        if self.particle_revigoration:
            belief_particle_revigoration(state, problem, self.root, self.k)

        # 4. Searching for the best action within the tree
        best_action = self.search(self.root, problem)
        self.root.show_qtable()
        return [best_action]