from isaaclab_experiments.src.planning_algorithms.node import IANode, IONode, \
                            find_new_information_root, informative_particle_revigoration

import random

class IBPOMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it
        
        self.alpha                  = kwargs.get('alpha',0.5) # information weight (alves2023information)
        self.discount_factor        = kwargs.get('discount_factor',0.95) # discount factor (historical weight)
        self.particle_revigoration  = kwargs.get('particle_revigoration',True) # enable particle revigoration (silver2010pomcp)
        self.k                      = kwargs.get('k', 100) # particle filter size

        self.state_distribution = {}
        self.state_entropy_hist = []
        self.state_entropy_var_hist = []

        self.observation_distributsion = {}

    def simulate_action(self, node, action):
        # 1. Acting
        next_state, reward, _, _ = node.state.step(action)
        next_node = IANode(action, next_state, node.depth+1, node)

        # 2. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        return random.choice(state.actions)

    def rollout(self,node,problem):
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

        # 4. Rolling out
        R = reward + (self.discount_factor * self.rollout(node, problem))
        return R

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return IONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.is_final_state()

    def simulate(self, node, problem):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            obs = node.observation
            observation_found = [obs]
            return 0, observation_found

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)

            obs = node.observation
            observation_found = [obs]
            return self.rollout(rollout_node, problem), observation_found
        
        # 3. Selecting the best action
        action = node.select_action(coef=self.alpha,mode='iucb') 

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

        # 7. Calculating the reward, quality and updating the node
        future_reward, observation_found = self.simulate(observation_node, problem)
        R = reward + (self.discount_factor * future_reward)

        # - node update
        node.add_to_observation_distribution(observation_found)
        node.particle_filter.append(node.state)
        node.update(action, R)

        observation_found.append(observation)
        return R, observation_found

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
            self.alpha = root.get_alpha()
            self.simulate(root, problem)
            it += 1

        self.alpha = root.get_alpha()
        return root.get_best_action(self.alpha)

    def plan(self, agent, problem):
        # 1. Getting the current state and previous action-observation pair
        print('P> Planning for agent:', agent['name'], '\n- max depth:', self.max_depth, \
              'P> max it:', self.max_it, 'k:', self.k)
        state = problem.get_current_state(agent['pos'])
        print('P> Agent pos:', state.agent_pos,'- Tasks found:',state.tasks_found)
        observation = state.get_observation()
        print('P> Current observation:', observation)
        previous_action = None if len(agent['action_history']) == 0 \
                            else agent['action_history'][-1]

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            print('<!> No previous root found, creating a new one')
            Px = 0
            self.root = IONode(observation=observation,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            print('<!> Advancing within the existent tree')
            self.root, Px = find_new_information_root(
                state, previous_action, observation, self.root
            )

        # 3. Performing particle revigoration
        if self.particle_revigoration:
            informative_particle_revigoration(state, problem, self.root, self.k, Px)

        # 4. Searching for the best action within the tree
        print('P> Starting the search for the best action')
        best_action = self.search(self.root, problem)
        self.root.show_qtable()
        return [best_action]