import random

from isaaclab_experiments.src.planning_algorithms.node.base import Node
from isaaclab_experiments.src.planning_algorithms.qlearn import \
    create_qtable, ucb_select_action

class QNode(Node):

    def __init__(self,action, state, depth, parent=None):
        super(QNode,self).__init__(state,depth,parent)
        self.value = 0
        self.action = action
        self.actions = state.actions
        self.qtable = create_qtable(self.actions)

    def update(self, action, result):
        self.qtable[str(action)]['trials'] += 1
        self.qtable[str(action)]['sumvalue'] += result
        self.qtable[str(action)]['qvalue'] += \
            (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])

        self.value += (result-self.value)/self.visits

    def select_action(self, coef=0.5, mode='ucb'):
        # UCB
        if mode == 'ucb' or mode == 'max' or mode == 'ucb-max':
            return ucb_select_action(self,c=coef,mode='max')
        elif mode == 'min'  or mode == 'ucb-min':
            return ucb_select_action(self,c=coef,mode=mode)
        # Not Implemented
        else:
            print('Invalid action selection mode:',mode)
            raise NotImplemented

    def get_actions_prob_distribution(self, mode='max', max_reward=1):
        prob_distribution = {}
        
        norm = 0.0
        for a in self.qtable:
            if mode == 'max':
                prob_distribution[a] = self.qtable[a]['qvalue']
            elif mode == 'min':
                prob_distribution[a] = (max_reward-self.qtable[a]['qvalue'])
            else:
                raise NotImplemented
            norm += prob_distribution[a]
        
        if norm == 0.0:
            for a in prob_distribution:
                prob_distribution[a] = 1/len(prob_distribution)
        else:
            for a in prob_distribution:
                prob_distribution[a] /= norm

        return prob_distribution

    def get_best_action(self, mode='max', *args, **kwargs):
        # 1. Intialising the support variables
        # - maximisation
        if mode == 'max' or mode == 'ucb':
            target = 'max'
            best_action, bestQ = None, -100000000000
        # - minimisation
        elif mode == 'min' or mode == 'ucb-min':
            target = 'min'
            best_action, bestQ = None, 100000000000
        # - not implemented
        else:
            print('Invalid best action mode:',mode)
            raise NotImplemented

        # 2. Looking for the best action (max qvalue action)
        for a in self.actions:
            if target == 'max' and  \
             self.qtable[str(a)]['qvalue'] > bestQ  and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = self.qtable[str(a)]['qvalue']
                best_action = a
            elif target == 'min' and \
             self.qtable[str(a)]['qvalue'] < bestQ and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = self.qtable[str(a)]['qvalue']
                best_action = a

        # 3. Checking if a tie case exists
        tieCases = []
        for a in self.actions:
            if self.qtable[str(a)]['qvalue'] == bestQ:
                tieCases.append(a)

        if len(tieCases) > 1:
            # trying tie break by number of visits
            trials = [self.qtable[str(a)]['trials'] for a in tieCases]
            max_trial = max(trials)
            trialTieCases = []
            for a in tieCases:
                if self.qtable[str(a)]['trials'] == max_trial:
                    trialTieCases.append(a)

            if len(trialTieCases) > 1:
                best_action = random.choice(trialTieCases)
            else:
                best_action = trialTieCases[0]

        # 4. Returning the best action
        if best_action is None:
            best_action = random.sample(self.actions,1)[0]
        
        return best_action
    
    def size_in_memory(self):
        raise NotImplemented

    def show_qtable(self):
        print('%8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1]), reverse=True)
        
        for a in action_dict:
            print('%8s %8.4f %8.4f %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                        self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')

class ANode(QNode):

    def __init__(self, action, state, depth, parent=None):
        super(ANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self, observation):
        # if the node with such observation already exists, return it
        for child in self.children:
            if child.state.observation_is_equal(observation):
                return child
            
        # else, create a new one
        state = self.state.copy()
        child = ONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child
    
    def get_child(self, observation):
        for child in self.children:
            if child.observation == observation:
                return child
        return None

class ONode(QNode):

    def __init__(self, observation, state, depth, parent=None):
        super(ONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.particles_set = {}
        
    def add_child(self, state, action):
        # if the node with such action already exists, return it
        for child in self.children:
            if action == child.action:
                child.state = state
                return child
            
        # else, create a new one
        child = ANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def get_child(self, action):
        for child in self.children:
            if child.action == action:
                return child
        return None

def find_new_PO_root(
 current_state, previous_action, current_observation, previous_root
) -> ONode:
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
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
        new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
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
        new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no observation node found')
        return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    print('<y> Walking on the tree to find the new root node')
    return new_root

def particle_revigoration(state, problem, root, k):
    # 1. Copying the current root particle filter
    current_particle_filter = []
    for particle in root.particle_filter:
        current_particle_filter.append(particle)
    
    # 2. Reinvigorating particles for the new particle filter or
    # picking particles from the uniform distribution
    root.particle_filter = []
    if len(current_particle_filter) > 0: # particle ~ F_r
        while(len(root.particle_filter) < k):
            particle = random.sample(current_particle_filter,1)[0]
            root.particle_filter.append(particle)
    else: # particle ~ U
        while(len(root.particle_filter) < k):
            particle = problem.sample_state(state)
            root.particle_filter.append(particle)