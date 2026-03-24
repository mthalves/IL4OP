import math
import random

from isaaclab_experiments.src.planning_algorithms.node.quality import ANode, ONode
from isaaclab_experiments.src.planning_algorithms.qlearn import create_etable, entropy, iucb_select_action

class IANode(ANode):

    def __init__(self, action, state, depth, parent=None):
        super(IANode,self).__init__(action,state,depth,parent)
        self.action = action

    def add_child(self, observation):
        # if the node with such observation already exists, return it
        for child in self.children:
            if child.state.observation_is_equal(observation):
                return child
            
        # else, create a new one
        state = self.state.copy()
        child = IONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class IONode(ONode):

    def __init__(self, observation, state, depth, parent=None):
        super(IONode,self).__init__(None,state,depth,parent)
        self.observation = observation
        
        self.particle_filter = []   

        self.observation_distribution = {}
        self.entropy = 0.0
        self.cumentropy = 0.0
        self.max_entropy = 1.0

        self.etable = create_etable(self.actions) 
        
    def add_child(self, state, action):
        # if the node with such action already exists, return it
        for child in self.children:
            if action == child.action:
                child.state = state
                return child
            
        # else, create a new one
        child = IANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def add_to_observation_distribution(self, observations_distribution):
        # adding info to state distribution
        for obs in observations_distribution:
            key = hash(str(obs))
            if key in self.observation_distribution:
                self.observation_distribution[key] += 1
            else:
                self.observation_distribution[key] = 1
    
    def get_alpha(self):
        adjust_value = 0.2
        if self.visits == 0:
            return 1 - adjust_value

        # Old Version
        decaying_factor = math.e*math.log(self.visits)/self.visits
        entropy_value_trend = self.cumentropy/\
                            (self.visits*self.max_entropy)
        norm_entropy = decaying_factor*entropy_value_trend

        return (1 - 2*adjust_value)*(norm_entropy) +adjust_value

    def update(self, action, result):
        # Q-table update
        self.qtable[str(action)]['trials'] += 1

        self.qtable[str(action)]['sumvalue'] += result

        self.qtable[str(action)]['qvalue'] += \
            (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])
        self.value += (result-self.value)/self.visits

        # Observation-table update
        result_entropy = entropy(self.observation_distribution)
        self.etable[str(action)]['trials'] += 1

        self.etable[str(action)]['cumentropy'] += result_entropy
        self.cumentropy += result_entropy

        self.etable[str(action)]['entropy'] += \
            (float(result_entropy) - self.etable[str(action)]['entropy']) /\
                 float(self.etable[str(action)]['trials'])
        self.entropy += (result_entropy - self.entropy) / self.visits

        if result_entropy > self.etable[str(action)]['max_entropy']:
            self.etable[str(action)]['max_entropy'] = result_entropy 

        if result_entropy > self.max_entropy:
            self.max_entropy = self.entropy


    def select_action(self, coef=0.5, mode='iucb'):
        # UCB
        if mode == 'iucb' or mode == 'max' or mode == 'iucb-max':
            return iucb_select_action(self,alpha=coef,mode='max')
        elif mode == 'min'  or mode == 'iucb-min':
            return iucb_select_action(self,alpha=coef,mode=mode)
        # Not Implemented
        else:
            print('Invalid action selection mode:',mode)
    
    def get_best_action(self, alpha, mode='iucb-max'):
        # 1. Intialising the support variables
        # - maximisation
        if mode == 'iucb-max' or mode == 'iucb':
            target = 'max'
            best_action, bestQ = None, -100000000000
        # - minimisation
        elif mode == 'iucb-min':
            target = 'min'
            best_action, bestQ = None, 100000000000
        # - not implemented
        else:
            print('Invalid best action mode:',mode)
            raise NotImplemented

        # 2. Looking for the best action (max qvalue action)
        actionsQ = {}
        for a in self.actions:
            actionsQ[str(a)] = (1-alpha)*self.qtable[str(a)]['qvalue'] + \
             (alpha)*(self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy'])

            # maximisation case
            if target == 'max' and  actionsQ[str(a)] > bestQ  and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = actionsQ[str(a)]
                best_action = a

            # minimisation case
            elif target == 'min' and actionsQ[str(a)] < bestQ and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = actionsQ[str(a)]
                best_action = a

        # 3. Checking if a tie case exists
        tieCases = []
        for a in self.actions:
            if actionsQ[str(a)] == bestQ:
                tieCases.append(a)

        if len(tieCases) > 0:
            # 3.1. trying tie break by number of visits
            trials = [self.qtable[str(a)]['trials'] for a in tieCases]
            max_trial = max(trials)
            trialTieCases = []
            for a in tieCases:
                if self.qtable[str(a)]['trials'] == max_trial:
                    trialTieCases.append(a)

            # 3.2. checking if a tie case persists
            if len(trialTieCases) > 1:
                best_action = random.choice(trialTieCases)
            else:
                best_action = trialTieCases[0]
        else:
            best_action = random.sample(self.actions,1)[0]
            
        # 4. Returning the best action
        return best_action

    def show_qtable(self):
        print('%8s %8s %8s %8s | %8s %8s %8s' % \
            ('Action','Q-Value','SumValue','Trials','Entropy','MaxEntropy','NormEntropy'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],\
                self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy'],\
                    self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1],action_dict[x][2]), reverse=True)
        
        for a in action_dict:
            print('%8s %8.4f %8.4f %8d | %8.4f %8.4f %8.4f' \
                % (str(a),self.qtable[str(a)]['qvalue'],\
                self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials'],\
                self.etable[str(a)]['entropy'],self.etable[str(a)]['max_entropy'],\
                self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy']))
                
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')

###
# POMCP's proposed modification 
###
# POMCP uses find_new_PO_root from node.py module
# > from src.reasoning.node import find_new_PO_root
def find_new_information_root(
 current_state, previous_action, current_observation, previous_root
) -> tuple[IONode, float]:
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    Px = 0
    if previous_root is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no previous root found')
        return new_root, Px

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no action node found')
        print('<!> Previous action:', previous_action)
        return new_root, Px

    # b. walking over observation nodes
    for child in action_node.children:
        obs = child.observation
        if child.state.observation_is_equal(obs, current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        print('<!> Creating new root node: no observation node found')
        return new_root, Px

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    Px = new_root.visits/previous_root.visits
    new_root.parent = None
    new_root.update_depth(0)
    print('<y> Walking on the tree to find the new root node')
    return new_root, Px

# POMCP uses particle_revigoration from node.py module
# > from src.reasoning.node import particle_revigoration
def informative_particle_revigoration(state, problem, root, k, Px):
    # 1. Copying the current root particle filter
    current_particle_filter = []
    for particle in root.particle_filter:
        current_particle_filter.append(particle)
    Px =  Px if len(current_particle_filter) > 1 else 0.0
    
    # 2. Reinvigorating particles for the new particle filter or
    # picking particles from the uniform distribution
    root.particle_filter = []
    particle_counter = 0
    while(particle_counter < (Px)*k):
        particle = random.sample(current_particle_filter,1)[0]
        root.particle_filter.append(particle)
        particle_counter += 1
        

    particle_counter = 0
    while(particle_counter < (1-Px)*k):
        particle = problem.sample_state(state)
        root.particle_filter.append(particle)
        particle_counter += 1