import random

from isaaclab_experiments.src.planning_algorithms.node.base import Node
from isaaclab_experiments.src.planning_algorithms.qlearn import create_qtable, ucb_select_action


class CNode(Node):

    def __init__(self, action, state, depth, parent=None):
        super(CNode,self).__init__(state,depth,parent)
        self.value = 0
        self.action = action
        self.action_range = state.action_range
        self.qtable = create_qtable(state.special_actions)

    @property
    def actions(self) -> list[float]:
        return [child.action for child in self.children]

    def update(self, action, result):
        # Actions already tried
        if str(action) in self.qtable:
            self.qtable[str(action)]['trials'] += 1
            self.qtable[str(action)]['sumvalue'] += result
            self.qtable[str(action)]['qvalue'] += \
                (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])
            self.value += (result-self.value)/self.visits
        # New actions
        else:
            self.qtable[str(action)] = {'qvalue':0.0,'sumvalue':0.0,'trials':0}
            self.qtable[str(action)]['trials'] += 1
            self.qtable[str(action)]['sumvalue'] += result
            self.qtable[str(action)]['qvalue'] += \
                (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])
            self.value += (result-self.value)/self.visits

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

    def get_best_action(self,mode='max'):
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

    def show_qtable(self):
        print('%8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials'))
        action_dict = {}
        for a in self.actions:
            action_dict[str(a)] = [self.qtable[str(a)]['qvalue'],self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1]), reverse=True)
        
        for a in action_dict:
            print('%8s %8.4f %8.4f %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                        self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')

class CANode(CNode):

    def __init__(self, action, state, depth, parent=None):
        super(CANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None
    
    def calculate_observation_weights(self):
        observation_weights = {}
        for child in self.children:
            observation_weights[str(child.observation)] = child.visits
        total = sum(observation_weights.values())

        if total != 0:
            for key in observation_weights:
                observation_weights[key] /= total
        return observation_weights
    
    def sample_child(self):
        sampled_child = None
        
        weights = self.calculate_observation_weights()
        if len(weights) > 0:
            # sample observation
            observation = random.choices(
                list(weights.keys()),
                weights=list(weights.values()), k=1)
        
            # search for the existent node (if not found, return None)
            for child in self.children:
                if self.state.observation_is_equal(child.observation, observation):
                    observation_node = child
                    observation_node.state = self.state.copy()
                    break

        return sampled_child


    def add_child(self, observation):
        state = self.state.copy()
        child = self.get_child(observation)
        if child is None:
            child = CONode(observation,state,self.depth+1,self)
            self.children.append(child)
        return child
    
    def get_child(self, observation):
        for child in self.children:
            if child.observation == observation:
                return child
        return None

class CONode(CNode):

    def __init__(self,observation, state, depth, parent=None):
        super(CONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.particles_set = {}

        self._set_default_actions()

    def _set_default_actions(self):
        for a in self.state.special_actions:
            if a not in [child.action for child in self.children]:
                next_state, _, _, _ = self.state.step(a)
                next_node = CANode(a, next_state, self.depth+1, self)
                _ = self.append_action_node(next_node)

    def append_action_node(self, action_node):
        if action_node.action in [c.action for c in self.children]:
            for child in self.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            self.children.append(action_node)
        return action_node

    def action_prog_widen(self,coef={'ka':0.5, 'alpha_a':0.5, 'c':0.5}, mode='max'):
        ka, alpha_a, c = coef['ka'], coef['alpha_a'], coef['c']

        # checking the widening condition
        if len(self.children) <= ka*self.visits**alpha_a:
            new_action = self.state.sample_new_action()
            next_state, reward, _, _ = self.state.step(new_action)
            self.add_child(next_state, new_action)
            self.qtable[str(new_action)] = {'qvalue':0.0,'sumvalue':0.0,'trials':0}

        # selecting the next action through ucb   
        return ucb_select_action(self,c=c,mode=mode)
        
    def add_child(self,state,action):
        child = self.get_child(action)
        if child is None:
            child = CANode(action,state,self.depth+1,self)
            self.children.append(child)
        child.state = state.copy()
        return child

    def get_child(self,action):
        for child in self.children:
            if str(child.action) == str(action):
                return child
        return None
    
    def show_qtable(self):
        print('#Actions %d - Children %d' % (len(self.actions), len(self.children)))
        print('%15s %8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials','#Obs'))
        action_dict = {}
        for a in self.actions:
            action_dict[str(a)] = [self.qtable[str(a)]['qvalue'],self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1]), reverse=True)
        
        for a in action_dict:
            child = self.get_child(a)
            print('%15s %8.4f %8.4f %8d %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                        self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials'], len(child.children)))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')
    
def find_new_CPO_root(
 current_state, previous_action, current_observation, previous_root
) -> CONode:
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = CONode(observation=None,state=current_state,depth=0,parent=None)
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
        new_root = CONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # b. walking over observation nodes
    for child in action_node.children:
        obs = child.observation
        if child.state.observation_is_equal(obs, current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = CONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root