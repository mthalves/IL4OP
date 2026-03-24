import random

from isaaclab_experiments.src.planning_algorithms.node.base import Node
from isaaclab_experiments.src.planning_algorithms.qlearn import create_qtable, ucb_select_action


class CBNode(Node):

    def __init__(self, action, state, depth, parent=None):
        super(CBNode,self).__init__(state,depth,parent)
        self.value = 0
        self.action = action
        self.action_range = state.action_range
        self.qtable = create_qtable(state.special_actions)

        self.particle_filter = []
        self.observation_dist = {}

        self._set_default_actions()

    def _set_default_actions(self):
        for a in self.state.special_actions:
            if a not in [child.action for child in self.children]:
                next_state, _, _, _ = self.state.step(a)
                next_node = CBNode(a, next_state, self.depth+1, self)
                _ = self.add_child(next_node)

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
    
    def sample_from_particle_filter(self, m):
        sampled_particles = random.sample(self.particle_filter, m)
        states = [particle[0] for particle in sampled_particles] 
        rewards = [particle[1] for particle in sampled_particles]
        return states, rewards
    
    def sample_states_from_particle_filter(self, m):
        sampled_particles = random.sample(self.particle_filter, m)
        return [particle[0] for particle in sampled_particles] # picking only the state, not reward

    def add_to_observation_dist(self, observation):
        key = self.state.hash_observation(observation)
        if key in self.observation_dist:
            self.observation_dist[key] += 1
        else:
            self.observation_dist[key] = 0

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

            

def belief_particle_revigoration(state, problem, root, k):
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
            sampled_state = problem.sample_state(state)
            root.particle_filter.append([sampled_state, 0])


def find_new_belief_root(
 current_state, previous_action, current_observation, previous_root
) -> CBNode:
    # 1. If the root doesn't exist yet, create it
    if previous_root is None:
        new_root = CBNode(action=previous_action,state=current_state,depth=0,parent=None)
        return new_root

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, new_root = None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = CBNode(action=previous_action,state=current_state,depth=0,parent=None)
        return new_root

    # 3. Definig the new root and updating the depth
    new_root = action_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root