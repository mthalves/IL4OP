from isaaclab_experiments.src.planning_algorithms.node.quality import ANode, ONode

class RhoANode(ANode):

    def __init__(self,action, state, depth, parent=None):
        super(RhoANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,observation):
        state = self.state.copy()
        child = RhoONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class RhoONode(ONode):

    def __init__(self,observation, state, depth, parent=None):
        super(RhoONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.cummulative_bag = {}
    
    def add_child(self,state,action):
        child = RhoANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def add_to_cummulative_bag(self,particle,action):
        obs_p = particle.get_obs_p(action)[1]
        hash_key = particle.hash_state()
        if hash_key not in self.cummulative_bag:
            self.cummulative_bag[hash_key] =  [particle,obs_p]
        else:
            self.cummulative_bag[hash_key] =  [particle,self.cummulative_bag[hash_key][1] + obs_p]