
class Node(object):

    def __init__(self, state, depth, parent=None):
        self.state = state
        self.depth = depth
        self.parent = parent
        self.children = []
        self.visits = 0 

    def add_child(self, state, *args, **kwargs):
        # if the node with such state already exists, return it
        for c in self.children:
            if c.state.is_equal(state):
                return c
            
        # else, create a new one
        child = Node(state,self.depth+1,self)
        self.children.append(child)
        return child

    def remove_child(self,child):
        for c in self.children:
            if c == child:
                self.children.remove(child)
                break

    def update_depth(self,new_depth):
        self.depth = new_depth
        for c in self.children:
            c.update_depth(new_depth+1)