from isaaclab_experiments.src.planning_algorithms.node.base import Node

class DespotANode(Node):
    def __init__(self, action, state, depth, parent, actions, scenarios=[]):
        super().__init__(state, depth, parent)
        self.scenarios = scenarios if scenarios is not None else []
        self.num_scenarios = len(scenarios)

        self.action = action
        self.actions = actions
        self.observation = None

        self.reward      = 0.0
        self.upper_bound = 0.0
        self.lower_bound = 0.0


class DespotONode(Node):
    def __init__(self, observation, state, depth, parent, actions, scenarios=[]):
        super().__init__(state, depth, parent)
        self.scenarios = scenarios if scenarios is not None else []
        self.num_scenarios = len(scenarios)

        self.action = None
        self.actions = actions
        self.observation = observation

        self.reward      = 0.0
        self.upper_bound = 0.0
        self.lower_bound = 0.0

    def get_best_action(self, mode="upper_bound"):
        """
        DESPOT action selection:
        argmax_a lower_bound(Q(b0, a))
        """
        assert len(self.children) > 0, "Root has no action children"

        if mode == "lower_bound":
            best = max(self.children, key=lambda a: a.lower_bound)
        elif mode == "upper_bound":
            best = max(self.children, key=lambda a: a.upper_bound)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return best.action

    def subtree_size(self, node):
        return 1 + sum(self.subtree_size(c) for c in node.children)
    
    def show(self):
        print("\n=== DESPOT Root Decision ===")
        print(f"{'Action':>10} | {'LB':>8} | {'UB':>8} | {'Scen':>6} | {'Size':>6}")
        print("-" * 50)

        for anode in self.children:
            lb = anode.lower_bound
            ub = anode.upper_bound
            scen = sum(c.num_scenarios for c in anode.children)
            size = self.subtree_size(anode)

            print(f"{str(anode.action):>10} | {lb:8.3f} | {ub:8.3f} | {scen:6d} | {size:6d}")

        best = max(self.children, key=lambda a: a.lower_bound)
        print("\n✔ Selected action:", best.action)


def find_new_despot_root(
 current_state, previous_action, current_observation, previous_root
) -> DespotONode:
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = DespotONode(\
            observation=current_observation,state=current_state,depth=0,parent=None,\
                actions=current_state.actions,scenarios=[])
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
        new_root = DespotONode(\
            observation=current_observation,state=current_state,depth=0,parent=None,\
                actions=current_state.actions,scenarios=[])
        print('<!> Creating new root node: no action node found')
        return new_root


    # b. walking over observation nodes
    for child in action_node.children:
        obs = child.observation
        if child.state.observation_is_equal(obs, current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = DespotONode(\
            observation=current_observation,state=current_state,depth=0,parent=None,\
                actions=current_state.actions,scenarios=[])
        print('<!> Creating new root node: no observation node found')
        return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    print('<y> Walking on the tree to find the new root node')
    return new_root