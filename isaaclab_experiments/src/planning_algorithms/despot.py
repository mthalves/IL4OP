from isaaclab_experiments.src.planning_algorithms.node import \
    DespotANode, DespotONode, find_new_despot_root

class DESPOT:

    def __init__(self, max_depth, max_it, kwargs):
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it

        self.discount_factor = kwargs.get("discount_factor", 0.95)
        self.num_scenarios   = kwargs.get("num_scenarios", 100)
        self.lambda_reg      = kwargs.get("lambda_reg", 0.005)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_leaf(self, node):
        return node.depth >= self.max_depth

    def is_terminal(self, node):
        return node.state is None or node.state.is_final_state()

    def subtree_size(self, node):
        return 1 + sum(self.subtree_size(c) for c in node.children)

    # ------------------------------------------------------------------
    # Bounds (simple but valid)
    # ------------------------------------------------------------------

    def estimate_lower_bound(self, node):
        # pessimistic default
        return 0.0

    def estimate_upper_bound(self, node):
        # optimistic remaining reward
        horizon = self.max_depth - node.depth
        r_max = 1.0
        return sum(
            (self.discount_factor ** t) * r_max for t in range(horizon)
        )

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(self, onode):
        # LEAF INITIALIZATION
        if len(onode.children) == 0:
            if onode.upper_bound == 0.0 and onode.lower_bound == 0.0:
                onode.lower_bound = self.estimate_lower_bound(onode)
                onode.upper_bound = self.estimate_upper_bound(onode)
            return onode.upper_bound, onode.lower_bound

        best_u = -float("inf")
        best_l = -float("inf")

        for anode in onode.children:
            u_sum, l_sum = 0.0, 0.0

            total_scenarios = sum(
                child.num_scenarios for child in anode.children
            )

            for child in anode.children:
                u, l = self.backup(child)
                w = child.num_scenarios / max(1, total_scenarios)

                u_sum += w * (child.reward + self.discount_factor * u)
                l_sum += w * (child.reward + self.discount_factor * l)

            penalty = self.lambda_reg * self.subtree_size(anode)

            anode.upper_bound = u_sum - penalty
            anode.lower_bound = l_sum - penalty

            best_u = max(best_u, anode.upper_bound)
            best_l = max(best_l, anode.lower_bound)

        onode.upper_bound = best_u
        onode.lower_bound = best_l
        return best_u, best_l

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self, onode, problem):
        if self.is_leaf(onode) or self.is_terminal(onode):
            return

        for action in onode.actions:
            anode = DespotANode(
                action=action,
                state=onode.state,
                depth=onode.depth + 1,
                parent=onode,
                actions=onode.state.actions,
                scenarios=onode.scenarios
            )
            onode.children.append(anode)

            obs_groups = {}

            for s in onode.scenarios:
                next_state, reward, _, _ = s.step(action)
                obs = next_state.get_observation()
                key = str(obs)

                if key not in obs_groups:
                    obs_groups[key] = []

                obs_groups[key].append((next_state, reward))

            for key, transitions in obs_groups.items():
                scenarios = [ns for ns, _ in transitions]
                rewards   = [r for _, r in transitions]

                child = DespotONode(
                    observation = key,
                    state       = scenarios[0],
                    depth       = onode.depth + 2,
                    parent      = anode,
                    actions     = problem.actions,
                    scenarios   = scenarios
                )

                child.reward = sum(rewards) / len(rewards)
                child.num_scenarios = len(scenarios)

                anode.children.append(child)

    # ------------------------------------------------------------------
    # Leaf selection (optimistic DESPOT traversal)
    # ------------------------------------------------------------------

    def select_leaf(self, onode):
        while True:
            if self.is_leaf(onode) or self.is_terminal(onode):
                return onode

            if len(onode.children) == 0:
                return onode

            # optimistic action
            best_anode = max(
                onode.children, key=lambda a: a.upper_bound
            )

            if len(best_anode.children) == 0:
                return onode

            # expand most uncertain observation
            onode = min(
                best_anode.children,
                key=lambda o: o.lower_bound
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, root, problem):
        for _ in range(self.max_it):
            leaf = self.select_leaf(root)
            self.expand(leaf, problem)
            self.backup(root)

        root.show()
        return root.get_best_action(mode="lower_bound")

    # ------------------------------------------------------------------
    # Planning entry point
    # ------------------------------------------------------------------

    def plan(self, agent, problem):
        state = problem.get_current_state(agent["pos"])
        observation = state.get_observation()
        prev_action = agent["action_history"][-1] if agent["action_history"] else None

        if self.root is None:
            scenarios = problem.sample_n_states(state, self.num_scenarios)
            self.root = DespotONode(
                observation = observation,
                state       = state,
                depth       = 0,
                parent      = None,
                actions     = state.actions,
                scenarios   = scenarios
            )
        else:
            self.root = find_new_despot_root(
                state, prev_action, observation, self.root
            )

            if len(self.root.scenarios) < self.num_scenarios:
                extra = problem.sample_n_states(
                    state, self.num_scenarios - len(self.root.scenarios)
                )
                self.root.scenarios.extend(extra)

        best_action = self.search(self.root, problem)
        return [best_action]