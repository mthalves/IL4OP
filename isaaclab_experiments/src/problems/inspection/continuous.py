import copy
import numpy as np
import random as rd

from isaaclab_experiments.src.mapping.utils import compute_dist
from isaaclab_experiments.src.mapping.continuous import ContinuousInflationMap

class ContinuousInspectionProblemState:

    def __init__(self,
        agent_pos           : tuple, 
        inf_map             : ContinuousInflationMap,
        special_actions     : dict, 
        actions_dict        : dict, 
        action_range        : list[tuple[float,float]],
        tasks_found         : dict, 
        inspection_counter  : dict, 
        max_inspection      : int, 
        max_inspection_dist : int | float, 
        visibility_radius   : int | float
    ):
        # All information here refers to the real world, except the map
        self.agent_pos = agent_pos  # (x,y) in meters
        self.map = inf_map

        self.special_actions = special_actions # default action space

        self.actions_dict = actions_dict # all actions tried
        self.action_range = action_range # in meters

        self.tasks_found = tasks_found
        self.max_inspection = max_inspection
        self.max_inspection_dist = max_inspection_dist
        self.inspection_counter = inspection_counter

        self.visibility_radius = visibility_radius # in meters

    # -------------------------------------------------
    # TASK UTILITIES
    # -------------------------------------------------

    def get_closest_visible_task(self):
        task_name, task_dist = None, np.inf
        for tname, tpos in self.tasks_found.items():
            if self.map.is_task_visible(tname):
                d = compute_dist(self.agent_pos, tpos)
                if d < task_dist:
                    task_name = tname
                    task_dist = d
            elif self.map.is_visible(self.agent_pos, tpos, self.visibility_radius):
                d = compute_dist(self.agent_pos, tpos)
                if d < task_dist:
                    task_name = tname
                    task_dist = d
        return task_name, task_dist

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------

    def step(self, action):
        reward = 0.0
        next_state = self.copy() # next state is initially a copy of the current state

        # -------------------------
        # INSPECTION ACTION
        # -------------------------
        # if action is the inspection action, calculate inspection reward
        if action == "X":
            task_name, task_dist = next_state.get_closest_visible_task()

            # - if there is no visible task, return next state with no reward
            if task_name is None:
                return next_state, reward, None, None

            # - if there is a visible task within the inspection distance, check if 
            # it can be inspected and calculate the reward
            if task_dist <= self.max_inspection_dist:
                if next_state.inspection_counter[task_name] < next_state.max_inspection:
                    reward += 1#/(min_dist + 1e-6)
                    next_state.inspection_counter[task_name] += 1

        # -------------------------
        # NAVIGATION ACTION
        # -------------------------
        # if action is a navigation action, calculate the new position
        else:
            pos = next_state.agent_pos

            # updating actions list
            if str(action) not in self.actions_dict:
                self.actions_dict[str(action)] = action

            # calculating new position
            dx, dy = action
            new_pos = (
                int(pos[0] + dx),
                int(pos[1] + dy),
            )
            # collision check via costmap
            if self.map.cost(new_pos) < self.map.eta * self.map.max_cost:
                next_state.agent_pos = new_pos

        return next_state, reward, None, None
    

    def sample_new_action(self):
        px, py = self.agent_pos
        max_global = np.max([self.action_range[0][1], self.action_range[1][1]])

        angle = np.random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)

        norm = np.hypot(dx, dy) + 1e-8
        dx /= norm
        dy /= norm

        step = self.map.resolution / 4.0
        dist = 0.0
        max_safe = 0.0

        cost_threshold = self.map.eta * self.map.max_cost
        while dist < max_global:

            nx = px + dx * dist
            ny = py + dy * dist

            mx, my = self.map.world_to_map(nx, ny)
            if not self.map.is_in_bounds((mx, my)):
                break

            cost = self.map.cost((nx, ny))
            if cost > cost_threshold:
                break

            if not self.map.is_visible((px, py), (nx, ny), self.visibility_radius):
                break

            max_safe = dist
            dist += step

        if max_safe <= self.map.resolution:
            return [0.0, 0.0]

        dist = np.random.uniform(self.map.resolution, max_safe)
        return [np.round(dist * dx, 2), np.round(dist * dy, 2)]

    # -------------------------------------------------
    # TERMINATION
    # -------------------------------------------------

    def is_final_state(self):
        # if all tasks were inspected, the game ended
        return all(
            c >= self.max_inspection
            for c in self.inspection_counter.values()
        )

    # -------------------------------------------------
    # TRANSITION / OBSERVATION (for planners)
    # -------------------------------------------------

    def get_trans_p(self, action):
        return [self.copy(), 1]

    def get_obs_p(self, action):
        return [self.get_observation(), 1]
    
    def get_obs_dist(self):
        # Observation distribution is not available to
        # this problem
        return NotImplemented

    # -------------------------------------------------
    # OBSERVATION MODEL
    # -------------------------------------------------

    def get_observation(self):
        """Get the state obsevation"""
        obs = []
        pos = self.agent_pos

        # Get tasks observation
        # - Task observation = list of [task name, x position, y position]
        for tname, tpos in self.tasks_found.items():
            if self.map.is_task_visible(tname):
                obs.append([tpos[0],tpos[1]])
            elif self.map.is_visible(pos, tpos, self.visibility_radius):
                obs.append([tpos[0],tpos[1]])
        return obs

    def observation_is_equal(self, obs1, obs2):
        if len(obs1) != len(obs2):
            return False
        for o1 in obs1:
            if o1 not in obs2:
                return False
        for o2 in obs2:
            if o2 not in obs1:
                return False
        return True

    # -------------------------------------------------
    # HASHING (important for planners)
    # -------------------------------------------------

    def hash_state(self, state=None):
        if state is None:
            res = self.map.resolution
            qx = int(self.agent_pos[0] / res)
            qy = int(self.agent_pos[1] / res)
        else:
            res = self.map.resolution
            qx = int(state.agent_pos[0] / res)
            qy = int(state.agent_pos[1] / res)
        return hash(str((qx, qy)))

    def hash_observation(self, obs=None):
        if obs is None:
            obs = self.get_observation()
        res = self.map.resolution
        qobs = [(int(o[0] / res), int(o[1] / res)) for o in obs]
        return hash(str(tuple(sorted(qobs))))

    # -------------------------------------------------
    # COPY
    # -------------------------------------------------

    def copy(self):
        return ContinuousInspectionProblemState(
            agent_pos=(self.agent_pos[0], self.agent_pos[1]),
            inf_map=self.map,
            special_actions=self.special_actions,
            actions_dict=self.actions_dict,
            action_range=self.action_range,
            tasks_found=copy.deepcopy(self.tasks_found),
            inspection_counter=copy.deepcopy(self.inspection_counter),
            max_inspection=self.max_inspection,
            max_inspection_dist=self.max_inspection_dist,
            visibility_radius=self.visibility_radius,
        )
    
class ContinuousInspectionProblem:

    navigation_radius = 3.0
    action_range = [
        (0.0, navigation_radius),   # x-axis
        (0.0, navigation_radius),   # y-axis
    ]

    special_actions = {
        "X": (0, 0),                # inspection
    }

    def __init__(
        self,
        env,
        map_size_w,
        resolution              : float=0.5,
        z_min                   : float=0.1,
        z_max                   : float=1.0,
        confirm_threshold       : int  =2,
        robot_radius            : float=0.25,
        inflation_radius        : float=0.6,
        cost_scaling_factor     : float=10.0,
        visibility_radius       : float=7.0,
        max_inspection          : int  =1,
        max_inspection_distance : float=2.0,
        tasks                   : dict={}
    ):

        self.map = ContinuousInflationMap(
            map_size_w,
            resolution,
            z_min,
            z_max,
            confirm_threshold,
            robot_radius,
            inflation_radius,
            cost_scaling_factor
        )

        self.memory_map = np.zeros(self.map.map_size, dtype=np.uint8)
        self.visibility_radius = visibility_radius

        self.tasks = {}
        for name in tasks:
            self.tasks[name] = tuple(
                env.scene[name].data.root_pos_w[0, 0:2].cpu().numpy()
            )

        self.tasks_found = {}
        self.inspection_counter = {}

        self.max_inspection = max_inspection
        self.max_inspection_distance = max_inspection_distance

        self.sample_index = len(self.tasks) * 10

        self.last_target_point = None
        self.last_target_dir = None
        self.last_vis_pos = None


    # -------------------------------------------------
    # TERMINATION
    # -------------------------------------------------

    def completed_all_tasks(self):
        return all(
            self.inspection_counter.get(t, 0) >= self.max_inspection
            for t in self.tasks
        )


    def reset(self):
        self.map.reset()
        self.memory_map[:] = 0

        self.tasks_found = {}
        self.inspection_counter = {}

        self.last_target_point = None
        self.last_target_dir = None


    # -------------------------------------------------
    # VISION / KNOWLEDGE
    # -------------------------------------------------

    def update_knowledge(self, agent_pos_w, lidar_readings, lidar_meshes):
        agent_pos_w = agent_pos_w[:2]
        # checking if agent's vision has changed significantly
        # - if it doesn't, do not update
        if self.last_vis_pos is not None:
            if compute_dist(agent_pos_w, self.last_vis_pos) < self.map.robot_radius_w:
                return

        self.last_vis_pos = agent_pos_w

        # updating map knowledge
        self.map.update_with_lidar(agent_pos_w, \
            lidar_readings, lidar_meshes=lidar_meshes, \
            max_dist=self.visibility_radius)

        # calculating the visibility area
        cx, cy = self.map.world_to_map(*agent_pos_w)
        r = int(self.visibility_radius / self.map.resolution)

        xmin = max(0, cx - r)
        xmax = min(self.map.map_size[0], cx + r + 1)
        ymin = max(0, cy - r)
        ymax = min(self.map.map_size[1], cy + r + 1)

        # updating mamory map
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                cell_w = self.map.map_to_world(x, y)
                if self.map.is_visible(agent_pos_w, cell_w, self.visibility_radius):
                    self.memory_map[x, y] = 1

        # detect tasks
        for tname, tpos in self.tasks.items():
            if self.map.is_task_visible(tname):
                if tname not in self.tasks_found:
                    print("Task found:", tname, 'at', tpos)
                    self.tasks_found[tname] = tpos
                    self.inspection_counter[tname] = 0
            elif self.map.is_visible(agent_pos_w, tpos, self.visibility_radius):
                if tname not in self.tasks_found:
                    print("Task found:", tname, 'at', tpos)
                    self.tasks_found[tname] = tpos
                    self.inspection_counter[tname] = 0


    # -------------------------------------------------
    # STATE
    # -------------------------------------------------

    def get_current_state(self, agent_pos):
        return ContinuousInspectionProblemState(
            agent_pos=agent_pos,
            inf_map=self.map,
            special_actions=self.special_actions,
            actions_dict=copy.deepcopy(self.special_actions),
            action_range=self.action_range,
            tasks_found=self.tasks_found.copy(),
            inspection_counter=self.inspection_counter.copy(), 
            max_inspection=self.max_inspection,
            max_inspection_dist=self.max_inspection_distance, 
            visibility_radius=self.visibility_radius,            
        )


    # -------------------------------------------------
    # UNKNOWN SPACE SAMPLING
    # -------------------------------------------------

    def get_unknown_positions(self):
        free_spaces = []
        for x in range(self.map.map_size[0]):
            for y in range(self.map.map_size[1]):
                if self.memory_map[x, y] == 0:
                    pos_w = self.map.map_to_world(x + 0.5, y + 0.5)
                    if self.map.cost(pos_w) < self.map.eta * self.map.max_cost:
                        free_spaces.append(pos_w)
        return free_spaces


    def sample_state(self, state):
        free_spaces = self.get_unknown_positions()
        sampled_state = self.get_current_state(state.agent_pos)

        while len(sampled_state.tasks_found) != len(self.tasks) and free_spaces:
            tpos = free_spaces.pop(rd.randrange(len(free_spaces)))
            task_key = "T" + str(self.sample_index)

            sampled_state.tasks_found[task_key] = tpos
            sampled_state.inspection_counter[task_key] = 0

            self.sample_index += 1

        return sampled_state


    # -------------------------------------------------
    # NAVIGATION
    # -------------------------------------------------

    def translate_actions2path(self, agent, action_sequence):
        translated_path = []
        pos = agent["pos"]
            
        for a in action_sequence:
            if str(a) in self.special_actions:
                dx, dy = self.special_actions[str(a)]
            else:
                dx, dy = a
            new_pos = (pos[0] + dx, pos[1] + dy)
            mnew_pos = self.map.world_to_map(*new_pos)

            if self.map.sdf(new_pos) > self.map.robot_radius_w and \
            self.map.is_in_bounds(mnew_pos):
                translated_path.append(new_pos)
            else:
                translated_path.append(pos)

        return translated_path


    # -------------------------------------------------
    # ACTION EXECUTION
    # -------------------------------------------------

    def compute_next_action(self, agent, path, action_sequence, planner_name):
        
        # -------------------------------------------------
        # SUPPORT FUNCTION
        # -------------------------------------------------
        def pop_step():
            if path:
                path.pop(0)
            if action_sequence:
                action_sequence.pop(0)

        # -------------------------------------------------
        # COMPUTING NEXT ACTION
        # -------------------------------------------------
        info = {"reward": 0}
        agent_pos = agent["pos"]

        # if no path is defined
        if (not path):
            if (self.last_target_dir is None) or (self.last_target_point is None):
                target_point = agent_pos
                target_dir = agent["heading"]
                self.last_target_point = target_point
                self.last_target_dir = target_dir
            else:
                target_point = self.last_target_point
                target_dir = self.last_target_dir 

        # otherwise, calculate the next point
        else:
            # === Handling task inspection ===
            if action_sequence[0] == 'X':
                target_point = self.last_target_point
                target_dir = self.last_target_dir 

                # Handle task inspection
                if self.tasks_found:
                    # getting closest task to inspect
                    tasks_dist, task_name = np.inf, None
                    for tname in self.tasks_found:
                        tpos = self.tasks_found[tname]
                        tmp_tasks_dist = compute_dist(agent['pos'], tpos)
                        if tmp_tasks_dist < tasks_dist:
                            tasks_dist = tmp_tasks_dist
                            task_name = tname

                    # checking inspection condition
                    print('Trying to inspect task',task_name,'at dist=',tasks_dist)
                    print('> Tasks inspection counter:',self.inspection_counter)
                    print('> Inspection done:',tasks_dist <= self.max_inspection_distance)
                    if tasks_dist <= self.max_inspection_distance:
                        # Increment inspection count
                        self.inspection_counter[task_name] += 1
                        if self.inspection_counter[task_name] <= self.max_inspection:
                            info['reward'] += 1

                    print('> Updated inspection counter:',self.inspection_counter)
                pop_step() 

            # === Handling navigation ===
            else:
                # checking if the robot is close to the current target position 
                # (reached the target position)
                current_target_pos = path[0]
                if compute_dist(agent_pos, current_target_pos) < self.map.robot_radius_w:
                    pop_step()
                
                # if there is still further points to reach, define the target point
                if len(path) > 0:
                    target_point = path[0]
                # otherwise, define it as the last target
                else:
                    target_point = self.last_target_point

                # updating the agent's  target direction/orientation
                # if the target point changed, re calculate the orientation
                if self.last_target_point != target_point:
                    target_dir = self.map.compute_orientation(
                        self.last_target_point,
                        target_point
                    )
                # otherwise, keep the same
                else:
                    target_dir = self.last_target_dir

        # updating the agent's action memory
        self.last_target_point = target_point
        self.last_target_dir = target_dir
        return target_point, target_dir, action_sequence, info