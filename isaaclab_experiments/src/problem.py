import numpy as np
import operator
import random as rd

from isaaclab_experiments.src.map import InflationMap, compute_dist, bresenham

class InspectionProblemState:

    def __init__(self, 
        agent_pos           : tuple, 
        inf_map             : InflationMap,
        actions_dict        : dict, 
        tasks_found         : dict, 
        inspection_counter  : dict, 
        max_inspection      : int, 
        max_inspection_dist : int | float, 
        visibility_radius   : int | float
    ):
        self.agent_pos = agent_pos
        self.map = inf_map
        
        self.actions_dict = actions_dict
        self.actions = [act for act in self.actions_dict]

        self.tasks_found = tasks_found
        self.max_inspection = max_inspection
        self.max_inspection_dist = max_inspection_dist
        self.inspection_counter = inspection_counter

        self.visibility_radius = visibility_radius

    def get_closest_visible_task(self, state):
        task_name, task_dist = None, np.inf

        # finding the closest task
        for tname in state.tasks_found:
            tpos = state.tasks_found[tname]
            task_is_visible = self.map.is_visible(state.agent_pos,tpos, self.visibility_radius)
            if task_is_visible:
                tmp_task_dist = compute_dist(state.agent_pos,tpos)
                if tmp_task_dist < task_dist:
                    task_dist = tmp_task_dist
                    task_name = tname
        
        return task_name, task_dist
        
    def step(self, action):
        reward = 0.
        next_state = self.copy() # next state is initially a copy of the current state

        # if action is the inspection action, calculate inspection reward
        if action == 'X':
            task_name, task_dist = self.get_closest_visible_task(next_state)
                        
            # - if there is no visible task, return next state with no reward
            if task_name is None:
                return next_state, reward, None, None
            
            # - if there is a visible task within the inspection distance, check if 
            # it can be inspected and calculate the reward
            if task_dist <= self.max_inspection_dist:
                if next_state.inspection_counter[task_name] < next_state.max_inspection:
                    reward += 1#/(min_dist + 1e-6)
                    next_state.inspection_counter[task_name] += 1

        # if action is a navigation action, calculate the new position
        else:
            pos = next_state.agent_pos
            new_pos = (\
                int(pos[0]+self.actions_dict[action][0]),\
                int(pos[1]+self.actions_dict[action][1]))
            if self.map.is_in_bounds(new_pos) and self.map.is_free_space(new_pos) \
            and new_pos not in self.tasks_found.values():
                next_state.agent_pos = new_pos
        return next_state, reward, None, None
    
    def is_final_state(self):
        # if all tasks were inspected, the game ended
        return all([c > 0 for c in self.inspection_counter.values()])

    def get_trans_p(self,action):
        return [self.copy(),1]
    
    def get_obs_p(self,action):
        return [self.get_observation(),1]
        
    def get_observation(self):
        """Get the state obsevation"""
        obs = []
        pos = self.agent_pos

        # Get tasks observation
        # - Task observation = list of [task name, x position, y position]
        for tname in self.tasks_found:
            # close 5m
            tpos = self.tasks_found[tname]
            if self.map.is_visible(pos, tpos, self.visibility_radius):
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

    def hash_state(self):
        return hash(str((self.agent_pos[0],self.agent_pos[1])))
    
    def hash_observation(self):
        obs = self.get_observation()
        return hash(str(obs))
    
    def copy(self):
        copied_state = InspectionProblemState(
            (self.agent_pos[0], self.agent_pos[1]),
            self.map,
            self.actions_dict,
            self.tasks_found.copy(),
            self.inspection_counter.copy(),
            self.max_inspection,
            self.max_inspection_dist,
            self.visibility_radius
        )
        return copied_state

class InspectionProblem:

    navigation_actions  = ['N', 'S', 'W', 'E']
    task_actions        = ['X']

    actions_dict = {
        'N':( 0, 1), 'W':(-1, 0), \
        'E':( 1, 0), 'S':( 0,-1), \
        'X':( 0, 0)
    }
    
    def __init__(
        self, env,
        map_size_w,
        resolution              :float=0.25,
        z_min                   :float=0.1,
        z_max                   :float=1.0,
        confirm_threshold       :int  =2,
        inscribed_radius        :float=0.25,
        inflation_radius        :float=0.6,
        cost_scaling_factor     :float=10.0,
        visibility_radius       :float=7.,
        max_inspection          :int  =1,
        max_inspection_distance :float=2.,
        tasks                   :dict ={}
    ):
        # initializing environment map
        self.map = InflationMap(
            map_size_w,resolution,z_min,z_max,confirm_threshold,
            inscribed_radius,inflation_radius,cost_scaling_factor
        )
        self.goal = self.map.world_to_map(*(15,3)) # value for testing purposes with astar
        self.memory_map  = np.zeros_like(self.map.obstacle_map)
        self.visibility_radius_w = visibility_radius
        self.visibility_radius = self.visibility_radius_w/self.map.resolution

        # real tasks
        self.tasks = {}
        for name in tasks:
            self.tasks[name] = tuple(env.scene[name].data.root_pos_w[0, 0:2].cpu().numpy())

        # tasks found during planning
        self.tasks_found = {}
        self.inspection_counter  = {}
        self.max_inspection = max_inspection
        self.max_inspection_distance_w = max_inspection_distance
        self.max_inspection_distance = max_inspection_distance/self.map.resolution

        self.sample_index = len(self.tasks)*10

        self.last_target_point, self.last_target_dir = None, None

    def completed_all_tasks(self):
        completed = []
        for task_name in self.tasks:
            if task_name in self.inspection_counter:
                completed.append(self.inspection_counter[task_name] >= self.max_inspection)
            else:
                completed.append(False)
        return all(completed)

    def reset(self):
        self.map.reset()
        self.memory_map  = np.zeros_like(self.map.obstacle_map)

        self.tasks_found = {}
        self.inspection_counter  = {}
        self.sample_index = len(self.tasks)*10

        self.last_target_point, self.last_target_dir = None, None
    
    ###
    ### PLANNING METHODS
    ###
    def update_knowledge(self, agent_pos_w, lidar_readings):
        # updating map knowledge
        self.map.update_with_lidar(agent_pos_w, lidar_readings, max_dist=self.visibility_radius)
        self.map.compute_cost_map()

        # saving visited positions
        agent_pos = self.map.world_to_map(*agent_pos_w[:2])
        current_state = self.get_current_state(agent_pos)

        for x in range(self.map.map_size[0]):
            for y in range(self.map.map_size[1]):
                if current_state.map.is_visible(agent_pos, (x,y), self.visibility_radius):
                    self.memory_map[x,y] = 1

        # updating task knowledge
        for tname in self.tasks:
            tpos = self.map.world_to_map(*self.tasks[tname])
            t_is_visible = current_state.map.is_visible(agent_pos, tpos, self.visibility_radius)
            if t_is_visible:
                if tname not in self.tasks_found:
                    print('Task found:', tname, 'at', tpos, 'from', agent_pos)
                    self.tasks_found[tname] = tpos
                    self.inspection_counter[tname] = 0

    def get_current_state(self, agent_pos):
        state = InspectionProblemState(\
            (agent_pos[0],agent_pos[1]), 
            self.map, 
            self.actions_dict, 
            self.tasks_found.copy(),
            self.inspection_counter.copy(), 
            self.max_inspection, 
            self.max_inspection_distance,
            self.visibility_radius
        )
        return state

    def get_unknown_positions(self):
        free_spaces = []
        for x in range(self.map.map_size[0]):
            for y in range(self.map.map_size[1]):
                if self.memory_map[x,y] == 0 and \
                self.map.cost_map[x,y] < 0.95*self.map.max_cost:
                    free_spaces.append((x,y))
        return free_spaces
    
    def sample_state(self, state):
        # collecting free spaces
        free_spaces = self.get_unknown_positions()

        # initialising new state
        sampled_state = self.get_current_state(state.agent_pos)

        # sampling unknown information
        # -> tasks positions
        while len(sampled_state.tasks_found) != len(self.tasks) and len(free_spaces) > 0:
            pos_index = rd.choice(range(len(free_spaces)))
            tpos = free_spaces.pop(pos_index)

            task_key = 'T'+str(self.sample_index)
            sampled_state.tasks_found[task_key] = tpos
            sampled_state.inspection_counter[task_key] = 0
            self.sample_index += 1
                
        return sampled_state
    
    def translate_actions2path(self, agent, action_sequence):
        translated_path = []
        agent_pos = agent['pos']
        
        if len(action_sequence) == 0:
            return translated_path
        
        # actions mod
        for a in action_sequence:
            translated_path.append(self.actions_dict[a])

        # translating first action to map coordinates based on agent position
        # - if agent is trying to walk through an obstacle, keep the current position
        # - else, translate the action to the next position
        new_pos = tuple(map(operator.add, translated_path[0], agent_pos))
        if self.map.cost_map[int(new_pos[0]), int(new_pos[1])] < 0.7*self.map.max_cost:
            translated_path[0] = new_pos
        else:
            translated_path[0] = agent_pos

        # translating the rest of the path following the same rule
        for i in range(1,len(translated_path)):
            new_pos = tuple(map(operator.add, translated_path[i-1], translated_path[i]))
            if self.map.cost_map[int(new_pos[0]), int(new_pos[1])] < 0.7*self.map.max_cost:
                translated_path[i] = new_pos
            else:
                translated_path[i] = translated_path[i-1]

        return translated_path
    
    def compute_next_action(self, agent, path, action_sequence, planner_name):
        # Helper: remove the current step from path and action sequence
        def pop_step():
            if path: path.pop(0)
            if action_sequence: action_sequence.pop(0)

        info = {'reward':0}

        # === Handle non-astar planners ===
        if planner_name != 'astar':
            current_action = action_sequence[0]

            if current_action in self.navigation_actions:
                # Move forward if agent reached the target point
                if agent['pos'] == tuple(map(int, path[0])) and \
                compute_dist(agent['pos'], path[0]) <= 0.1*self.map.resolution:
                    pop_step()
            else:
                # Handle task inspection
                if self.tasks_found:
                    tasks_dist, task_name = np.inf, None
                    for tname in self.tasks_found:
                        tpos = self.tasks_found[tname]
                        tmp_tasks_dist = compute_dist(agent['pos'], tpos)
                        if tmp_tasks_dist < tasks_dist:
                            tasks_dist = tmp_tasks_dist
                            task_name = tname

                    print('Trying to inspect task',task_name,'at dist=',tasks_dist)
                    print('> Tasks inspection counter:',self.inspection_counter)
                    print('> Inspection done:',tasks_dist <= self.max_inspection_distance)
                    if tasks_dist <= self.max_inspection_distance:
                        # Increment inspection count
                        self.inspection_counter[task_name] += 1
                        info['reward'] += 1
                        # Remove action if fully inspected
                        if self.inspection_counter[task_name] >= self.max_inspection:
                            pop_step()
                    else:
                        pop_step()  # Too far, discard action
                else:
                    pop_step()  # No tasks left, discard action

        # === Handle astar planner ===
        else:
            if agent['pos'] == tuple(map(int, path[0])):
                pop_step()

        # === Compute target point and orientation ===
        if not path:
        # - No path and no last target
            if self.last_target_point is None:
                # A. path navigation point
                target_point = self.map.map_to_world(*(\
                    agent['pos'][0]+self.map.resolution/2.,\
                    agent['pos'][1]+self.map.resolution/2.))
                
                # A. robot desired orientation
                target_dir = agent['heading']

                # A. updating current point and dir
                self.last_target_point = agent['pos']
                self.last_target_dir = target_dir

        # - No path but last target
            else:
                # B. path navigation point
                target_point = self.map.map_to_world(*(\
                    self.last_target_point[0]+self.map.resolution/2.,\
                    self.last_target_point[1]+self.map.resolution/2.))
                # B. robot desired orientation
                target_dir = self.last_target_dir

        # - Path and last target
        else:
            # C. path navigation point
            target_point = self.map.map_to_world(*(\
                path[0][0]+(self.map.resolution/2.),\
                path[0][1]+(self.map.resolution/2.)))

            # C. robot desired orientation
            if self.last_target_point != path[0]:
                if self.last_target_point is None:
                    self.last_target_point = agent['pos']
                target_dir = self.map.compute_orientation(self.last_target_point, path[0])
            else:
                target_dir = self.last_target_dir

            # C. updating current point and dir
            self.last_target_point = path[0]
            self.last_target_dir = target_dir

        return target_point, target_dir, action_sequence, info
