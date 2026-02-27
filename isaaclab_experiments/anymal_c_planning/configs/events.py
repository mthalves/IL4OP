
import torch
import time
import math

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs.manager_based_env import ManagerBasedEnv
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

##
# Planning function
##
import multiprocessing as mp

class OnlinePlanning(ManagerTermBase):

    available_planning_algorithms = {
        'astar': 'AStarPlanner',
        'pomcp': 'POMCP',
        'ibpomcp': 'IBPOMCP',
        'tbrhopomcp':'TBRhoPOMCP',
    }
    available_problems = {
        'inspection': 'InspectionProblem',
    }

    def __init__(self, cfg: EventTerm, env: ManagerBasedEnv):
        print(f"Initializing {self.__class__.__name__} with config: {cfg}")
        super().__init__(cfg, env)

        # Problem and planner initialization
        self.problem_env = self.get_problem(
            env,
            cfg.params["problem"]["name"],
            cfg.params["problem"]["args"]
        )

        self.planner = self.get_planner(
            cfg.params['planning_method']['name'],
            cfg.params['planning_method']['args']
        )

        # Suport variables for planning
        self.action_sequence = []
        self.action_history = []
        self.command_history = []
        self.path = []

        # Navigation 2D commands
        self.target_point, self.target_dir = (-1,-1), (-1, -1)

        # Support variables for log
        self.last_time2reason = 0.
        self.update = False

        # Launching planning
        self.plan_req_q = mp.Queue(maxsize=1)
        self.plan_res_q = mp.Queue(maxsize=1)

        self.planner_proc = mp.Process(
            target=self.planner_worker,
            args=(self.plan_req_q, self.plan_res_q),
            daemon=True,
        )
        self.planner_proc.start()

        self.planning_in_progress = False

    def get_problem(self, env, problem_name, args):
        if problem_name is None:
            print(f"Setting map generator class to default value.")
            self.problem_name = 'inspection'
        else:
            self.problem_name = problem_name.lower()
        ProblemClass = self.import_problem(self.problem_name)
        return ProblemClass(env,**args)
    
    def import_problem(self,problem_name):
        from importlib import import_module
        path = 'isaaclab_experiments.src.problem'
        module = import_module(path)
        try:
            method = getattr(module, self.available_problems[problem_name])
        except:
            print('The choosen map generator is not implemented:',problem_name)
            print('Loading default planning method: inspection')
            method = getattr(module, self.available_problems['inspection'])
        return method

    def get_planner(self, planning_method_name, planning_method_args):
        if planning_method_name is None:
            print(f"Setting planning class to default value.")
            self.pĺanner_name = 'astar'
        else:
            self.planner_name = planning_method_name.lower()
        PlanningClass = self.import_planner()
        if self.planner_name != 'astar':
            return PlanningClass(**planning_method_args)
        else:
            return PlanningClass(self.problem_env.map.map_size)

    def import_planner(self):
        from importlib import import_module
        path = 'isaaclab_experiments.src.planning_algorithms.'
        try:
            module = import_module(path+self.planner_name)
            method = getattr(module, self.available_planning_algorithms[self.planner_name])
        except:
            print('The choosen method is not implemented:',self.planner_name)
            print('Loading default planning method: astar')
            self.planner_name = 'astar'
            module = import_module(path+self.planner_name)
            method = getattr(module, self.available_planning_algorithms['astar'])
        return method

    def planner_worker(self, req_q, res_q):
        while True:
            msg = req_q.get()
            if msg is None:
                break

            agent, problem_env = msg
            action_sequence = self.planner.plan(agent, problem_env)

            if self.planner_name != 'astar':
                path = problem_env.translate_actions2path(agent, action_sequence)
            else:
                path = action_sequence

            res_q.put((action_sequence, path))

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None,
         planning_method: dict = {},
         problem: dict = {}, command_name: str = 'pose_commands',
         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
         lidar_cfg: SceneEntityCfg = SceneEntityCfg("lidar_sensor")):

        # Real world information
        robot = env.scene[robot_cfg.name]
        root_pos = robot.data.root_state_w[0, 0:2].cpu().numpy()
        robot_pos_w = (root_pos[0], root_pos[1])
        robot_heading = robot.data.heading_w[0].cpu().numpy()
        lidar_readings_w = env.scene[lidar_cfg.name].data.ray_hits_w[0]

        # Planning information
        robot_pos = self.problem_env.map.world_to_map(*robot_pos_w)
        current_state = self.problem_env.get_current_state(robot_pos)
        agent = {
            'name': 'Bob',
            'pos': robot_pos,
            'heading': robot_heading,
            'radius': 0.5,  # Assuming a fixed radius for the robot
            'visibility_radius': current_state.visibility_radius,
            'action_history': self.action_history,
        }

        # Updating knwoledge with real world information
        self.problem_env.update_knowledge(robot_pos_w, lidar_readings_w)

        # =============================
        # === PLANNING ===
        # =============================
        # if no action sequence, plan a new one
        self.update = (len(self.action_sequence) == 0 or self.planner_name == 'astar')
        print('Updating/Checking plan...')
        if self.update and not self.planning_in_progress:
            try:
                self.plan_req_q.put_nowait((agent, self.problem_env))
                self.planning_in_progress = True
            except:
                pass  # queue full → planner still busy

        # planning the action sequence
        print('Trying to get the action')
        if self.planning_in_progress and not self.plan_res_q.empty():
            self.action_sequence, self.path = self.plan_res_q.get()
            self.planning_in_progress = False

            print(self.action_sequence)
            for action in self.action_sequence:
                self.action_history.append(action)

            print("Action sequence received:", self.action_sequence)
            print("Translated path:", self.path)

        # ===========================
        # === COMMAND SETTING ===
        # ===========================
        # defining the next action: target point and direction
        self.target_point, self.target_dir, self.action_sequence, self.info = \
            self.problem_env.compute_next_action(\
                agent, self.path, self.action_sequence, self.planner_name)
        self.last_reward = self.info['reward']

        # -- visualizing planning map
        self.problem_env.map.visualize(agent, self.path,\
             self.problem_env.tasks_found, self.problem_env.memory_map)

        # setting the command to the robot
        cmd = env.command_manager._terms[command_name]
        cmd.pos_command_w[:, 0] = float(self.target_point[0])
        cmd.pos_command_w[:, 1] = float(self.target_point[1])
        cmd.pos_command_w[:, 2] = robot.data.default_root_state[0, 2]

        target_direction = torch.tensor(self.target_dir, device=env.device)
        flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

        curr_to_target = wrap_to_pi(target_direction - robot.data.heading_w[env_ids]).abs()
        curr_to_flipped_target = wrap_to_pi(flipped_target_direction - robot.data.heading_w[env_ids]).abs()

        cmd.heading_command_w[:] = torch.where(
            curr_to_target < curr_to_flipped_target,
            target_direction,
            flipped_target_direction,
        )

        target_vec_b = cmd.pos_command_w - robot.data.root_pos_w[:, :3]
        cmd.pos_command_b[:] = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_vec_b)
        cmd.heading_command_b[:] = wrap_to_pi(cmd.heading_command_w - robot.data.heading_w)

    def reset(self, env_ids: torch.Tensor):
        print('Reseting planner and map')
        self.path, self.path_w = [], []
        self.target_point, self.target_dir = (-1,-1), (-1, -1)
        self.action_history = []
        self.action_sequence = []

        if hasattr(self.planner, 'reset'):
            self.planner.reset()
        if hasattr(self.problem_env, 'reset'):
            self.problem_env.reset()

    def get_map_size(self): # function used outside to set the came
        return self.problem_env.map.map_size_w

def reset_command( 
 env: ManagerBasedEnv, env_ids: torch.Tensor | None,
 command_name: str = 'pose_commands', robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    

    robot = env.scene[robot_cfg.name]
    root_pos = robot.data.default_root_state[0]

    cmd = env.command_manager._terms[command_name]
    cmd.pos_command_w[:, 0] = float(root_pos[0])
    cmd.pos_command_w[:, 1] = float(root_pos[1])
    cmd.pos_command_w[:, 2] = float(root_pos[2])

    cmd.heading_command_w[:] = robot.data.heading_w[0]

    target_vec_b = cmd.pos_command_w - robot.data.root_pos_w[:, :3]
    cmd.pos_command_b[:] = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_vec_b)
    cmd.heading_command_b[:] = wrap_to_pi(cmd.heading_command_w - robot.data.heading_w)

@configclass
class EventCfg:
    """Configuration for planning."""

    planning = EventTerm(
        func=OnlinePlanning,
        mode="interval",
        interval_range_s=(1.,1.),
        params={
            "planning_method":{
                #"name":"astar",
                #"args":{},
                #"name":"pomcp",
                #"args":{
                #    "max_depth":20,
                #    "max_it":1000,
                #    "kwargs":{},
                #},
                "name":"ibpomcp",
                "args":{
                    "max_depth":20,
                    "max_it":1000,
                    "kwargs":{},
                },
                #"name":"tbrhopomcp",
                #"args":{
                #    "max_depth":20,
                #    "max_it":1000,
                #    "kwargs":{'time_budget':2.0,'smallbag_size':10},
                #},
            },
            "problem": {
                "name":"inspection",
                "args": {
                    "map_size_w": (17, 17),
                    "z_min":0.1, "z_max":1.0,
                    "resolution":               1., 
                    "confirm_threshold":        2,
                    "inscribed_radius":         0.1, 
                    "inflation_radius":         0.0, 
                    "cost_scaling_factor":      0.5,
                    "visibility_radius":        5.,
                    "max_inspection":           1,
                    "max_inspection_distance":  2.9,
                    "tasks": ['box_1','box_2','box_3'],
                },
            },
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
            "lidar_cfg": SceneEntityCfg("lidar_sensor"),
        },
    )

    reset_game = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

    reset_command = EventTerm(
        func=reset_command,
        mode="reset",
        params={
            "command_name": 'pose_commands',
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )