import gym
from gym import spaces
import numpy as np
import copy
from collections import defaultdict
import os

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *

from post_processing import plot_video_with_sphere_cylinder
from MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class Environment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        final_time,
        n_elem,
        num_steps_per_update,
        number_of_control_points,
        alpha,
        beta,
        target_position,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=2.0e-5,
        mode=1,
        num_obstacles=0,
        GENERATE_NEW_OBSTACLES=True,
        max_episode_steps=1000,
        *args,
        **kwargs,
    ):
        super(Environment, self).__init__()

        # Time setup
        self.dim = 3.0
        self.StatefulStepper = PositionVerlet()
        self.n_elem = n_elem
        self.final_time = final_time
        self.h_time_step = sim_dt
        self.max_episode_steps = max_episode_steps
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        print("Total steps", self.total_steps)

        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Control + learning steps
        self.number_of_control_points = number_of_control_points
        self.alpha, self.beta = alpha, beta
        self.target_position = target_position
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        # Discrete torques (for DQN)
        self.torque_patterns = [
            np.array([-1.0, 0.0, 0.0, 0.0]),  # left
            np.array([+1.0, 0.0, 0.0, 0.0]),  # right
            np.array([0.0, -1.0, 0.0, 0.0]),  # down
            np.array([0.0, +1.0, 0.0, 0.0]),  # up
            np.array([0.0, 0.0, 0.0, 0.0]),   # none
        ]
        self.action_space = spaces.Discrete(len(self.torque_patterns))

        # Observation space
        self.obs_state_points = 10
        num_points = int(self.n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(self.n_elem + 1)[0::num_points])

        self.N_OBSTACLE = num_obstacles
        self.num_new_obstacle = 0
        self.N_OBSTACLE_ALL = self.N_OBSTACLE + self.num_new_obstacle
        self.number_of_points_on_cylinder = 5

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                num_rod_state * 3
                + 4  # vel norm + dir
                + 3  # sphere pos
                + self.N_OBSTACLE_ALL * self.number_of_points_on_cylinder * 3,
            ),
            dtype=np.float64,
        )

        # Other physics
        self.mode = mode
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING
        self.time_tracker = np.float64(0.0)
        self.acti_diff_coef = kwargs.get("acti_diff_coef", 9e-1)
        self.acti_coef = kwargs.get("acti_coef", 1e-1)
        self.max_rate_of_change_of_activation = kwargs.get(
            "max_rate_of_change_of_activation", 0.1
        )
        self.E = kwargs.get("E", 1e7)
        self.NU = kwargs.get("NU", 10)

        # Obstacles
        self.filename_obstacles = kwargs.get("filename_obstacles", "new_obstacles.npz")

        if GENERATE_NEW_OBSTACLES:
            self.obstacle_direction = [None for _ in range(self.N_OBSTACLE)]
            self.obstacle_normal = [None for _ in range(self.N_OBSTACLE)]
            self.obstacle_length = [1.0 for _ in range(self.N_OBSTACLE)]
            self.obstacle_radii = [0.03 for _ in range(self.N_OBSTACLE)]
            self.obstacle_start = [None for _ in range(self.N_OBSTACLE)]

            # Spread in X,Z and randomize sizes
            nest_start_pos_x, nest_end_pos_x = -0.8, 0.8
            nest_start_pos_z, nest_end_pos_z = -0.8, 0.8

            for i in range(self.N_OBSTACLE):
                # Random orientation
                alpha = np.random.uniform(0, np.pi)
                beta = np.random.uniform(0, 2*np.pi)
                direction = np.array([
                    np.cos(alpha) * np.cos(beta),
                    np.sin(alpha),
                    np.cos(alpha) * np.sin(beta),
                ])
                direction /= np.linalg.norm(direction)

                normal = np.cross(direction, np.array([0,1,0]))
                if np.linalg.norm(normal) < 1e-6:
                    normal = np.cross(direction, np.array([0,0,1]))
                normal /= np.linalg.norm(normal)

                self.obstacle_direction[i] = direction
                self.obstacle_normal[i] = normal

                # Start pos (spread in X,Z)
                start = np.zeros((3,))
                start[0] = np.random.uniform(nest_start_pos_x, nest_end_pos_x)
                start[1] = self.target_position[1] - (0.5 * self.obstacle_length[i] * direction)[1]
                start[2] = np.random.uniform(nest_start_pos_z, nest_end_pos_z)
                self.obstacle_start[i] = start

                # Random length/radius
                self.obstacle_length[i] = np.random.uniform(0.4, 1.2)
                self.obstacle_radii[i] = np.random.uniform(0.02, 0.06)

            save_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(save_folder, exist_ok=True)
            np.savez(
                os.path.join(save_folder, self.filename_obstacles),
                N_OBSTACLE=self.N_OBSTACLE,
                obstacle_direction=self.obstacle_direction,
                obstacle_normal=self.obstacle_normal,
                obstacle_length=self.obstacle_length,
                obstacle_radii=self.obstacle_radii,
                obstacle_start=self.obstacle_start,
                allow_pickle=True,
            )
        else:
            data = np.load("data/" + self.filename_obstacles, allow_pickle=True)
            self.N_OBSTACLE = data["N_OBSTACLE"]
            self.obstacle_direction = data["obstacle_direction"]
            self.obstacle_normal = data["obstacle_normal"]
            self.obstacle_length = data["obstacle_length"]
            self.obstacle_radii = data["obstacle_radii"]
            self.obstacle_start = data["obstacle_start"]

    # Callbacks
    class ArmMuscleBasisCallBack(CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["com"].append(system.compute_position_center_of_mass())

    class RigidSphereCallBack(CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["radius"].append(copy.deepcopy(system.radius))
                self.callback_params["com"].append(system.compute_position_center_of_mass())

    class RigidCylinderCallBack(CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["com"].append(system.compute_position_center_of_mass())

    # reset()
    def reset(self):
        """Reset: create arm, target, obstacles, callbacks"""
        self.simulator = BaseSimulator()

        # Arm
        n_elem = self.n_elem
        start = np.zeros((3,))
        start[1] = self.target_position[1]
        direction = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        density, nu, E, poisson_ratio = 1000, self.NU, self.E, 0.5
        base_length, radius_tip, radius_base = 1.0, 0.05, 0.05
        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)

        self.shearable_rod = CosseratRod.straight_rod(
            n_elem, start, direction, normal, base_length,
            base_radius=radius_along_rod,
            density=density, nu=nu, youngs_modulus=E, poisson_ratio=poisson_ratio,
        )
        self.simulator.append(self.shearable_rod)

        # Sphere
        target_position = self.target_position
        self.sphere = Sphere(center=target_position, base_radius=0.05, density=1000)
        self.simulator.append(self.sphere)

        # Obstacles
        self.obstacle = [None for _ in range(self.N_OBSTACLE)]
        self.obstacle_histories = [defaultdict(list) for _ in range(self.N_OBSTACLE)]
        self.obstacle_states = np.zeros(
            (self.number_of_points_on_cylinder * self.N_OBSTACLE, 3)
        )

        for i in range(self.N_OBSTACLE):
            self.obstacle[i] = Cylinder(
                start=self.obstacle_start[i],
                direction=self.obstacle_direction[i],
                normal=self.obstacle_normal[i],
                base_length=self.obstacle_length[i],
                base_radius=self.obstacle_radii[i],
                density=1000,
            )

            # RL state points
            self.obstacle_states[
                i * self.number_of_points_on_cylinder : (i + 1) * self.number_of_points_on_cylinder,
                :,
            ] = (
                self.obstacle_start[i].reshape(3, 1)
                + self.obstacle_direction[i].reshape(3, 1)
                * np.linspace(0, self.obstacle_length[i], self.number_of_points_on_cylinder)
            ).T

            # For postproc
            self.obstacle_histories[i]["radius"] = self.obstacle_radii[i]
            self.obstacle_histories[i]["height"] = self.obstacle_length[i]
            self.obstacle_histories[i]["direction"] = self.obstacle_direction[i].copy()

            n_elem_for_plotting = 10
            position_collection_for_plotting = np.zeros((3, n_elem_for_plotting))
            end = self.obstacle_start[i] + self.obstacle_direction[i] * self.obstacle_length[i]
            for k in range(0, 3):
                position_collection_for_plotting[k, ...] = np.linspace(
                    self.obstacle_start[i][k], end[k], n_elem_for_plotting
                )
            self.obstacle_histories[i]["position_plotting"] = position_collection_for_plotting.copy()

            self.simulator.append(self.obstacle[i])

            self.simulator.constrain(self.obstacle[i]).using(
                OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
            )
            self.simulator.connect(self.shearable_rod, self.obstacle[i]).using(
                ExternalContact, k=8e4, nu=4.0
            )

        # Muscle torques
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction="normal",
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction="binormal",
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_tangent_dir = defaultdict(list)
        self.spline_points_func_array_tangent_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_tangent_dir,
            muscle_torque_scale=self.beta,
            direction="tangent",
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_tangent_dir,
        )

        # Callbacks
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.post_processing_dict_rod = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                self.ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_sphere = defaultdict(list)
            self.simulator.collect_diagnostics(self.sphere).using(
                self.RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

            for i in range(self.N_OBSTACLE):
                self.simulator.collect_diagnostics(self.obstacle[i]).using(
                    self.RigidCylinderCallBack,
                    step_skip=self.step_skip,
                    callback_params=self.obstacle_histories[i],
                )

        # Finalize
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # Trackers
        state = self.get_state()
        self.on_goal, self.current_step, self.time_tracker = 0, 0, np.float64(0.0)
        self.previous_action = None
        return state

    # get_state()
    def get_state(self):
        rod_state = self.shearable_rod.position_collection
        r_s_a, r_s_b, r_s_c = rod_state

        num_points = int(self.n_elem / self.obs_state_points)
        rod_compact_state = np.concatenate(
            (
                r_s_a[0: len(r_s_a) + 1: num_points],
                r_s_b[0: len(r_s_b) + 1: num_points],
                r_s_c[0: len(r_s_b) + 1: num_points],
            )
        )

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0.0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()
        obstacle_data = self.obstacle_states.flatten()

        return np.concatenate(
            (
                rod_compact_state,
                rod_compact_velocity_norm,
                rod_compact_velocity_dir,
                sphere_compact_state,
                obstacle_data,
            )
        )

    # step()
    def step(self, action):
        if isinstance(action, (int, np.integer)):
            action = self.torque_patterns[action]

        self.spline_points_func_array_normal_dir[:] = action[: self.number_of_control_points]
        self.spline_points_func_array_binormal_dir[:] = action[self.number_of_control_points:]
        self.spline_points_func_array_tangent_dir[:] = np.zeros(self.number_of_control_points)

        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        self.current_step += 1
        state = self.get_state()

        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )
        reward = -dist**2
        if np.isclose(dist, 0.0, atol=0.05).all():
            reward += 1.5
        elif np.isclose(dist, 0.0, atol=0.1).all():
            reward += 0.5

        done = self.current_step >= self.total_learning_steps or self.current_step >= self.max_episode_steps

        if _isnan_check(state):
            print(" NaN detected in state, terminating")
            reward = -100
            state[np.argwhere(np.isnan(state))] = self.state_buffer[
                np.argwhere(np.isnan(state))
            ]
            done = True

        self.state_buffer = state
        self.previous_action = action

        return state, reward, done, {"ctime": self.time_tracker}

    # post_processing()
    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):
        """Make video (rod + sphere + obstacles)"""
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            plot_video_with_sphere_cylinder(
                [self.post_processing_dict_rod],
                self.obstacle_histories,
                [self.post_processing_dict_sphere],
                video_name=filename_video,
                fps=self.rendering_fps,
                step=10,
                vis2D=True,
                vis3D=True,
                **kwargs,
            )

            if SAVE_DATA:
                save_folder = os.path.join(os.getcwd(), "data")
                os.makedirs(save_folder, exist_ok=True)

                # Rod positions → elemental midpoints
                position_rod = np.array(self.post_processing_dict_rod["position"])
                position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

                np.savez(
                    os.path.join(save_folder, "arm_data.npz"),
                    position_rod=position_rod,
                    radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                    n_elems_rod=self.shearable_rod.n_elems,
                    position_sphere=np.array(self.post_processing_dict_sphere["position"]),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
                )

                np.savez(
                    os.path.join(save_folder, "obstacle_data.npz"),
                    obstacle_history=self.obstacle_histories,
                    allow_pickle=True,
                )
        else:
            raise RuntimeError(
                "Enable COLLECT_DATA_FOR_POSTPROCESSING=True to save videos or data."
            )
