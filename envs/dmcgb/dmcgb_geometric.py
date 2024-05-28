

import numpy as np
import dm_env


dmcgb_geometric_modes= ['rotate_easy', 'rotate_hard', 'shift_easy', 'shift_hard', 'rotate_shift_easy', 'rotate_shift_hard'] 
CAMERA_MODES = ['fixed', 'track', 'trackcom', 'targetbody', 'targetbodycom']


class ShiftWrapper(dm_env.Environment):
    """Shifts camera by rotating its lookat point to push the agent towards the edges of the image"""
    def __init__(self, env, mode, seed):
        self._env = env
        self._random_state_shift = np.random.RandomState(seed)
        self._curr_cam_shift = np.zeros((3,3))
        self._start_shift_ind = 0
        self._num_starting_shifts = 100
        self._curr_cam_mode = CAMERA_MODES[self._env.physics.model.cam_mode[0]]
        self._shift_in_effect = "shift" in mode

        if self._shift_in_effect:
            self._get_corner_coordinates(mode)
            self._sample_cam_positions()
            self._random_state_shift.shuffle(self._all_cam_shifts) 



    def _get_corner_coordinates(self, mode):
        cam_edges = { 
            'walker': [10, -10, 10, -10], # Right, Left, Down, Up
            'cheetah': [10, -10, 10, -10], # Right, Left, Down, Up
            'cartpole': [8, -8, 10, -10], # Left, Right, Up, Down
            'finger': [6, -6, 12, -6], # Left, Right, Up, Down
            'ball_in_cup': [6, -6, 12, -6], # Left, Right, Up, Down
        }
        max_roll, min_roll, max_pitch, min_pitch = cam_edges[self._env._domain_name] 
        if "easy" in mode:
            max_roll /= 1.5
            min_roll /= 1.5
            max_pitch /= 1.5
            min_pitch /= 1.5
        self._four_corners = np.array([[0, min_roll, min_pitch], [0, max_roll, min_pitch],[0, max_roll, max_pitch],  [0, min_roll, max_pitch]]) 


    def _sample_cam_positions(self):
        # Interpolating rotations between four corners
        num_points_each_path = self._num_starting_shifts // 4
        positions = []
        for i in range(4):
            next_i = (i + 1) % 4 
            diff = self._four_corners[next_i] - self._four_corners[i]
            scale = diff / num_points_each_path
            sampled_positions = (np.arange(num_points_each_path)[..., np.newaxis] * scale) + self._four_corners[i]
            positions.append(sampled_positions)
        positions = np.concatenate(positions, axis=0)
        
        # Converting to rotation mats
        shifts =[]
        for rot_combo in positions:
            yaw, pitch, roll = rot_combo
            # Convert angles to radians
            yaw = np.radians(yaw)
            pitch = np.radians(pitch)
            roll = np.radians(roll)

            # Individual rotation matrices
            R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

            R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                [0, 1, 0],
                                [-np.sin(pitch), 0, np.cos(pitch)]])

            R_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

            # Combine rotation matrices
            rotation_matrix = np.dot(R_yaw, np.dot(R_pitch, R_roll))
            shifts.append(rotation_matrix)
        # Appending extra to fill required starting positions
        for i in range(self._num_starting_shifts - len(shifts)):
            shifts.append(shifts[i])
        # Store them
        self._all_cam_shifts = shifts


    def _set_cam_shift(self):
        # Set Camera Shift
        cam_xmat = np.reshape(self._env.physics.data.cam_xmat[0], (3,3))
        self._env.physics.data.cam_xmat[0] = np.dot(cam_xmat, self._curr_cam_shift).flatten() # Slide to view

    def reset(self):
        time_step = self._env.reset()
        if self._shift_in_effect:
            self._curr_cam_shift[:] = self._all_cam_shifts[self._start_shift_ind][:]
            self._start_shift_ind = (self._start_shift_ind + 1) % len(self._all_cam_shifts) # Next time new cam position
            self._set_cam_shift()
        return time_step 

    def step(self, action):
        time_step = self._env.step(action)
        if self._shift_in_effect:
            self._set_cam_shift()
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RotateWrapper(dm_env.Environment):
    """Rotates the frame by rotating the camera's yaw 360 deg"""
    def __init__(self, env, mode, seed):
        self._env = env
        self._rotate_in_effect = "rotate" in mode
        if self._rotate_in_effect:
            self._random_state_rot = np.random.RandomState(seed)
            self._start_rot_ind = 0
            self._num_starting_rots = 100 
            self._curr_cam_rot = np.zeros((3,3))
            self._get_rotation_angles(mode)
            self._random_state_rot.shuffle(self._all_cam_rots) 


    def _get_rotation_angles(self, mode):
        bound_angles = np.array([-180.0, 180.0])
        if "easy" in mode:
            bound_angles /= 2.0 
        min_angle, max_angle = bound_angles
        scale = (max_angle - min_angle) / self._num_starting_rots
        rot_angles = (np.arange(self._num_starting_rots) * scale) + min_angle
        # Converting to rotation mats
        rots =[]
        for yaw in rot_angles:
            yaw = np.radians(yaw)
            # Individual rotation matrices
            R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
            rots.append(R_yaw)
        # Appending extra to fill required starting positions
        for i in range(self._num_starting_rots - len(rots)):
            rots.append(rots[i])
        self._all_cam_rots = rots 


    def _set_cam_rot(self):
        # Set Camera Rotation
        cam_xmat = np.reshape(self._env.physics.data.cam_xmat[0], (3,3))
        self._env.physics.data.cam_xmat[0] = np.dot(cam_xmat, self._curr_cam_rot).flatten() # Slide to view


    def reset(self):
        time_step = self._env.reset()
        if self._rotate_in_effect:
            self._curr_cam_rot = self._all_cam_rots[self._start_rot_ind]
            self._start_rot_ind = (self._start_rot_ind + 1) % len(self._all_cam_rots) # Next time new cam position
            self._set_cam_rot()
        return time_step 


    def step(self, action):
        time_step = self._env.step(action)
        if self._rotate_in_effect:
            self._set_cam_rot()
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


