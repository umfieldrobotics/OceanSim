# Omniverse import
import numpy as np
from pxr import Gf
import omni.kit.commands
import omni.graph.core as og
import carb

# Isaac sim import
from isaacsim.core.api.sensors import BaseSensor
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.sensors.physx import _range_sensor

# Custom import
from isaacsim.oceansim.utils.MultivariateNormal import MultivariateNormal


class DVLsensor:
    def __init__(self,
                 name: str = "DVL",
                 elevation:float = 22.5, # deg
                 rotation: float = 45, # deg
                 vel_cov = 0,
                 depth_cov = 0,
                 min_range: float = 0.1,
                 max_range: float = 100,
                 num_beams_out_range_threshold: int = 2,
                 freq: int = None, # Hz
                 freq_bound: tuple[int] = [5, 100], # Hz
                 freq_dependenet_range_bound: tuple[float] = [7.5, 50.0], # m
                 sound_speed: float = 1500, # m/s
                 ):
        """Initialize a DVL sensor with configurable beam geometry and operating parameters.

        Args:
            name (str): Identifier for the sensor. Defaults to "DVL".
            elevation (float): Beam elevation angle from horizontal in degrees. Defaults to 22.5°.
            rotation (float): Beam rotation about Z-axis in degrees. Defaults to 45° (Janus configuration).
            vel_cov (float): Velocity measurement noise covariance. Defaults to 0 (no noise).
            depth_cov (float): Depth measurement noise covariance. Defaults to 0 (no noise).
            min_range (float): Minimum valid range in meters. Defaults to 0.1m.
            max_range (float): Maximum valid range in meters. Defaults to 100m.
            num_beams_out_range_threshold (int): Number of lost beams before declaring dropout. Defaults to 2.
            freq (int, optional): Fixed operating frequency in Hz. If None, uses adaptive frequency. Defaults to None.
            freq_bound (tuple[int]): (min_freq, max_freq) for adaptive operation. Defaults to (5, 100)Hz.
            freq_dependenet_range_bound (tuple[float]): (min_range, max_range) for frequency adaptation. Defaults to (7.5, 50.0)m.
            sound_speed (float): Speed of sound in water in m/s. Defaults to 1500m/s.
        """


        self._name = name

        # DVL configuration params
        self._elevation = elevation
        self._rotation = rotation
        self._min_range = min_range
        self._max_range = max_range

        # DVL noise params
        self._mvn_vel = MultivariateNormal(4)
        self._mvn_vel.init_cov(vel_cov)
        self._mvn_dep = MultivariateNormal(4)
        self._mvn_dep.init_cov(depth_cov)
        
        sinElev = np.sin(np.deg2rad(self._elevation))
        cosElev = np.cos(np.deg2rad(self._elevation))
        self._transform = np.array([[1/(2*sinElev), 0, -1/(2*sinElev), 0],
                                    [0, 1/(2*sinElev), 0, -1/(2*sinElev)],
                                    [1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev)]
                                    ])

        # sensor dropout related params
        self._num_beams_out_range_threshold = num_beams_out_range_threshold
        
        # Realistic DVL frequency dependent params
        self._user_static_freq_flag = False
        if freq is not None:
            self._user_static_freq_flag = True
            self._dt = 1/freq
        else:
            self._freq_bound = freq_bound
            self._freq_dependent_range_bound = freq_dependenet_range_bound
            self._sound_speed = sound_speed

        # Initialization 
        self._rigid_body_path = None
        self._beam_paths = []
        self._elapsed_time_vel = 0.0
        self._elapsed_time_depth = 0.0

        
        

    def attachDVL(self, 
                  rigid_body_path:str, 
                  position = None,
                  translation = None,
                  orientation = None
                  ):
        
        """Attach the DVL sensor to a rigid body in the simulation.
        ..note::
            This function will create a BaseSensor object under the parent rigid body prim and create 4 LightBeamSensors.  
        
        Args:
            rigid_body_path (str): USD path to the parent rigid body prim.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                    Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
        Raises:
            Exception: if translation and position defined at the same time

        """
        self._rigid_body_path = rigid_body_path
        self._rigid_body_prim = SingleRigidPrim(prim_path=self._rigid_body_path)
        sensor_prim_path = rigid_body_path + "/" + self._name
        self._DVL = BaseSensor(prim_path=sensor_prim_path,
                               position=position,
                               translation=translation,
                               orientation=orientation)
        
        elevation = self._elevation
        rotation = self._rotation
        orients_euler = np.array([[elevation, 0.0, rotation], 
                                  [0.0, elevation, rotation], 
                                  [-elevation, 0.0, rotation], 
                                  [0.0, -elevation, rotation]])
        orients_quat = []
        for i in range(orients_euler.shape[0]):
            orients_quat.append(euler_angles_to_quat(orients_euler[i,:], degrees=True))
            self._beam_paths.append(sensor_prim_path + f"/beam_{i}")

            result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateLightBeamSensor",
                path=self._beam_paths[i],
                min_range=self._min_range,
                max_range=self._max_range,
                forward_axis=Gf.Vec3d(0, 0, -1),
                num_rays=1,
                )
            SingleXFormPrim(prim_path=self._beam_paths[i]).set_local_pose(orientation=orients_quat[i])
        if result:
            self._DVL_interface = _range_sensor.acquire_lightbeam_sensor_interface()
        else:
            carb.log_error(f"[{self._name}] Beam Sensor fails to be loaded")

    def add_single_beam(self):
        self._single_beam_path = self._rigid_body_path + "/" + self._name +  "/SingleBeam"
        result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateLightBeamSensor",
                path=self._single_beam_path,
                min_range=self._min_range,
                max_range=self._max_range,
                forward_axis=Gf.Vec3d(0, 0, -1),
                num_rays=1,
                )
        """Add a single vertical beam to the DVL for simplified depth measurements.
    
        Creates an additional beam sensor oriented straight downward (along -Z axis).
        The beam is created at: <rigid_body_path>/<DVL_name>/SingleBeam
        
        Note:
            Primarily used for debugging or when single-beam depth measurement is sufficient.
            Uses the same min/max range settings as the main DVL beams.
        """

    def get_single_beam_range(self):
        """Get depth measurement from the vertical single beam. Only call this function after you added a singlebeam.
        
        Returns:
            float: Depth measurement in meters along the central beam.
                Returns 0 if no valid return (unlike main beams which return NaN).
                
        Note:
            This is a simpler alternative to get_depth() when only vertical range is needed.

        """
        return self._DVL_interface.get_linear_depth_data(self._single_beam_path)[0]
    
    def get_DVL_interface(self):
        """Get direct access to the underlying DVL sensor interface.
        
        Returns:
            _range_sensor.LightBeamSensorInterface: The raw physics sensor interface.
            
        Note:
            Advanced use only - provides low-level access to beam physics data.
        """
        return self._DVL_interface
    
    def get_baseSensor(self):
        """Get the core BaseSensor instance of the DVL.
        
        Returns:
            BaseSensor: The fundamental sensor prim wrapper.
            
        Note:
            Useful for modifying transform or visibility properties.
        """
        return self._DVL
    
    def get_beam_paths(self):
        """Get USD paths to all four DVL beam sensors.
        
        Returns:
            list[str]: List of four prim paths in the order:
                    [beam_0, beam_1, beam_2, beam_3]
                    
        Note:
            Paths follow pattern: <rigid_body_path>/<DVL_name>/beam_<index>
        """
        return self._beam_paths
    
    def get_depth(self):
        """Get depth measurements from all four beams.
    
        Returns:
            list[float]: Four depth measurements in meters. Returns NaN for beams with no return.
            
        Note:
            - Applies Gaussian noise if depth_cov > 0
            - Logs warning if >= num_beams_out_range_threshold beams are lost
        """
        depth = []
        if_hit = []
        for beam_path in self._beam_paths:
            depth.append(self._DVL_interface.get_linear_depth_data(beam_path)[0])
            if_hit.append(self._DVL_interface.get_beam_hit_data(beam_path)[0])
        if (self._mvn_dep.is_uncertain()):
            for i in range(4):
                sample = self._mvn_dep.sample_array()
                depth[i] += sample[i]
        # check if the sensor is in dropout state
        if if_hit.count(False) >= self._num_beams_out_range_threshold:
            carb.log_warn(f'[{self._name}] Measurement is dropped out')

        # set the no hit depth to nan
        depth = [value if hit else float('nan') for value, hit in zip(depth, if_hit)]
        return depth
    
    def get_dt(self):
        """Get current sensor update period based on operating mode.
    
        Returns:
            float: Update period in seconds.
            
        Note:
            For adaptive frequency mode, calculates period based on:
            - Fixed maximum frequency at close range
            - Sound-speed limited frequency at long range
            - Linear transition between bounds
        """
        if self._user_static_freq_flag:
            return self._dt
        else:
            min_range = min(self.get_depth())
            if min_range <= self._freq_dependent_range_bound[0]:
                self._dt = 1 / self._freq_bound[1]
            elif self._freq_dependent_range_bound[0] < min_range < self._freq_dependent_range_bound[1]:
                # To avoid abrupt jumps at h_min and h_max, smooth the transitions with linear ramp
                freq = self._freq_bound[1] - (self._freq_bound[1] - self._sound_speed/(2 * min_range))/(self._freq_dependent_range_bound[1] - self._freq_dependent_range_bound[0]) * (min_range - self._freq_dependent_range_bound[0])
                self._dt = 1 / freq
            else:
                self._dt = 1 / self._freq_bound[0]
            return self._dt
        
    def get_beam_hit(self):
        """Get hit detection status for all four DVL beams.
    
        Returns:
            list[bool]: Boolean hit status for each beam in order [beam_0, beam_1, beam_2, beam_3]
                        True indicates beam has valid return, False indicates no return detected.
    
        Note:
            - Useful for monitoring individual beam performance
            - Mirrors the hit detection used internally in get_depth() and get_linear_vel()
            - Return order matches get_beam_paths() indices
        """
        beam_hit = []
        for beam_path in self._beam_paths:
            beam_hit.append(self._DVL_interface.get_beam_hit_data(beam_path)[0].astype(bool))
        return beam_hit
    
    def get_linear_vel(self):
        """Get 3D velocity vector in body frame.
    
        Returns:
            np.ndarray: [vx, vy, vz] velocity in m/s. Returns zeros during dropout.
            
        Note:
            - Applies Gaussian noise if vel_cov > 0
        """
        if_hit = []
        for beam_path in self._beam_paths:
            if_hit.append(self._DVL_interface.get_beam_hit_data(beam_path)[0])
        if if_hit.count(False) >= self._num_beams_out_range_threshold:
            carb.log_warn(f'[{self._name}] Measurement is dropped out')
            return np.zeros(3)

        world_vel = self._rigid_body_prim.get_linear_velocity()
        _, world_orient = self._rigid_body_prim.get_world_pose()
        rot_m = quat_to_rot_matrix(world_orient)
        vel = rot_m.T @ world_vel
        if (self._mvn_vel.is_uncertain()):
            sample = self._mvn_vel.sample_array()
            for i in range(4):
                for j in range(3):
                    vel[j] += self._transform[j][i] * sample[i] 
        
        return vel
    

    def get_linear_vel_fd(self, physics_dt: float):
        """Frequency-dependent version of get_linear_vel() that respects sensor update rate.
    
        Args:
            physics_dt (float): Current physics timestep duration.
    
        Returns:
            Union[np.ndarray, float]: Velocity vector if update is due, otherwise NaN.
        """
        if self.get_dt() < physics_dt:
            carb.log_warn(f'[{self._name}] Simulation physics_dt is larger than sensor_dt. Reduced to get_linear_vel().')
        self._elapsed_time_vel += physics_dt
        if self._elapsed_time_vel >= self.get_dt():
            self._elapsed_time_vel = 0.0
            return self.get_linear_vel()
        else:
            return float('nan')

    def get_depth_fd(self, physics_dt: float):
        """Frequency-dependent version of get_depth() that respects sensor update rate.
    
        Args:
            physics_dt (float): Current physics timestep duration.
    
        Returns:
            Union[list[float], float]: Depth measurements if update is due, otherwise NaN.
        """
        if self.get_dt() < physics_dt:
            carb.log_warn(f'[{self._name}] Simulation physics_dt is larger than sensor_dt. Reduced to get_depth().')
        self._elapsed_time_depth += physics_dt
        if self._elapsed_time_depth >= self.get_dt():
            self._elapsed_time_depth = 0.0
            return self.get_depth()
        else:
            return float('nan')
        
    def set_freq(self, freq: float):
        """Set a fixed operating frequency for the DVL sensor.
    
        Args:
            freq (float): Desired operating frequency in Hz (must be > 0)
    
        Note:
            - Overrides any adaptive frequency behavior
            - Automatically calculates the corresponding period (dt = 1/freq)
            - Sets internal flag to maintain fixed frequency mode
            - To revert to adaptive frequency, create a new DVL instance
    
        Example:
            >>> dvl.set_freq(10)  # Sets DVL to update at 10Hz
        """        
        self._user_static_freq_flag = True
        self._dt = 1 / freq

    def add_debug_lines(self):
        """Visualize DVL beams in the viewport using debug drawing.
        
        Creates an action graph that continuously draws the beam paths.
        """

        (action_graph, new_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/debugLines", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacReadLightBeam0", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                    ("IsaacReadLightBeam1", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                    ("IsaacReadLightBeam2", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                    ("IsaacReadLightBeam3", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                    ("DebugDrawRayCast0", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                    ("DebugDrawRayCast1", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                    ("DebugDrawRayCast2", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                    ("DebugDrawRayCast3", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("IsaacReadLightBeam0.inputs:lightbeamPrim", self._beam_paths[0]),
                    ("IsaacReadLightBeam1.inputs:lightbeamPrim", self._beam_paths[1]),
                    ("IsaacReadLightBeam2.inputs:lightbeamPrim", self._beam_paths[2]),
                    ("IsaacReadLightBeam3.inputs:lightbeamPrim", self._beam_paths[3]),

                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam0.inputs:execIn"),
                    ("IsaacReadLightBeam0.outputs:execOut", "DebugDrawRayCast0.inputs:exec"),
                    ("IsaacReadLightBeam0.outputs:beamOrigins", "DebugDrawRayCast0.inputs:beamOrigins"),
                    ("IsaacReadLightBeam0.outputs:beamEndPoints", "DebugDrawRayCast0.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam0.outputs:numRays", "DebugDrawRayCast0.inputs:numRays"),

                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam1.inputs:execIn"),
                    ("IsaacReadLightBeam1.outputs:execOut", "DebugDrawRayCast1.inputs:exec"),
                    ("IsaacReadLightBeam1.outputs:beamOrigins", "DebugDrawRayCast1.inputs:beamOrigins"),
                    ("IsaacReadLightBeam1.outputs:beamEndPoints", "DebugDrawRayCast1.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam1.outputs:numRays", "DebugDrawRayCast1.inputs:numRays"),

                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam2.inputs:execIn"),
                    ("IsaacReadLightBeam2.outputs:execOut", "DebugDrawRayCast2.inputs:exec"),
                    ("IsaacReadLightBeam2.outputs:beamOrigins", "DebugDrawRayCast2.inputs:beamOrigins"),
                    ("IsaacReadLightBeam2.outputs:beamEndPoints", "DebugDrawRayCast2.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam2.outputs:numRays", "DebugDrawRayCast2.inputs:numRays"),

                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam3.inputs:execIn"),
                    ("IsaacReadLightBeam3.outputs:execOut", "DebugDrawRayCast3.inputs:exec"),
                    ("IsaacReadLightBeam3.outputs:beamOrigins", "DebugDrawRayCast3.inputs:beamOrigins"),
                    ("IsaacReadLightBeam3.outputs:beamEndPoints", "DebugDrawRayCast3.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam3.outputs:numRays", "DebugDrawRayCast3.inputs:numRays"),
                ],
            },
        )
