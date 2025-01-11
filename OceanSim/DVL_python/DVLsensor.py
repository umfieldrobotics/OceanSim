import numpy as np
from pxr import Gf
from omni.isaac.core.prims import BaseSensor
import omni.isaac.core.utils.rotations as rotations_utils
from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.dynamic_control import _dynamic_control
from ..utils.MultivariateNormal import MultivariateNormal
import omni.graph.core as og
import carb
class DVLsensor:
    def __init__(self,
                 elevation:float = 22.5, # deg
                 vel_cov = 0,
                 depth_cov = 0,
                 min_range: float = 0.1,
                 max_range: float = 10,
                 ):
        self._elevation = elevation
        self._min_range = min_range
        self._max_range = max_range
        self._mvn_vel = MultivariateNormal(4)
        self._mvn_vel.init_cov(vel_cov)

        self._mvn_dep = MultivariateNormal(4)
        self._mvn_dep.init_cov(depth_cov)
        
        self._rigid_body_path = None
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._beam_paths = []

        sinElev = np.sin(np.deg2rad(self._elevation))
        cosElev = np.cos(np.deg2rad(self._elevation))
        self._transform = np.array([[1/(2*sinElev), 0, -1/(2*sinElev), 0],
                                    [0, 1/(2*sinElev), 0, -1/(2*sinElev)],
                                    [1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev)]
                                    ])
        

    def attachDVL(self, rigid_body_path:str, location:np.ndarray = np.array([0.0, 0.0, 0.0])):
        self._rigid_body_path = rigid_body_path
        sensor_prim_path = rigid_body_path + "/DVL"
        self._DVL = BaseSensor(prim_path=sensor_prim_path,translation=location)
        
        elevation = np.deg2rad(self._elevation)
        orients_euler = np.array([[elevation, 0.0, 0.0], [0.0, elevation, 0.0], [-elevation, 0.0, 0.0], [0.0, -elevation, 0.0]])

        orients_quat = []
        for i in range(orients_euler.shape[0]):
            orients_quat.append(rotations_utils.euler_angles_to_quat(orients_euler[i,:]))
            self._beam_paths.append(sensor_prim_path + "/beam" + str(i))

            result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateLightBeamSensor",
                path=self._beam_paths[i],
                parent=None,
                min_range=self._min_range,
                max_range=self._max_range,
                translation=Gf.Vec3d(0, 0, 0),
                orientation=Gf.Quatd(*orients_quat[i]),
                forward_axis=Gf.Vec3d(0, 0, -1),
                num_rays=1,
                ) 
        if result:
            self._DVL_interface = _sensor.acquire_lightbeam_sensor_interface()
        else:
            carb.log_error("Beam Sensor fails to be loaded")
    

    def get_DVL_interface(self):
        return self._DVL_interface
    
    def get_baseSensor(self):
        return self._DVL
    
    def get_beam_paths(self):
        return self._beam_paths
    
    def get_depth(self):
        depth = []
        for beam_path in self._beam_paths:
            depth.append(self._DVL_interface.get_linear_depth_data(beam_path).squeeze())
        
        if (self._mvn_dep.is_uncertain()):
            for i in range(4):
                sample = self._mvn_dep.sample_array()
                depth[i] += sample[i]
        
        return depth
    
    
    def get_beam_hit(self):
        beam_hit = []
        for beam_path in self._beam_paths:
            beam_hit.append(self._DVL_interface.get_beam_hit_data(beam_path).astype(bool).squeeze())
        return beam_hit
    
    def get_linear_vel(self):
        rob_body = self._dc.get_rigid_body(self._rigid_body_path)
        
        vel = self._dc.get_rigid_body_linear_velocity(rob_body)

        if (self._mvn_vel.is_uncertain()):
            sample = self._mvn_vel.sample_array()
            for i in range(4):
                for j in range(3):
                    vel[j] += self._transform[j][i] * sample[i] 
        
        return vel

    def add_debug_lines(self):

        (action_graph, new_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/debugLines", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacReadLightBeam0", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("IsaacReadLightBeam1", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("IsaacReadLightBeam2", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("IsaacReadLightBeam3", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("DebugDrawRayCast0", "omni.isaac.debug_draw.DebugDrawRayCast"),
                    ("DebugDrawRayCast1", "omni.isaac.debug_draw.DebugDrawRayCast"),
                    ("DebugDrawRayCast2", "omni.isaac.debug_draw.DebugDrawRayCast"),
                    ("DebugDrawRayCast3", "omni.isaac.debug_draw.DebugDrawRayCast"),
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