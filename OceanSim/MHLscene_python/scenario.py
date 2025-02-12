# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
class ScenarioTemplate:
    def __init__(self):
        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass


import numpy as np
from isaacsim.core.prims import XFormPrim
from omni.isaac.dynamic_control import _dynamic_control
from omni.replicator.core.scripts.functional import write_np, write_image
import omni.replicator.core as rep
import carb


class MHLScenario(ScenarioTemplate):
    def __init__(self):
        self._rob = None
        self._DVL = None
        self._cam = None
        self._running_scenario = False
        self._time = 0.0

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._output_dir = '/home/haoyu-ma/Desktop/MHL_replica'

        self._fourBeam_buffer = []
        self._singleBeam_buffer = []
        self._vel_buffer = []


    def setup_scenario(self, rob, DVL, cam):
        self._rob = rob
        self._DVL = DVL
        self._cam = cam


        self._running_scenario = True
        self._rob_body = self._dc.get_rigid_body("/root/rob")
        
        self._backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})
        rp = rep.create.render_product(
            camera='/root/rob/Camera',
            resolution=(1920,1080),

            )
        
        self._ldr = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self._depth = rep.AnnotatorRegistry.get_annotator('distance_to_camera')
        self._ldr.attach(rp)
        self._depth.attach(rp)


    def teardown_scenario(self):
        self._rob = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        self._time += step
        if self._ldr.get_data().size == 0:
            return
        

        # if (self._time < 1):
        #     self._dc.apply_body_force(self._rob_body, carb.Float3(0.1,0.0,0.0), carb.Float3(0,0,0), 0)
        self._dc.set_rigid_body_linear_velocity(self._rob_body, carb.Float3(5,0.0,0.0))
        rob_pos, rob_orient = XFormPrim('/root/rob').get_world_poses()
        XFormPrim('/root/skin').set_world_poses(positions=rob_pos, orientations=rob_orient)
        # print(f'Vel:{self._DVL.get_linear_vel()}')
        # print(f'depth:{self._DVL.get_depth()}')
        # print(f'single beam:{self._DVL.get_singleBeam_range()}')

        # self._backend.schedule(write_np, path=f'vel/vel_{self._id}.npy', data=self._DVL.get_linear_vel())
        # self._backend.schedule(write_np, path=f'singleBeam/singleBeam_{self._id}.npy', data=self._DVL.get_singleBeam_range())
        # self._backend.schedule(write_np, path=f'fourBeam/fourBeam_{self._id}.npy', data=self._DVL.get_depth())
        self._singleBeam_buffer.append(self._DVL.get_singleBeam_range())
        self._fourBeam_buffer.append(self._DVL.get_depth())
        self._vel_buffer.append(self._DVL.get_linear_vel())
        self._backend.schedule(write_image, path=f'cam/rgb_{self._id}.png', data=self._ldr.get_data())
        self._backend.schedule(write_np, path=f'depth/depth_{self._id}.npy', data=self._depth.get_data())
        print(f'writing [{self._id}]')
        self._id += 1


    def save(self):
        np.save(file=self._output_dir+"/vel.npy", arr=self._vel_buffer)
        np.save(file=self._output_dir+"/singleBeam.npy", arr=self._singleBeam_buffer)
        np.save(file=self._output_dir+"/fourBeam.npy", arr=self._fourBeam_buffer)   
        print(f'data written to {self._output_dir}')