import numpy as np
from omni.replicator.core.scripts.functional import write_np
import omni.replicator.core as rep
from pxr import Gf, PhysxSchema

from isaacsim.oceansim.utils.keyboard_cmd import keyboard_cmd
from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor

class ImagingSonarScenario():
    def __init__(self):
        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0
        self._output_dir = '/home/haoyu/Desktop/viz'
        self._force_cmd = None
        self._torque_cmd = None
        
    def setup_scenario(self, rob, sonar : ImagingSonarSensor):
        self._rob = rob
        self._sonar = sonar     
        self._sonar.sonar_initialize(include_unlabelled=True)
   
        self._running_scenario = True
        self.backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})

        self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
        self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                    input_keyboard_mapping={
                                    # forward command
                                    "W": [10.0, 0.0, 0.0],
                                    # backward command
                                    "S": [-10.0, 0.0, 0.0],
                                    # leftward command
                                    "A": [0.0, 10.0, 0.0],
                                    # rightward command
                                    "D": [0.0, -10.0, 0.0],
                                        # rise command
                                    "UP": [0.0, 0.0, 10.0],
                                    # sink command
                                    "DOWN": [0.0, 0.0, -10.0],
                                    })
        self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                    input_keyboard_mapping={
                                    # yaw command (left)
                                    "J": [0.0, 0.0, 10.0],
                                    # yaw command (right)
                                    "L": [0.0, 0.0, -10.0],
                                    # pitch command (up)
                                    "I": [0.0, -10.0, 0.0],
                                    # pitch command (down)
                                    "K": [0.0, 10.0, 0.0],
                                    # row command (left)
                                    "LEFT": [-10.0, 0.0, 0.0],
                                    # row command (negative)
                                    "RIGHT": [10.0, 0.0, 0.0],
                                    })

    def teardown_scenario(self):

        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0

        if self._force_cmd is not None:
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()
        if self._sonar is not None:
            self._sonar.close()



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step


        self._sonar.make_sonar_data(gau_noise_param=0.5)
        force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
        torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
        self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
        self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)


