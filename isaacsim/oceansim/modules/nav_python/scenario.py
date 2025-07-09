from pxr import PhysxSchema, Gf
import numpy as np
from omni.replicator.core import BackendDispatch
import omni.replicator.core.scripts.functional as F
from isaacsim.sensors.camera import Camera
from collections import defaultdict
import isaacsim.core.utils.rotations as rotations_utils
import isaacsim.core.utils.numpy as numpy_utils
import pickle
class NavScenario:
    def __init__(self):
        
        self._running_scenario = False
        self._time = 0.0

        self._rob = None
        self._cam = None
        self._force_cmd = None
        self._torque_cmd = None
        self._id = 0 
        self._output_dir = "/home/haoyu/Desktop/nav_SDG/traj_5"
        self._backend = BackendDispatch(output_dir=self._output_dir)
        self._data_logger = defaultdict(list)
        self._log_frequency = 1 / 30
        self._prev_log_times = 0
        
    def setup_scenario(self, rob, cam: Camera):
        self._rob = rob
        self._cam = cam
        self._cam.initialize()
        from ...utils.keyboard_cmd import keyboard_cmd

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
            
        self._running_scenario = True
        
    def teardown_scenario(self):
        
        if self._force_cmd:
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()
        self._data_logger.clear()

        self._running_scenario = False
        self._time = 0.0
        self._id = 0
        



    def update_scenario(self, step: float):
        if (not self._running_scenario) or (self._cam.get_rgb().size == 0):
            return
        
        force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
        torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
        self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
        self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)

        
        log_times, _ = divmod(self._time, self._log_frequency)
        if log_times > self._prev_log_times:
            self.log_data()
            self._prev_log_times = log_times


        self._time += step

    def save_log(self):
        pickle_path = self._output_dir + "/traj_data.pkl"
        print("------ Data structure -------")
        for key, value in self._data_logger.items():
            self._data_logger[key] = np.stack(value)
            print(key, " ", self._data_logger[key].shape, " ",self._data_logger[key].dtype)
        print("-----------------------------")
        with open(pickle_path, "wb") as f:
            pickle.dump(self._data_logger, f)
        T = len(self._data_logger["position"])
        
        print(f"Trajectory pickle saved to {pickle_path}")
        print(f"Images saved to {self._output_dir}. Wait for a while for images to be fully saved before reset.")
        


    def log_data(self):
        position = self._rob.GetAttribute("xformOp:translate").Get()
        position = np.array([*position])
        orient = self._rob.GetAttribute("xformOp:orient").Get()
        orient = rotations_utils.quat_to_euler_angles(rotations_utils.gf_quat_to_np_array(orient))
        print("Logged Position ", position)
        print("Logged Orient", orient)
        
        self._backend.schedule(F.write_jpeg, data = self._cam.get_rgb(), path = f"{self._id}.jpg")
        self._data_logger["position"].append(position[:2])
        self._data_logger["yaw"].append(orient[2]) 
        self._data_logger["time"].append(self._time)
        self._data_logger["id"].append(self._id)       
        
        self._id += 1   
