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
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.dynamic_control import _dynamic_control
from .vehicle_dynamics import *
from .dynamics import *
from omni.isaac.sensor import _sensor
import carb
import omni.isaac.core.utils.rotations as rotations_utils

"""
This scenario takes in a robot Articulation and makes it move through its joint DOFs.
Additionally, it adds a cuboid prim to the stage that moves in a circle around the robot.

The particular framework under which this scenario operates should not be taken as a direct
recomendation to the user about how to structure their code.  In the simple example put together
in this template, this particular structure served to improve code readability and separate
the logic that runs the example from the UI design.
"""


class FossenScenario(ScenarioTemplate):
    def __init__(self):
        self._rob_prim_path = None
        self._articulation = None

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._IMU = _sensor.acquire_imu_sensor_interface()
        self._running_scenario = False
        self._time = 0.0

        self.pos_buffer = [[0,0,-10]]
        self.time_buffer = [0]
        self.linear_accel_buffer = [np.array(np.zeros(3), float)]
        self.angular_accel_buffer = [np.array(np.zeros(3), float)]


        
        

    def setup_scenario(self, articulation, rob_prim_path):
        self._articulation = articulation
        self._rob_prim_path = rob_prim_path
                
        self._rob_body = self._dc.get_rigid_body(self._rob_prim_path) 

        # Fossen scenario setup
        self._ticks_per_sec = 50

        initial_location = [0,0,-10] #Translation in NWU coordinate system
        initial_rotation = [0,0,0] #Roll, pitch, Yaw in Euler angle order ZYX and in degrees NWU coordinate system

        self._scenario_config = {
        "name": "torpedo_dynamics",
        "package_name": "Ocean",
        "world": "OpenWater",
        "main_agent": "auv0",
        "ticks_per_sec": self._ticks_per_sec,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "TorpedoAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": False  # Use quaternion for dynamics
                        }
                    },
                ],
                "control_scheme": 1,  # Control scheme 1 is how custom dynamics are applied to TAUV
                "location": initial_location,
                "rotation": initial_rotation,
                "dynamics": 
                    {
                        "mass": 16,
                        "length": 1.6,
                        "rho": 1026,
                        "diam": 0.19,
                        "r_bg": [0, 0, 0.02],
                        "r_bb": [0, 0, 0],
                        "r44": 0.3,
                        "Cd": 0.42,
                        "T_surge": 20,
                        "T_sway": 20,
                        "zeta_roll": 0.3,
                        "zeta_pitch": 0.8,
                        "T_yaw": 1,
                        "K_nomoto": 5.0 / 20.0
                    },
                "actuator": 
                    {
                        "fin_area": 0.00665,
                        "deltaMax_fin_deg": 15,
                        "nMax": 1525,
                        "T_delta": 0.1,
                        "T_n": 0.1,
                        "CL_delta_r": 0.5,
                        "CL_delta_s": 0.7
                    },
                "autopilot": 
                    {
                        'depth': {
                            'wn_d_z': 0.2,
                            'Kp_z': 0.08,
                            'T_z': 100,
                            'Kp_theta': 4.0,
                            'Kd_theta': 2.3,
                            'Ki_theta': 0.3,
                            'K_w':  5.0,
                        },
                        'heading': {
                            'wn_d': 1.2,
                            'zeta_d': 0.8,
                            'r_max': 0.9,
                            'lam': 0.1,
                            'phi_b': 0.1,
                            'K_d': 0.5,
                            'K_sigma': 0.05,
                        }
                    },
                
                }
            ]
        }
        
        vehicle = fourFinDep(scenario=self._scenario_config,
                             vehicle_name='auv0',
                             controlSystem='manualControl')
        period = 1.0/self._ticks_per_sec
        self._torpedo_dynamics = FossenDynamics(vehicle=vehicle, 
                                          sample_period=period)

        
        
        ############## MANUAL CONTROL EXAMPLE: ###########
        #Set control surfaces command
        fins_degrees = np.array([5 , 5]) #Rudder and Stern Fin Deflection (degrees)
        fin_radians = np.radians(fins_degrees)
        thruster_rpm = 800
        self.u_control = np.append(fin_radians,thruster_rpm)  #[RudderAngle, SternAngle,Thruster]
        self.accel = np.array(np.zeros(6),float)


        self._running_scenario = True


    def teardown_scenario(self):
        self._time = 0.0


    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step


        self.action(self.accel)
        self._torpedo_dynamics.set_u_control_rad(self.u_control)
        state = self.read_dynamics_info()
        self.accel = self._torpedo_dynamics.update(state)

        x = state["DynamicsSensor"]

        self.pos_buffer.append(x[6:9])
        self.time_buffer.append(self._time)
        self.linear_accel_buffer.append(self.accel[:3])
        self.angular_accel_buffer.append(self.accel[3:])

    
    
    
    
    
    
    def read_dynamics_info(self):
        # Use the COM of the entire rob for reading and manipulating dynamics

        # Use dynamics control plugin for reading position, orientation, linear\angular vel
        # Needs to convert them from carb array to numpy
        
        rob_body_transform = self._dc.get_rigid_body_pose(self._rob_body)
        position = np.array([*(rob_body_transform.p)])
        orient_quad = np.array([*(rob_body_transform.r)])
        orient_rpy = rotations_utils.quat_to_euler_angles(orient_quad)
        angular_vel = self._dc.get_rigid_body_angular_velocity(self._rob_body)
        angular_vel = np.array([*angular_vel])
        linear_vel = self._dc.get_rigid_body_linear_velocity(self._rob_body)
        linear_vel = np.array([*linear_vel])
        
        # Get acceleration and orientation reading from IMU sensor
        IMU_reading = self._IMU.get_sensor_reading(self._rob_prim_path + "/IMU")
        linear_acc = np.array([float(IMU_reading.lin_acc_x), 
                               float(IMU_reading.lin_acc_y), 
                               float(IMU_reading.lin_acc_z)])
        # This is a naming error in their IMU sensor "ang_vel" actually means "ang_acc"
        # In Isaac Lab, they use velocity numerical differentiation to compute acceleration.
        angular_acc = np.array([float(IMU_reading.ang_vel_x), 
                                float(IMU_reading.ang_vel_y), 
                                float(IMU_reading.ang_vel_z)])
        return {
            'DynamicsSensor' : np.concatenate([
                linear_acc,
                linear_vel,
                position,
                angular_acc,
                angular_vel,
                orient_quad,
            ]),
            't' : self._time

        }

         

    def action(self, accel):
        # dynamic control API uses cm/s as unit
        accel = 0.01 * accel
        state = self.read_dynamics_info()
        new_linear_vel = state["DynamicsSensor"][3:6] + accel[:3]
        new_angular_vel = state["DynamicsSensor"][12:15] + accel[3:]
        self._dc.set_rigid_body_linear_velocity(self._rob_body,
                                                carb.Float3(*new_linear_vel))
        self._dc.set_rigid_body_angular_velocity(self._rob_body,
                                                 carb.Float3(*new_angular_vel))

    def plot(self):
        import matplotlib.pyplot as plt

        saved_path = '/home/haoyu-ma/Desktop'
        # Convert position list to a numpy array for easier slicing
        pos_array = np.array(self.pos_buffer)
        linear_accel_array = np.array(self.linear_accel_buffer)
        angular_accel_array = np.array(self.angular_accel_buffer)
        # Extract x, y, and z positions
        x_positions = pos_array[:, 0] #North Position
        y_positions = pos_array[:, 1]  #West Position
        east_positions = [-y for y in y_positions] #Convert from west to east
        z_positions = pos_array[:, 2]   #Depth

        # Plot x and y positions
        plt.figure()
        plt.plot( east_positions,x_positions, marker='o')
        plt.title('X and Y Positions')
        plt.xlabel('East (meters)')
        plt.ylabel('North (meters)')
        plt.grid(True)

        plt.savefig(saved_path + '/XY.png')

        # Plot z positions over time
        plt.figure()
        plt.plot(self.time_buffer, z_positions, marker='o')
        plt.title('Z Position over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Z Position')
        plt.grid(True)
        
        plt.savefig(saved_path + '/Z.png')


        plt.figure()
        plt.plot( self.time_buffer,linear_accel_array[:,0], marker='o', label='x_acc')
        plt.plot( self.time_buffer,linear_accel_array[:,1], marker='x', label='y_acc')
        plt.plot( self.time_buffer,linear_accel_array[:,2], marker='*', label='z_acc')
        plt.legend()
        plt.title('Accel')
        plt.xlabel('Accel')
        plt.ylabel('t')
        plt.grid(True)

        plt.savefig(saved_path + '/acc.png')


        plt.figure()
        plt.plot( self.time_buffer,angular_accel_array[:,0], marker='o', label='ang_x_acc')
        plt.plot( self.time_buffer,angular_accel_array[:,1], marker='x', label='ang_y_acc')
        plt.plot( self.time_buffer,angular_accel_array[:,2], marker='*', label='ang_z_acc')
        plt.legend()
        plt.title('ang_Accel')
        plt.xlabel('ang_Accel')
        plt.ylabel('t')
        plt.grid(True)

        plt.savefig(saved_path + '/ang_acc.png')

        print(f"Plot save to {saved_path}")

