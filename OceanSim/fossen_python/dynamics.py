"""
Used for attaching Fossen Dynamics to Holocean 
Conversion between the two coordinate systems
"""

import numpy as np
# from holoocean.dynamics_controls import *
from scipy.spatial.transform import Rotation

class FossenDynamics:
    """
    Class for handling the connection between a holocean agent and
    the fossen dynamic model to control the motion of the vehicle
    """
    
    def __init__(self,vehicle,sample_period,initial_u_actual=None):
        
        self.sampleTime = sample_period

        self.vehicle = vehicle

        #Scenario Check (Dynamics Sensor, Control Scheme)
        if self.vehicle._scenario is not None:
            check_scenario_configuration(self.vehicle._scenario, self.vehicle.agent_name)

        if initial_u_actual is not None:
            self.u_actual = initial_u_actual
            self.vehicle.u_actual = initial_u_actual
        else:
            self.u_actual = self.vehicle.u_actual

        self.u_control = self.vehicle.u_actual

    def update(self, state):
        '''
        Update state of the vehicle to return the acceleartion of the vehicle in holoocean world frame
        '''
        x = state["DynamicsSensor"]

        #From world frame state change to body frame in correct coordinate sys
        eta, nu = self.convert_NWU_to_NED(x) 

        t = state["t"]     

        # Vehicle specific control systems
        if (self.vehicle.controlMode == 'depthAutopilot'):
            self.u_control = self.vehicle.depthAutopilot(eta,nu,self.sampleTime)
        elif (self.vehicle.controlMode == 'headingAutopilot'):
            self.u_control = self.vehicle.headingAutopilot(eta,nu,self.sampleTime)   
        elif (self.vehicle.controlMode == 'depthHeadingAutopilot'):
            self.u_control = self.vehicle.depthHeadingAutopilot(eta,nu,self.sampleTime)             
        elif (self.vehicle.controlMode == 'DPcontrol'):
            self.u_control = self.vehicle.DPcontrol(eta,nu,self.sampleTime)                   
        elif (self.vehicle.controlMode == 'stepInput'):
            self.u_control = self.vehicle.stepInput(t)
        elif(self.vehicle.controlMode != 'manualControl'):
            raise ValueError(f"Unknown control mode: {self.vehicle.controlMode}")

        #Compute dynamics in body frame, NED coordinate system
        [nu_dot, self.u_actual] = self.vehicle.dynamics(eta, nu, self.u_actual, self.u_control, self.sampleTime)
        
        #return the acceleration in world frame for holoocean
        return self.convert_NED_to_NWU(x, nu_dot)

    def set_u_control_rad(self, u_control):
        '''Set the control surface commands for the vehicle control surfaces with angles in radians'''
        if len(u_control) != self.vehicle.dimU: 
            raise ValueError(f"Expected length of control commands for this vehicle is {self.vehicle.dimU}, but got {len(u_control)}.")
        
        self.u_control=u_control

    #This Matrix transforms coordinate frames between NED and NWU
    T_coord_sys = np.array([
        [ 1,  0,  0 ],
        [ 0,  -1,  0],
        [ 0,  0,     -1] ])

    def R2euler(self, R):
        """
        Computes the Euler angles from the rotation matrix R.

        Args:
            R (numpy.ndarray): 3x3 rotation matrix

        Returns:
            phi, theta, psi (float): Euler angles (in radians)  
        """
        phi = np.arctan2(R[2, 1], R[2, 2])
        theta = -np.arcsin(R[2, 0])
        psi = np.arctan2(R[1, 0], R[0, 0])
        
        return phi, theta, psi

    def convert_NWU_to_NED(self,x):
        '''
        Converts position and velocity from NWU to NED and world frame velocities to body frame
        
        Args:
            x: Holoocean Dynamic Sensor return of position (with quaternion), velocity, and acceleration (len = 19)
        
        Returns:
            eta, nu (np.ndarray - float): [x, y, z, roll, pitch, yaw], [xdot, ydot, zdot, p, q, r]
        '''
        #world frame values for vehicle
        pos = x[6:9]
        quat = x[15:19]
        vel = x[3:6]
        omega = x[12:15]

        R_body_to_world = Rotation.from_quat(quat).as_matrix()
        R_world_to_body = np.matrix.transpose(R_body_to_world)

        #Transform Position to NED coordinate system
        eta = np.matmul(self.T_coord_sys, pos)

        #calculate RPY in the right Euler order (zyx) for Fossen Dynamics
        R_Fos = np.matmul(np.matmul(self.T_coord_sys, R_body_to_world), self.T_coord_sys) 
        RPY = np.array(self.R2euler(R_Fos))

        #World frame velocities changed from world frame to body frame and then to NED coord system
        vel_body_NED = np.matmul(self.T_coord_sys, np.matmul(R_world_to_body, vel))
        omega_body_NED = np.matmul(self.T_coord_sys, np.matmul(R_world_to_body, omega))

        #Returns the state in the format [eta,nu], frame(body frame), and coordinate system (NED) for fossen dynamic simulations
        return np.append(eta, RPY), np.append(vel_body_NED, omega_body_NED)

    def convert_NED_to_NWU(self, x, accel):
        '''
        Calculates rotation to world frame and changes the coordinate system from NED to NWU    
        Rotates linear and angular accelaration from body to world      

        Args:
            x: Holoocean Dynamic Sensor return of position (with quaternion), velocity, and acceleration (len = 19)
            accel (np.ndarray) - [accelx, accely, accelz, pdot, qdot, rdot] - Body frame in NED
        
        Returns:
            accel (np.ndarray - float): [accelx, accely, accelz, pdot, qdot, rdot] - World frame in NWU
        '''

        quat = x[15:19]
        R_body_to_world = Rotation.from_quat(quat).as_matrix()

        lin_accel = accel[:3]
        ang_accel = accel[3:6]

        #Calculate acceleration from body frame (NED) to world frame (NWU) 
        lin_accel = np.matmul(R_body_to_world, np.matmul(self.T_coord_sys, lin_accel))  
        ang_accel = np.matmul(R_body_to_world, np.matmul(self.T_coord_sys, ang_accel))  

        #Return acceleration commands in NWU world frame
        return np.append(lin_accel, ang_accel) 



def check_scenario_configuration(scenario, vehicle_name):
    if "agents" not in scenario:
        raise ValueError("The scenario must contain an 'agents' key.")
    
    agent = next((agent for agent in scenario["agents"] if agent["agent_name"] == vehicle_name), None)
    if agent is None:
        raise ValueError(f"No agent with the name '{vehicle_name}' found in the scenario.")
    
    if agent.get("control_scheme") != 1:
        raise ValueError(f"Agent {agent['agent_name']} should have a control scheme of 1.")
    
    if agent.get("agent_type") != "TorpedoAUV":
        raise ValueError(f"Agent {agent['agent_name']} should be of type 'TorpedoAUV'.")
    
    sensors = agent.get("sensors", [])
    dynamics_sensor_found = False
    
    for sensor in sensors:
        if sensor.get("sensor_type") == "DynamicsSensor":
            if not ("sensor_name" in sensor) or sensor.get("sensor_name") == "DynamicsSensor":
                config = sensor.get("configuration", {})
                
                if not config.get("UseCOM"):
                    raise ValueError(f"Agent {agent['agent_name']} DynamicsSensor must have 'UseCOM' set to True.")
                
                if config.get("UseRPY"):
                    raise ValueError(f"Agent {agent['agent_name']} DynamicsSensor must have 'UseRPY' set to False.")
                dynamics_sensor_found = True
            
    
    if not dynamics_sensor_found:
        raise ValueError(f"Agent {agent['agent_name']} must have a 'DynamicsSensor' in its sensors that is named DynamicsSensor")
