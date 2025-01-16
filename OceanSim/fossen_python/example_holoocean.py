"""
This file is not being used. 
It is an example from Holoocean that shows how to do fossen implementation.
"""
import numpy as np
# import holoocean
# from holoocean.environments import *
from vehicle_dynamics import *
from dynamics import *


"""
Example of how Thor Fossen models can be used in the HoloOcean simulator to model 
the motion of the vehicle based on control surface commands. Uses the FossenDyanmics
class that is is the dynamics.py file. This vehicle has 4 fins controlled by two 
angle inputs of the stern and rudder fins. Torpedo vehicles with 3 or 4 independently 
controlled fins can also be used with the current model. 

NOTE: Vehicle parameters are currently only tuned for the REMUS100 vehicle.
Mass and other parameters set in engine are ignored with this control scheme as they are taken into 
account witht the Fossen Models. 
"""

ticks_per_sec = 50
# print("Change Additional Launch Parameters to match ticks_per_sec if running live")
# numSteps = 600
# print("Total Simulation Time:", (numSteps/ticks_per_sec))

initial_location = [0,0,-10] #Translation in NWU coordinate system
initial_rotation = [0,0,0] #Roll, pitch, Yaw in Euler angle order ZYX and in degrees NWU coordinate system

scenario = {
"name": "torpedo_dynamics",
"package_name": "Ocean",
"world": "OpenWater",
"main_agent": "auv0",
"ticks_per_sec": ticks_per_sec,
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

# env = holoocean.make(scenario_cfg=scenario)

#Create vehicle object attached to holoocean agent with dynamic parameters 
vehicle = fourFinDep(scenario, 'auv0','manualControl')    

period = 1.0/ticks_per_sec
#Create dynamics object passing in the vehicle created
torpedo_dynamics = FossenDynamics(vehicle,period)  

accel = np.array(np.zeros(6),float)  #HoloOcean parameter input 


pos_list = []
time_list = []

############## MANUAL CONTROL EXAMPLE: ###########
#Set control surfaces command
fins_degrees = np.array([5 , 5]) #Rudder and Stern Fin Deflection (degrees)
fin_radians = np.radians(fins_degrees)
thruster_rpm = 800
u_control = np.append(fin_radians,thruster_rpm)  #[RudderAngle, SternAngle,Thruster]
vehicle.set_control_mode('manualControl')



for i in range(600):
    # state = env.step(accel)
    torpedo_dynamics.set_u_control_rad(u_control) #If desired you can change control command here
    # The below line return the acceleration in the world frame
    # accel = torpedo_dynamics.update(state) #Calculate accelerations to be applied to HoloOcean agent

    #For Plotting
    # pos = state['DynamicsSensor'][6:9]  # [x, y, z]
    # pos_list.append(pos)
    # time_list.append(state['t'])


    ############ Depth Heading Control Example:  ############ 
    # env.reset()
    # numSteps = 800

    # depth = 13
    # heading = 50    
    # vehicle.set_goal(depth,heading,1525)     #Changes depth (positive depth), heading, thruster RPM goals for controller 
    # vehicle.set_control_mode('depthHeadingAutopilot') #In this mode PID controller calculates control commands (u_control)
    # for i in range(numSteps):
    #     state = env.step(accel)
    #     accel = torpedo_dynamics.update(state)


    #     # For plotting and arrows 
    #     pos = state['DynamicsSensor'][6:9]  # [x, y, z]
    #     x_end = pos[0] + 3 * np.cos(np.deg2rad(heading))
    #     y_end = pos[1] - 3 * np.sin(np.deg2rad(heading))
    #     pos_list.append(pos)
    #     time_list.append(state['t'])

    #     #change color if within 2 meters
    #     if abs(depth + pos[2]) <= 2.0:
    #         color = [0,255,0]
    #     else:
    #         color = [255,0,0]

    #     env.draw_arrow(pos.tolist(), end=[x_end, y_end, -depth], color=color, thickness=5, lifetime=0.03)
    

    # ################ Plot vehicle State ###################
    # plot = True

    # if plot:
    #     import matplotlib.pyplot as plt
    #     # Convert position list to a numpy array for easier slicing
    #     pos_array = np.array(pos_list)

    #     # Extract x, y, and z positions
    #     x_positions = pos_array[:, 0] #North Position
    #     y_positions = pos_array[:, 1]  #West Position
    #     east_positions = [-y for y in y_positions] #Convert from west to east
    #     z_positions = pos_array[:, 2]   #Depth

    #     # Plot x and y positions
    #     plt.figure()
    #     plt.plot( east_positions,x_positions, marker='o')
    #     plt.title('X and Y Positions')
    #     plt.xlabel('East (meters)')
    #     plt.ylabel('North (meters)')
    #     plt.grid(True)

    #     # Plot z positions over time
    #     plt.figure()
    #     plt.plot(time_list, z_positions, marker='o')
    #     plt.title('Z Position over Time')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Z Position')
    #     plt.grid(True)

    #     # Show the plots
    #     plt.show()