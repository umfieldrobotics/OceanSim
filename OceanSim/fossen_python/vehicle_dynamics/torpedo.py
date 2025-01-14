import numpy as np
from dynamics_controls.control import integralSMC
from dynamics_controls.gnc import crossFlowDrag, forceLiftDrag, Hmtrx, m2c, gvect, ssa

"""
torpedo.py:  

   Class for the torpedo-shaped autonomous underwater vehicle (AUV), 
   which is controlled using fins at the back and a propeller. The 
   default parameters match the REMUS 100 vehicle.               

References: 
    
    B. Allen, W. S. Vorus and T. Prestero, "Propulsion system performance 
         enhancements on REMUS AUVs," OCEANS 2000 MTS/IEEE Conference and 
         Exhibition. Conference Proceedings, 2000, pp. 1869-1873 vol.3, 
         doi: 10.1109/OCEANS.2000.882209.    
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
         Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
Modified:   Braden Meyers
"""


class TAUV:
    """
    Parent class of the torpedo-shaped vehicle. General parameters and 
    calculations for all torpedo vehicles. Actuator parameters and calculations
    are implemented in subclasses.

    :param dict scenario: scenario dictionary for holoocean 
    :param str vehicle_name: name of vehicle to initialize that matches agent in scenario dictionary
    :param str controlSystem: autopilot method for controlling the actuators
    :param float r_z: desired depth (m), positive downwards
    :param float r_psi: desired yaw angle (deg)
    :param float r_rpm: desired propeller revolution (rpm)
    :param float r_rpm: desired surge speed (m/s)
    :param float V_current: current speed (m/s)
    :param float beta_current: current direction (deg)
    """
    def __init__(
        self,
        scenario=None,
        vehicle_name=None,
        controlSystem="stepInput",
        r_z=0,
        r_psi=0,
        r_rpm=0,
        r_surge=0,
        V_current=0,
        beta_current=0,
    ):

        self.D2R = np.pi / 180
        self.g = 9.81                        # acceleration of gravity (m/s^2)
        self.dimU = len(self.controls)

        self.configure_from_scenario(scenario, vehicle_name)
        # Control parameters
        self.set_control_mode(controlSystem)
        self.set_goal(r_z, r_psi,r_rpm,r_surge)
        self.V_c = V_current
        self.beta_c = beta_current * self.D2R

        # Initialize the AUV model
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)  # velocity vector

        # Call configure_from_json if scenario is provided
            
        #Use input parameters to calculate other vehicle paramaters
        self.calculate_additional_parameters()

        self.init_depth = False  #Set the LP filter inital state to current depth when false

    def configure_from_scenario(self, scenario, vehicle_name):
        """
        Dynamics Parameters:

        :param float mass: Mass of vehicle in kilograms
        :param float length: length of vehicle in meters
        :param float rho: density of water in kg/m^3
        :param float diam: diameter of vehicle in m
        :param float r_bg: Center of gravity of the vehicle (x, y, z) in body frame x forward, y right, z down
        :param float r_bb: Center of boyancy of the vehicle (x, y, z) in body frame x forward, y right, z down
        :param float r44: 
        :param float Cd: Coefficient of drag
        :param float T_surge: 
        :param float T_sway: 
        :param float T_yaw: 
        :param float zeta_roll: 
        :param float zeta_ptich: 
        :param float K_nomoto: 

        Autopilot Paramters:
            
        - Depth

        :param float wn_d_z: damped natural frequency for low pass filter for depth commands
        :param float Kp_z: Portional gain for depth controller
        :param float T_z:
        :param float Kp_theta: Porportional gain for pitch angle for depth controller
        :param float Ki_theta: Integral gain for pitch angle for depth controller
        :param float Kd_theta: Derivative gain for pitch angle for depth controller
        :param float K_w: 
        :param float theta_max_deg: Max output of pitch controller inner loop

        
        - Heading

        :param float wn_d: Damped natural frequency of input commands for low pass filter
        :param zeta_d: Damping coefficient 
        :param r_max: 
        :param lam:
        :param phi_b:
        :param K_d:
        :param K_sigma:

        - Surge

        :param kp_surge: Porportional gain for surge
        :param ki_surge: Integral gain for surge
        :param kd_surge: Derivative gain for surge
    
        Acutator Parameters:

        :param fin_area: Surface area of one side of a fin 
        :param fin_center: Positive Z distance from center of mass to center of pressure
        :param deltaMax_fin_deg: Max deflection of the fin (degrees)
        :param nMax: Max rpm of the thruster
        :param T_delta: Time constant for fin actuation. (s)
        :param T_n: Time constant for thruster actuation. (s)
        :param CL_delta_r: Coefficient of lift for rudder 
        :param CL_delta_r: Coefficient of lift for stern 

        """

        self._scenario = scenario
        self.agent_name = vehicle_name
        
        self.dynamic_parameters ={
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
        }

        self.autopilot_parameters = {
            'depth': {
                'wn_d_z': 0.2,
                'Kp_z': 0.1,
                'T_z': 100,
                'Kp_theta': 5.0,
                'Kd_theta': 2.0,
                'Ki_theta': 0.3,
                'K_w':  5.0,
                'theta_max_deg': 30,
            },
            'heading': {
                'wn_d': 1.2,
                'zeta_d': 0.8,
                'r_max': 0.9,
                'lam': 0.1,
                'phi_b': 0.1,
                'K_d': 0.5,
                'K_sigma': 0.05,
            },
            'surge':{
                'kp_surge': 400.0,
                'ki_surge': 50.0,
                'kd_surge': 30.0,
            }
        }

        self.actuator_parameters = {
            "fin_area": 0.00697,
            "fin_center": 0.07,
            "deltaMax_fin_deg": 20,
            "nMax": 2000,
            "T_delta": 0.1,
            "T_n": 0.1,
            "CL_delta_r": 0.5,
            "CL_delta_s": 0.7
        }   

        if scenario is not None:
            if vehicle_name is None:
                raise ValueError("Vehicle name must be provided if a scenario is specified.")

            # Find the correct agent dictionary by agent_name
            agent_dict = None
            for agent in scenario.get('agents', []):
                if agent.get('agent_name') == vehicle_name:
                    agent_dict = agent
                    break

            if agent_dict is None:
                raise ValueError(f"No agent with name {vehicle_name} found in the scenario.")

            # Set vehicle parameters from the agent's 'dynamics' if it exists
            dynamics = agent_dict.get('dynamics')
            if dynamics is not None:
                self.set_vehicle_parameters(dynamics)
            else:
                self.set_vehicle_parameters(self.dynamic_parameters)
            
            # Set autopilot parameters from the agent's 'autopilot' if it exists
            autopilot_parameters = agent_dict.get('autopilot')
            if autopilot_parameters is not None:
                self.set_autopilot_parameters(autopilot_parameters)
            else:
                self.set_autopilot_parameters(self.autopilot_parameters)

            # Set actuator parameters from the agent's 'actuator' if it exists
            actuator = agent_dict.get('actuator')
            if actuator is not None:
                self.set_actuator_parameters(actuator)
            else:
                self.set_actuator_parameters(self.actuator_parameters)

    def set_vehicle_parameters(self, dynamics):
        """
        Set vehicle dynamics parameters. If not provided, will default to previous value
        """
        self.dynamic_parameters.update(dynamics)

        self.m = self.dynamic_parameters.get('mass')
        self.L = self.dynamic_parameters.get('length')
        self.rho = self.dynamic_parameters.get('rho')
        self.diam = self.dynamic_parameters.get('diam')
        self.r_bg = np.array(self.dynamic_parameters.get('r_bg'))
        self.r_bb = np.array(self.dynamic_parameters.get('r_bb'))
        self.r44 = self.dynamic_parameters.get('r44')
        self.Cd = self.dynamic_parameters.get('Cd')
        self.T_surge = self.dynamic_parameters.get('T_surge')
        self.T_sway = self.dynamic_parameters.get('T_sway')
        self.zeta_roll = self.dynamic_parameters.get('zeta_roll')
        self.zeta_pitch = self.dynamic_parameters.get('zeta_pitch')
        
        self.T_yaw = self.dynamic_parameters.get('T_yaw')
        self.K_nomoto = self.dynamic_parameters.get('K_nomoto')

        #Use input parameters to calculate other vehicle paramaters
        self.calculate_additional_parameters()
      
    def calculate_additional_parameters(self):
        """
        After updating the vehicle parameters calculations will be run 
        to update other variables related to these parameters
        """
        # Hydrodynamics (Fossen 2021, Section 8.4.2)    
        self.S = 0.7 * self.L * self.diam    # S = 70% of rectangle L * diam
        self.a = self.L/2                         # semi-axes
        b = self.diam/2                  

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)   
        # F_drag = 0.5 * rho * CD_0 * S
        self.CD_0 = self.Cd * np.pi * b**2 / self.S
        
        # Rigid-body mass matrix expressed in CO
        m = self.m                  # mass of spheriod - 4/3 * np.pi * self.rho * self.a * b**2
        Ix = (2/5) * m * b**2                       # moment of inertia
        Iy = (1/5) * m * (self.a**2 + b**2)
        Iz = Iy
        MRB_CG = np.diag([ m, m, m, Ix, Iy, Iz ])   # MRB expressed in the CG     
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg           # MRB expressed in the CO

        # Weight and buoyancy
        self.W = m * self.g
        self.B = self.W
        
        # Added moment of inertia in roll: A44 = r44 * Ix           
        MA_44 = self.r44 * Ix
        
        # Lamb's k-factors
        e = np.sqrt( 1-(b/self.a)**2 )
        alpha_0 = ( 2 * (1-e**2)/pow(e,3) ) * ( 0.5 * np.log( (1+e)/(1-e) ) - e )  
        beta_0  = 1/(e**2) - (1-e**2) / (2*pow(e,3)) * np.log( (1+e)/(1-e) )

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0  / (2 - beta_0)
        k_prime = pow(e,4) * (beta_0-alpha_0) / ( 
            (2-e**2) * ( 2*e**2 - (2-e**2) * (beta_0-alpha_0) ) )   

        # Added mass system matrix expressed in the CO
        self.MA = np.diag([ m*k1, m*k2, m*k2, MA_44, k_prime*Iy, k_prime*Iy ])
          
        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Natural frequencies in roll and pitch
        self.w_roll = np.sqrt( self.W * ( self.r_bg[2]-self.r_bb[2] ) / 
            self.M[3][3] )
        self.w_pitch = np.sqrt( self.W * ( self.r_bg[2]-self.r_bb[2] ) / 
            self.M[4][4] )
            
        # Low-speed linear damping matrix parameters
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        # Feed forward gains (Nomoto gain parameters)
        self.T_nomoto = self.T_yaw  # Time constant in yaw

    def set_autopilot_parameters(self, autopilot):
        """
        Set depth and heading parameters from a configuration dictionary.

        :param cfg: Dictionary containing 'depth' and 'heading' sections with their respective parameters.
        """
        # Update depth parameters
        if 'depth' in autopilot:
            self.autopilot_parameters['depth'].update(autopilot['depth'])
        
        # Update heading parameters
        if 'heading' in autopilot:
            self.autopilot_parameters['heading'].update(autopilot['heading'])

        # Update heading parameters
        if 'surge' in autopilot:
            self.autopilot_parameters['surge'].update(autopilot['surge'])
        
        depth_cfg = self.autopilot_parameters.get('depth', {})
        heading_cfg = self.autopilot_parameters.get('heading', {})
        surge_cfg = self.autopilot_parameters.get('surge', {})

        #### Surge Parameters
        self.surge_control = False
        self.kp_surge = surge_cfg.get('kp_surge')
        self.ki_surge = surge_cfg.get('ki_surge')
        self.kd_surge = surge_cfg.get('kd_surge')
        self.u_int = 0   # surge error integral state
        self._last_error = 0

        #### Set depth parameters
        self.wn_d_z = depth_cfg.get('wn_d_z')   # desired natural frequency, reference mode
        self.Kp_z = depth_cfg.get('Kp_z')         # heave proportional gain, outer loop
        self.T_z = depth_cfg.get('T_z')            # heave integral gain, outer loop
        self.Kp_theta = depth_cfg.get('Kp_theta')    # pitch PID controller 
        self.Kd_theta = depth_cfg.get('Kd_theta')
        self.Ki_theta = depth_cfg.get('Ki_theta')
        self.K_w = depth_cfg.get('K_w')               # optional heave velocity feedback gain
        self.theta_max = np.deg2rad(depth_cfg.get('theta_max_deg'))               # optional heave velocity feedback gain
        

        # Heading autopilot (Equation 16.479 in Fossen 2021)
        # sigma = r-r_d + 2*lambda*ssa(psi-psi_d) + lambda^2 * integral(ssa(psi-psi_d))
        # delta = (T_nomoto * r_r_dot + r_r - K_d * sigma 
        #       - K_sigma * (sigma/phi_b)) / K_nomoto
        ##### heading parameters
        self.wn_d = heading_cfg.get('wn_d')      # desired natural frequency
        self.zeta_d = heading_cfg.get('zeta_d')    # desired realtive damping ratio
        self.r_max = heading_cfg.get('r_max')   # maximum yaw rate
        self.lam = heading_cfg.get('lam')
        self.phi_b = heading_cfg.get('phi_b')   # boundary layer thickness
        self.K_d = heading_cfg.get('K_d')         # PID gain
        self.K_sigma = heading_cfg.get('K_sigma') # SMC switching gain

        self.z_int = 0         # heave position integral state
        self.z_d = 0           # desired position, LP filter initial state
        self.theta_int = 0     # pitch angle integral state
        self.psi_d = 0   # desired heading from control loop
        self.r_d = 0     # desired yaw rate from control loop
        self.a_d = 0
        self.e_psi_int = 0   # yaw angle error integral state
        self.prev_pitch = 0
        self.prev_yaw = 0

    def set_control_mode(self, controlSystem, init_depth=False):
        """
        Sets the control mode for the vehicle.

        :param str controlSystem: The control system to use. Possible values are:
        
        - ``"depthHeadingAutopilot"``: Depth and heading autopilots.
        - ``"manualControl"``: Manual input control with set_u_control().
        - ``"stepInput"``: Step inputs for stern planes, rudder, and propeller
        - Any other value: controlSystem is set to "stepInput".

        :param bool init_depth: Whether to initialize depth (default is False).

        :returns: None
        """
        if controlSystem == "depthHeadingAutopilot":
            self.controlDescription = "Depth and heading autopilots"
            self.init_depth = init_depth
            self.z_int = 0
            self.e_psi_int = 0 
            self.u_int = 0
            print("Warning: Setting control mode resets controller so be careful to set control mode only when necessary")
        elif controlSystem == 'manualControl':
            self.controlDescription = 'Manual input control with set_u_control()'
        else:
            self.controlDescription = "Step inputs for stern planes, rudder and propeller"
            controlSystem = "stepInput"
        self.controlMode = controlSystem
        print(self.controlDescription)

    def set_goal(self, depth=None, heading=None, rpm=None, surge=None):
        """
        Set the goals for the autopilot.

        :param float depth: Desired depth (m), positive downwards.
        :param float heading: Desired yaw angle (deg). (-180 to 180)
        :param float rpm: Desired propeller revolution (rpm).
        :param float surge: Desired body frame x velocity (m/s).

        :returns: None
        """
        if rpm is not None:
            self.ref_n = rpm
            self.surge_control = False
            if rpm < 0.0 or rpm > self.nMax:
                raise ValueError(f"The RPM value should be in the interval 0-{self.nMax}")
        if heading is not None:
            self.ref_psi = heading
            if abs(heading) > 180.0:
                raise ValueError(f"The heading command value should be on the interval -180 to 180")
        if depth is not None:
            self.ref_z = depth
            if depth > 100.0 or depth < 0.0:
                raise ValueError(f"The depth command value should be in the interval 0-100(m)")
        if surge is not None:
            self.ref_u = surge
            self.surge_control = True

    def set_surge_goal(self, surge):
        """
        Set the surge goals for the autopilot.

        :param float depth: Desired surge (m/s), positive forward in body frame.

        :returns: None
        """
        #TODO add caps? negative values? and max surge?
        self.ref_u = surge
        self.surge_control = True

    def set_heading_goal(self, heading):
        """
        Set the heading goals for the autopilot.

        :param float depth: Desired heading (deg), -180 to 180 in NED frame

        :returns: None
        """
        

        self.ref_psi = heading
        if abs(heading) > 180.0:
            raise ValueError(f"The heading command value should be on the interval -180 to 180")

    def set_depth_goal(self, depth):
        """
        Set the depth goals for the autopilot.

        :param float depth: Desired depth (m), positive downward in world frame.

        :returns: None
        """

        self.ref_z = depth

        if depth > 100.0 or depth < 0.0:
            raise ValueError(f"The depth command value should be in the interval 0-100(m)")

    def set_rpm_goal(self, rpm):
        """
        Set the rpm goals for the autopilot.

        :param float depth: Desired rpm for thruster

        :returns: None
        """

        self.ref_n = rpm
        self.surge_control = False

        if rpm < 0.0 or rpm > self.nMax:
            raise ValueError(f"The RPM value should be in the interval 0-{self.nMax}")

    def surgeAutopilot(self, nu, sampleTime):
        #TODO: Check that this is working and grabbing the right variables

        u = nu[0]                   # surge velocity
        # TODO: get nu_dot from linear acceleration IMU - gravity
        # udot = nu_dot[0]            # surge acceleration

        setpoint = self.ref_u
        error = setpoint - u
        derivative = (error - self._last_error) / sampleTime

        n = self.kp_surge * error + self.ki_surge * self.u_int + self.kd_surge * derivative

        self.u_int += sampleTime * (error)

        if n > self.nMax:
            n = self.nMax       #Max out surge controller to the propeller command

        return n

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        Integrates the AUV equations of motion using Euler's method.

        :param array-like eta: State/pose of the vehicle in the world frame.
        :param array-like nu: Velocity of the vehicle in the body frame.
        :param array-like nu_dot: Acceleration of the vehicle in the body frame.
        :param array-like u_actual: Current control surface position.
        :param array-like u_control: Commanded control surface position.
        :param float sampleTime: Time since the last step.

        :returns: Three arrays: nu, u_actual, and nu_dot.
        """

        # Current velocities
        u_c = self.V_c * np.cos(self.beta_c - eta[5])  # current surge velocity
        v_c = self.V_c * np.sin(self.beta_c - eta[5])  # current sway velocity

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float) # current velocity 
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = nu - nu_c                               # relative velocity        
        alpha = np.arctan2( nu_r[2], nu_r[0] )         # angle of attack 
        U = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)  # vehicle speed
        U_r = np.sqrt(nu_r[0]**2 + nu_r[1]**2 + nu_r[2]**2)  # relative speed

        #Forces and moments from actuators
        forces, moments = self.actuator_dynamics(u_actual, nu_r) #TODO fix this actuator dynamics

        n = u_actual[-1] #Make sure the propeller command is always the last item in control list
        ################# Propeller Calulations ################

        # Propeller coeffs. KT and KQ are computed as a function of advance no.
        # Ja = Va/(n*D_prop) where Va = (1-w)*U = 0.944 * U; Allen et al. (2000)
        D_prop = 0.14   # propeller diameter corresponding to 5.5 inches
        t_prop = 0.1    # thrust deduction number
        n_rps = n / 60  # propeller revolution (rps) 
        Va = 0.944 * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        Ja_max = 0.6632
        
        # Single-screw propeller with 3 blades and blade-area ratio = 0.718.
        # Coffes. are computed using the Matlab MSS toolbox:     
        # >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
        KT_0 = 0.4566
        KQ_0 = 0.0700
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3) 
        KT_max = 0.1798
        KQ_max = 0.0312
        
        # Propeller thrust and propeller-induced roll moment
        # Linear approximations for positive Ja values
        # KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
        # KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja  
      
        if n_rps > 0:   # forward thrust

            X_prop = self.rho * pow(D_prop,4) * ( 
                KT_0 * abs(n_rps) * n_rps + (KT_max-KT_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )        
            K_prop = self.rho * pow(D_prop,5) * (
                KQ_0 * abs(n_rps) * n_rps + (KQ_max-KQ_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )           
            
        else:    # reverse thrust (braking)
        
            X_prop = self.rho * pow(D_prop,4) * KT_0 * abs(n_rps) * n_rps 
            K_prop = self.rho * pow(D_prop,5) * KQ_0 * abs(n_rps) * n_rps 
        
        ###################### F = MA ##################

        # Rigi-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = m2c(self.MRB, nu_r)
        CA  = m2c(self.MA, nu_r)
               
        # CA-terms in roll, pitch and yaw can destabilize the model if quadratic
        # rotational damping is missing. These terms are assumed to be zero
        CA[4][0] = 0     # Quadratic velocity terms due to pitching
        CA[0][4] = 0  
        CA[2][4] = 0
        CA[5][0] = 0    #Munk moment in yaw
        CA[5][1] = 0
        CA[1][5] = 0
        
        C = CRB + CA

        # Dissipative forces and moments
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll  * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw
            ])
        
        # Linear surge and sway damping
        D[0][0] = D[0][0] * np.exp(-3*U_r) # vanish at high speed where quadratic
        D[1][1] = D[1][1] * np.exp(-3*U_r) # drag and lift forces dominates

        tau_liftdrag = forceLiftDrag(self.diam,self.S,self.CD_0,alpha,U_r)
        tau_crossflow = crossFlowDrag(self.L,self.diam,self.diam,nu_r)

        # Restoring forces and moments
        g = gvect(self.W,self.B,eta[4],eta[3],self.r_bg,self.r_bb)
        
        # Generalized force vector
        tau = np.array([
            (1-t_prop) * X_prop + forces[0], 
            forces[1], 
            forces[2],
            K_prop / 10 + moments[0],   # scaled down by a factor of 10 to match exp. results
            moments[1],
            moments[2]
            ], float)
    
        # AUV dynamics
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C+D,nu_r)  - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum) #Acceleration from forces plus ocean current acceleration

        #Move the actuators towards commanded value 
        u_actual = self.move_actuator(sampleTime, u_control, u_actual)
        #Amplitutde saturation of control surfaces
        u_actual = self.saturate_actuator(u_actual) 

        return nu_dot, u_actual 
    
    def move_actuator(self, sampleTime, u_control, u_actual):
        
        u_actual_dot = []

        #Fin Speed
        for i in range(self.dimU-1):
            u_actual_dot.append((u_control[i] - u_actual[i]) / self.T_delta) 

        #Thruster acceleration
        u_actual_dot.append((u_control[-1] - u_actual[-1]) / self.T_n) 

        #Control surface integration
        for i in range(self.dimU):
            u_actual[i] += sampleTime * u_actual_dot[i]

        return u_actual

    def saturate_actuator(self,u_actual):

        #Saturate fins
        for i in range(self.dimU-1):
            # Amplitude saturation of the control signals
            if abs(u_actual[i]) >= self.deltaMax:
                u_actual[i] = np.sign(u_actual[i]) * self.deltaMax

        # Saturate thruster value  
        if abs(u_actual[-1]) >= self.nMax:
            u_actual[-1] = np.sign(u_actual[-1]) * self.nMax 

        return u_actual
    
    ################ Functions implmented in subclasses below: #################

    def set_actuator_parameters(self, actuator_parameters):
        """
        Set fin area limits, time constants, and lift coefficients for control surfaces
        """
        

        self.S_fin = self.actuator_parameters.get('fin_area')
        self.T_delta = self.actuator_parameters.get('T_delta')
        self.T_n = self.actuator_parameters.get('T_n')
        self.nMax = self.actuator_parameters.get('nMax')
        
        #Max fin angles
        self.deltaMax = np.radians(self.actuator_parameters.get('deltaMax_fin_deg'))

        #Z Distance from center of mass to center of pressure on fin
        self.z_r = self.actuator_parameters.get('fin_center')

        #Lift Coefficients
        self.CL_delta_r = self.actuator_parameters.get('CL_delta_r')
        self.CL_delta_s = self.actuator_parameters.get('CL_delta_s')

        #TODO: ADD THE x position parameter for the fins

    def stepInput(self, t):
        """
        Generates step inputs.

        :param float t: Time parameter.

        :returns: The control input u_control.
        """
        pass #placeholder for stepInput function
    
    def depthHeadingAutopilot(self, eta, nu, sampleTime, imu=True):
        """
        Simultaneously control the heading and depth of the AUV using control laws of PID type.
        Propeller rpm is given as a step command.

        :param array-like eta: State/pose of the vehicle in the world frame. (RPY - Euler angle order zyx in radians)
        :param array-like nu: Velocity of the vehicle in the body frame.
        :param float sampleTime: Time since the last step.

        :returns: The control input u_control.
        """
        z = eta[2]                  # heave position (depth)
        theta = eta[4]              # pitch angle (Radians)
        psi = eta[5]                # yaw angle   (Radians)
        w = nu[2]                   # heave velocity

        if imu:
            q = nu[4]                   # pitch rate
            r = nu[5]                   # yaw rate
        else:
            q = (psi - self.prev_pitch) / sampleTime
            r = (theta - self.prev_yaw) / sampleTime
            self.prev_pitch = theta
            self.prev_yaw = psi

        e_psi = psi - self.psi_d    # yaw angle tracking error
        e_r   = r - self.r_d        # yaw rate tracking error
        z_ref = self.ref_z          # heave position (depth) setpoint
        psi_ref = self.ref_psi * self.D2R   # yaw angle setpoint
        
        #If surge command is 0 then control loop should not run 
        if self.ref_n > 0 or self.ref_u > 0:
            #######################################################################
            # Propeller command
            #######################################################################
            if self.surge_control:
                #TODO: Super of self?
                n = self.surgeAutopilot(nu,sampleTime)
            else:
                n = self.ref_n 
            
            #######################################################################            
            # Depth autopilot (succesive loop closure)
            #######################################################################
            # LP filtered desired depth command 
            if not self.init_depth:
                self.z_d = z    #On initialization of the autopilot the commanded depth is set to the current depth
                self.init_depth = True
            self.z_d  = np.exp( -sampleTime * self.wn_d_z ) * self.z_d \
                + ( 1 - np.exp( -sampleTime * self.wn_d_z) ) * z_ref  
                
            # PI controller    
            theta_d = self.Kp_z * ( (z - self.z_d) + (1/self.T_z) * self.z_int )

            if abs(theta_d) > self.theta_max:
                theta_d = np.sign(theta_d) * self.theta_max

            delta_s = -self.Kp_theta * ssa( theta - theta_d ) - self.Kd_theta * q \
                - self.Ki_theta * self.theta_int - self.K_w * w

            # Euler's integration method (k+1)
            self.z_int     += sampleTime * ( z - self.z_d )
            self.theta_int += sampleTime * ssa( theta - theta_d )

            #######################################################################
            # Heading autopilot (SMC controller)
            #######################################################################
            
            wn_d = self.wn_d            # reference model natural frequency
            zeta_d = self.zeta_d        # reference model relative damping factor


            # Integral SMC with 3rd-order reference model
            [delta_r, self.e_psi_int, self.psi_d, self.r_d, self.a_d] = \
                integralSMC( 
                    self.e_psi_int, 
                    e_psi, e_r, 
                    self.psi_d, 
                    self.r_d, 
                    self.a_d, 
                    self.T_nomoto, 
                    self.K_nomoto, 
                    wn_d, 
                    zeta_d, 
                    self.K_d, 
                    self.K_sigma, 
                    self.lam,
                    self.phi_b,
                    psi_ref, 
                    self.r_max, 
                    sampleTime 
                    )
                    
            # Euler's integration method (k+1)
            self.e_psi_int += sampleTime * ssa( psi - self.psi_d )
            
            
            u_control = np.array([ delta_r, delta_s, n], float)

        else:
            u_control = np.array([ 0.0, 0.0, 0.0], float)

        return u_control

    def actuator_dynamics(self, u_actual, nu_r):
        """
        Vehicle-specific calculations for dynamics of the actuators (fins and thruster).

        Note: For Torpedo Vehicles, positive fin deflection will pitch the vehicle up and yaw to the starboard side.

        :param array-like u_actual: Current control surface position.
        :param array-like nu_r: Reference velocity of the vehicle in the body frame.

        :returns: two arrays: forces and moments
        """
        pass



class fourFinDep(TAUV):
    """
    Torpedo Vehicle with four fins where two fins move together on same plane (Rudder, Stern)
    """
    def __init__(self, scenario=None, vehicle_name=None, controlSystem="stepInput", r_z=0, r_psi=0, r_rpm=0, V_current=0, beta_current=0):
        
        self.u_actual = np.array([0, 0, 0], float)  # control input vector

        self.controls = [
            "Tail rudder (deg)",
            "Stern plane (deg)",
            "Propeller revolution (rpm)"
            ]

        super().__init__(scenario, vehicle_name, controlSystem, r_z, r_psi, r_rpm, V_current, beta_current)
        
        # Tail rudder parameters (single)
        self.A_r = 2 * self.S_fin        # rudder area (m2)
        self.x_r = -self.a               # rudder x-position (m)

        # Stern-plane paramaters (double)
        self.A_s = 2 * self.S_fin        # stern-plane area (m2)
        self.x_s = -self.a               # stern-plane x-position (m)
    
    def actuator_dynamics(self, u_actual, nu_r):
        
        delta_r = u_actual[0]       # actual tail rudder (rad)
        delta_s = u_actual[1]       # actual stern plane (rad)

        # Horizontal- and vertical-plane relative speed
        U_rh = np.sqrt( nu_r[0]**2 + nu_r[1]**2 )
        U_rv = np.sqrt( nu_r[0]**2 + nu_r[2]**2 ) 

        # Rudder and stern-plane drag
        X_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r**2
        X_s = -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_s**2

        # Rudder sway force (Positive deflection yaws vehicle to starboard)
        Y_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r

        # Stern-plane heave force (Postive deflection pitches vehicle up)
        Z_s = 0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_s

        forces = [X_r + X_s, Y_r, Z_s]
        moments = [0, -1 * self.x_s * Z_s, self.x_r * Y_r]  #No total roll moment induced by fins moving together on both sides 

        return forces, moments

    def stepInput(self, t):
        """
        Returns:
            list:
                The control input u_control as a list: [delta_r, delta_s, n], where:
                
                - delta_r (float): Rudder angle (rad)
                - delta_s (float): Stern plane angle (rad)
                - n (float): Propeller revolution (rpm)
        """
        delta_r =  15 * self.D2R      # rudder angle (rad)
        delta_s =  0 * self.D2R      # stern angle (rad)
        n = 1525                     # propeller revolution (rpm)
        
        if t > 100:
            delta_r = 0
            
        if t > 50:
            delta_s = 0     

        u_control = np.array([ delta_r, delta_s, n], float)

        return u_control
    
    #TODO: Seperate control loop into 3 different functions
    def depthHeadingAutopilot(self, eta, nu, sampleTime,imu=True):
        """
        Returns:
            list:
                The control input u_control as a list: [delta_r, delta_s, n], where:
                
                - delta_r (float): Rudder angle (rad)
                - delta_s (float): Stern plane angle (rad)
                - n (float): Propeller revolution (rpm)
        """
        control = super().depthHeadingAutopilot(eta,nu,sampleTime,imu)
        u_control = control

        return u_control
    
class fourFinInd(TAUV):
    """
    Torpedo vehicle with four independetly controlled fins (Rudder Top, Rudder Bottom, Stern left, Stern Right)
    """

    def __init__(self, scenario=None, vehicle_name=None, controlSystem="stepInput", r_z=0, r_psi=0, r_rpm=0, V_current=0, beta_current=0):

        self.u_actual = np.array([0, 0, 0, 0, 0], float)  # control input vector

        self.controls = [
            "Top Tail rudder (deg)",
            "Bottom Tail rudder (deg)",
            "Left Stern (deg)",
            "Right Stern  (deg)",
            "Propeller revolution (rpm)"
            ]
        
        super().__init__(scenario, vehicle_name, controlSystem, r_z, r_psi, r_rpm, V_current, beta_current)
        
        # Tail rudder parameters (single)
        self.A_r = self.S_fin           # rudder area (m2)
        self.x_r = -self.a               # rudder x-position (m)

        # Stern-plane paramaters (double)
        self.A_s =  self.S_fin        # stern-plane area (m2)
        self.x_s = -self.a             # Stern fin x position

    def actuator_dynamics(self, u_actual, nu_r):
        
        #Amplitude saturation of fins and propeller control
        delta_rt = u_actual[0]       # actual tail rudder top (rad)
        delta_rb = u_actual[1]       # actual tail rudder bottom (rad)
        delta_sl = u_actual[2]       # actual stern plane left (rad)
        delta_sr = u_actual[3]       # actual stern plane right (rad)

        # Horizontal- and vertical-plane relative speed
        U_rh = np.sqrt( nu_r[0]**2 + nu_r[1]**2 )
        U_rv = np.sqrt( nu_r[0]**2 + nu_r[2]**2 )  

        # Rudder and stern-plane drag (Always in negative direction regardless of fin deflection sign)
        X_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_rt**2
        X_r += -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_rb**2
        X_s = -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_sl**2
        X_s += -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_sr**2

        # Rudder sway force (Positive deflection causes negative force in Y body frame NED, yaw to starboard)
        Y_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_rt
        Y_r += -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_rb

        # Stern-plane heave force (Postive deflection causes positve force in Z body frame NED, pitches up)
        Z_s = 0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_sl
        Z_s += 0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_sr

        #Roll induced by offset fins not calculated negligable for current simulation
        
        forces = [X_r + X_s, Y_r, Z_s]
        moments = [0, -self.x_s * Z_s, self.x_r * Y_r] 

        return forces, moments

    def stepInput(self, t):
        """
        Returns:
            list:
                The control input u_control as a list: [delta_rt, delta_rb, delta_sl, delta_sr, n], where:
                
                - delta_rt: Rudder top angle (rad).
                - delta_rb: Rudder bottom angle (rad).
                - delta_sl: Stern left angle (rad).
                - delta_sr: Stern right angle (rad).
                - n: Propeller revolution (rpm).
        """
        delta_rt =  5 * self.D2R      # rudder angle (rad)
        delta_rb =  5 * self.D2R      # rudder angle (rad)
        delta_sl = -5 * self.D2R      # stern angle (rad)
        delta_sr = -5 * self.D2R      # stern angle (rad)
        n = 1525                     # propeller revolution (rpm)
        
        if t > 100:
            delta_rt = 0
            delta_rb = 0
           
        if t > 50:
            delta_sl = 0     
            delta_sr = 0     
        u_control = np.array([delta_rt, delta_rb, delta_sl, delta_sr, n], float)

        return u_control
    
    def depthHeadingAutopilot(self, eta, nu, sampleTime,imu=True):
        """
        Returns:
            list:
                The control input u_control as a list: [delta_rt, delta_rb, delta_sl, delta_sr, n], where:

                - delta_rt: Rudder top angle (rad).
                - delta_rb: Rudder bottom angle (rad).
                - delta_sl: Stern left angle (rad).
                - delta_sr: Stern right angle (rad).
                - n: Propeller revolution (rpm).
        """
        control = super().depthHeadingAutopilot(eta,nu,sampleTime,imu)
        delta_rt = delta_rb = control[0]
        delta_sl = delta_sr = control[1]
        n = control[2]

        u_control = np.array([ delta_rt, delta_rb,delta_sl,delta_sr, n], float)


        return u_control


class threeFinInd(TAUV):
    """
    Torpedo vehicle with four independetly controlled fins (Rudder Top, Rudder Bottom, Stern left, Stern Right)
    """

    def __init__(self, scenario=None, vehicle_name=None, controlSystem="stepInput", r_z=0, r_psi=0, r_rpm=0, V_current=0, beta_current=0):

        self.u_actual = np.array([0, 0, 0, 0], float)  # control input vector

        self.controls = [
            "Tail rudder (rad)",
            "Left Elevator (rad)",
            "Right Elevator (rad)",
            "Propeller revolution (rpm)"
            ]
        
        super().__init__(scenario, vehicle_name, controlSystem, r_z, r_psi, r_rpm, V_current, beta_current)
        
        # Fin parameters (single)
        self.A_fin = self.S_fin           # rudder area (m2)
        self.x_fin = -self.a               # rudder x-position (m)

    def actuator_dynamics(self, u_actual, nu_r):
        
        delta_r = u_actual[0]       # actual tail rudder (rad)
        delta_re = u_actual[1]       # actual right elevator (rad)
        delta_le = u_actual[2]       # actual left elevator (rad)

        # Horizontal- and vertical-plane relative speed   
        U_rh = np.sqrt( nu_r[0]**2 + nu_r[1]**2 )
        U_re = np.sqrt(nu_r[0]**2 + (nu_r[1] * np.sin(self.D2R * 30))**2 + (nu_r[2] * np.sin(self.D2R * 60))**2)

        #Positive rudder deflection turn the vehicle right, postive elevator deflection pitches vehicle up

        #lift forces on the elevator fins on right and left both positve set direction below
        fl_re = 0.5 * self.rho * U_re**2 * self.A_fin * self.CL_delta_s * delta_re
        fl_le = 0.5 * self.rho * U_re**2 * self.A_fin * self.CL_delta_s * delta_le

        # Rudder and elevator drag 
        X_r = -0.5 * self.rho * U_rh**2 * self.A_fin * self.CL_delta_r * delta_r**2   
        X_re = -0.5 * self.rho * U_re**2 * self.A_fin * self.CL_delta_s * delta_re**2    
        X_le = -0.5 * self.rho * U_re**2 * self.A_fin * self.CL_delta_s * delta_le**2  
        fx = X_r + X_re + X_le

        # Rudder and elevator sway force (Positive deflection -> negative Y force -> postive Z moment (yaw right))
        Y_r = -0.5 * self.rho * U_rh**2 * self.A_fin * self.CL_delta_r * delta_r
        Y_re = -fl_re * np.sin(30 * self.D2R)
        Y_le = fl_le * np.sin(30 * self.D2R)  
        fy = Y_r + Y_re + Y_le        

        # elevator heave force  (positve z force )
        Z_re = fl_re * np.sin(60 * self.D2R)     
        Z_le = fl_le * np.sin(60 * self.D2R)
        fz = Z_le + Z_re

        Mx = 0  #Rolling moment from the fins has a negligable moment arm 
        My = self.x_fin * fz * -1 # -1 comes from the cross product of x with z                           
        Mz =  (self.x_fin * Y_r) + (self.x_fin * Y_re) + (self.x_fin * Y_le)

        forces = [fx, fy, fz]
        moments = [Mx, My, Mz]

        return forces, moments

    def stepInput(self, t):
        """
        Returns:
            list: [delta_r, delta_rb, delta_sl, delta_sr, n], where:

            - delta_r: Rudder top angle (rad).
            - delta_re: right elevator angle (rad).
            - delta_le: left elevator angle (rad).
            - n: Propeller revolution (rpm).
        """
        delta_r =  5 * self.D2R      # rudder angle (rad)
        delta_re = 0 * self.D2R      # right elevator angle (rad)
        delta_le = 0 * self.D2R      # left elevator angle (rad)  
        n = 1000 #1525                    # propeller revolution (rpm)
        
        if t > 50:
            delta_r = 0
            
        if t > 25:
            delta_re = 0     
            delta_le = 0     

        u_control = np.array([ delta_r, delta_re, delta_le, n], float)

        return u_control
    
    def depthHeadingAutopilot(self, eta, nu, sampleTime,imu=True):
        """
        Returns:
            list: [delta_r, delta_rb, delta_sl, delta_sr, n], where:
            
            - delta_r: Rudder top angle (rad).
            - delta_re: right elevator angle (rad).
            - delta_le: left elevator angle (rad).
            - n: Propeller revolution (rpm).
        """
        control = super().depthHeadingAutopilot(eta,nu,sampleTime,imu)
        delta_r = control[0]
        delta_re = delta_le = control[1]
        n = control[2]

        u_control = np.array([ delta_r, delta_re, delta_le, n], float)

        return u_control