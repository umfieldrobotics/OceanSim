# Python Vehicle Simulator

Holoocean uses Thor Fossens Python Vehicle simulator to more accurately simulate the dynamics
of vehicles compared to the physics on Unreal engine. For more information on this simulator
the source code can be found here along with links to his textbook and other vehicles that 
could be used in holoocean by putting them into the src/holoocean/vehicle_dynamics folder and 
modifying them to follow the torpedo.py format

The Python Vehicle Simulator supplements the Matlab MSS (Marine Systems Simulator) toolbox. It includes models for autonomous underwater vehicles (AUVs), unmanned surface vehicles (USVs), and ships. The vehicle models are based on the MSS vessel models in /MSS/VESSELS/catalog. Each vehicle is modeled as an object in Python, and the vehicle class has methods for guidance, navigation, and control. 

    Library files:        
        control.py              - feedback control systems
        gnc.py                  - generic GNC functions
        guidance.py             - guidance functions  
        models.py                          
        
For more information about the mathematical modeling of marine craft and methods for guidance, navigation, and control, please consult:

T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. 2nd. Edition, Wiley. 
https://wiley.fossen.biz