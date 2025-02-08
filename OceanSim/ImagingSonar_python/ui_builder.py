# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
import omni.timeline
import omni.ui as ui
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from omni.isaac.core.world import World
from omni.isaac.ui.element_wrappers import CollapsableFrame, Frame, StateButton, XYPlot
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux
from .scenario import ImagingSonarScenario
from omni.isaac.core.objects import GroundPlane
import omni.isaac.core.utils.rotations as rotations_utils
import omni.isaac.core.utils.prims as prims_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.viewports import set_camera_view
from .ImagingSonarSensor import ImagingSonarSensor

class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario", collapsed=False)

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        self._outputs_frame = CollapsableFrame("Outputs", collapsed=False)



    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):
        self._rob = None
        self._sea_floor = None
        self._sonar = None
        self._sensor_location = [0.5, 0.0, 0.0]
        self._init_rob_pos = np.array([3, -1, 2])
        self._init_rob_orien = rotations_utils.euler_angles_to_quat(np.array([0, np.deg2rad(55), 0]))


        self._scenario = ImagingSonarScenario()

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])
       

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.

        In this example, a new stage is loaded explicitly, and all assets are reloaded.
        If the user is relying on hot-reloading and does not want to reload assets every time,
        they may perform a check here to see if their desired assets are already on the stage,
        and avoid loading anything if they are.  In this case, the user would still need to add
        their assets to the World (which has low overhead).  See commented code section in this function.
        """
        # you can create a new stage. Or comment out this line to load on current stage
        create_new_stage()
        
        # Add light 
        self._add_light_to_stage()

        # Load the robot
        robot_prim_path = "/rob"

        robot_usd_path = '/home/haoyu-ma/projects/OceanSim_utils/assets/usd/BlueRov/BlueRov2_heavy.usd'
        # For now use a dynamic cube as the robot
        self._rob = add_reference_to_stage(usd_path=robot_usd_path,prim_path=robot_prim_path)
        XFormPrim(robot_prim_path).set_world_pose(position=self._init_rob_pos, orientation=self._init_rob_orien)
        # Toggle rigid body and apply zero gravity and zero damping
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(self._rob)
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        rob_rigidBody_API.GetLinearDampingAttr().Set(0.0)
        rob_rigidBody_API.GetAngularDampingAttr().Set(0.0)

        # Set the default viewport view angle
        set_camera_view(eye=[-5.0, 5.0, 5.0], target=[2.00, 2.00, 2.0], camera_prim_path="/OmniverseKit_Persp")


        self._sonar = ImagingSonarSensor(prim_path=robot_prim_path,
                                        trans=self._sensor_location)        

        #For now use the flat ground plane as the seafloor
        sea_floor_prim_path = "/GroundPlane"
        self._sea_floor = GroundPlane(prim_path=sea_floor_prim_path)
        
        #Reference the obstacle on stage
        obstacle_asset_path = ['/home/haoyu-ma/projects/OceanSim_utils/assets/usd/imagingSonar_usd/toy_biplane_idle.usdz',
                               '/home/haoyu-ma/projects/OceanSim_utils/assets/usd/imagingSonar_usd/toy_car.usdz']
        obstacle_prim_path = ['/World/Obstacle/obstacle_0', '/World/Obstalce/obstacle_1']

        add_reference_to_stage(obstacle_asset_path[0], obstacle_prim_path[0])
        XFormPrim(obstacle_prim_path[0]).set_local_scale(scale=[0.1, 0.1, 0.1])  
        XFormPrim(obstacle_prim_path[0]).set_world_pose(position=[5.0,-2.5,0.0],orientation=rotations_utils.euler_angles_to_quat([np.pi/2,0,0]))

        add_reference_to_stage(obstacle_asset_path[1], obstacle_prim_path[1])
        XFormPrim(obstacle_prim_path[1]).set_local_scale(scale=[0.1, 0.1, 0.1])  
        XFormPrim(obstacle_prim_path[1]).set_world_pose(position=[5.0,1.0,0.0],orientation=rotations_utils.euler_angles_to_quat([np.pi/2,0,0]))

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._rob, self._sonar)

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        Args:
            step (float): The dt of the current physics step
        """
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False
