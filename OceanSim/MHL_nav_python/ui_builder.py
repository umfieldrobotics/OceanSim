# Omniverse import
import numpy as np
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.gui.components import CollapsableFrame, Frame, StateButton, get_style
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid


# Custom import
from ..sensors.DVLsensor import DVLsensor
from .scenario import MHL_straighline_navigation_Scenario

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



    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters

        self._rob_mass = 10 # kg
        # DVl parameters
        self._DVL = None
        self._DVL_elevation = 30.0 # deg
        self._DVL_min_range = 0.01  # m
        self._DVL_max_range = 20.0 # m
        self._DVL_location = np.array([0.0,0.0,-0.2])
        self._DVL_vel_cov = 0
        self._DVL_depth_cov = 0


        # Camera parameters
        self._cam = None
        self._cam_res = (1920, 1080)
        self._cam_pose = ([0.5,0.0,0.0], [0.5,0.5,-0.5,-0.5])
        self._cam_focal = 6
        self._scenario = MHL_straighline_navigation_Scenario()


    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        # Open MHL scene
        # open_stage("/home/haoyu/isaacsim_assets/USD/mhl_scaled/MHL_Water.usd")
        open_stage('/home/haoyu/isaacsim_assets/USD/DVl_test.usd')
        # Load the robot
        robot_prim_path = "/World/rob"
        # robot_usd_path = '/home/haoyu-ma/projects/OceanSim_utils/assets/usd/BlueRov/BROV2-HEAVY_0.5down.usd'
        # add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        DynamicCuboid(prim_path=robot_prim_path, size=0.25, color=np.array([0.5,0.5,1]))
        # Load the rock
        # rock_prim_path = '/MHL/rock'
        # rock_usd_path = '/home/haoyu-ma/projects/OceanSim_utils/assets/usd/3d_model/rock/rock.usd'
        # add_reference_to_stage(usd_path=rock_usd_path, prim_path=rock_prim_path)
        
        # Toggle rigid body and collider preset for rob and rock
        self._rob = get_prim_at_path(robot_prim_path)
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(self._rob)
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        rob_rigidBody_API.GetLinearDampingAttr().Set(0.0)
        rob_rigidBody_API.GetAngularDampingAttr().Set(0.0)
        rob_rigid_prim = SingleRigidPrim(prim_path=robot_prim_path,
                                         mass=self._rob_mass)
        
        # rock_collider_prim = SingleGeometryPrim(prim_path=rock_prim_path,
        #                    translation=np.array([0.0, 2, -2]),
        #                    orientation=euler_angles_to_quat(np.array([0.0,0.0,125]), degrees=True),
        #                    collision=True,
        #                    )
        # rock_collider_prim.set_collision_approximation('convexDecomposition')
        # rock_rigid_prim = SingleRigidPrim(prim_path=rock_prim_path)
        
        
        
        # Initialize the DVL and attach it to the rob already being the rigid body
        self._DVL = DVLsensor()
        self._DVL.attachDVL(rigid_body_path=robot_prim_path, location=self._DVL_location)
        self._DVL.add_debug_lines()
        
        # Attach the front camera
        cam_prim_path = robot_prim_path + '/Camera'
        self._cam = Camera(
            prim_path=cam_prim_path,
            resolution=self._cam_res,
            )
        self._cam.set_focal_length(0.1 * self._cam_focal)
        SingleXFormPrim(cam_prim_path).set_local_pose(translation=self._cam_pose[0],orientation=self._cam_pose[1])


        # MHLMesh_prim_path = '/World/mhl_scaled/Mesh'
        # SingleGeometryPrim(prim_path=MHLMesh_prim_path, collision=True)


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
        self._scenario.setup_scenario(self._rob, self._DVL, self._cam)

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
        # self._scenario.save()

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
