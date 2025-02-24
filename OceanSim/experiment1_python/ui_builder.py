# Omniverse import
import numpy as np
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# Isaac sim import
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, create_new_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.gui.components import CollapsableFrame, Frame, StateButton, get_style
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton


# Custom import
from ..sensors.ImagingSonarSensor_warp import ImagingSonarSensor
from ..sensors.UW_Camera import UW_Camera
from .scenario import pier_Scenario

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

        self._rob_mass = 5 # kg
        # Sonar parameters
        self._sonar = None

        # Camera parameters
        self._cam = None
        self._cam_res = (1920, 1080)

        self._cam_focal = 8.0
        # Scenario
        self._scenario = pier_Scenario()

    def _add_domelight(self):
        domelight = UsdLux.DomeLight.Define(get_current_stage(), Sdf.Path('/World/Domelight'))
        domelight.CreateIntensityAttr(1000)
        SingleXFormPrim(str(domelight.GetPath())).set_world_pose(position=np.array([0.0, 0.0, 100]))
       
    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._add_domelight()


        scene_prim_path = '/World/Pier'
        # scene_usd_path = "/home/haoyu-ma/projects/OceanSim_utils/assets/usd/Pier/pier.usd"
        scene_usd_path = '/home/haoyu/isaacsim_assets/USD/Pier/pier.usd'
        add_reference_to_stage(usd_path=scene_usd_path, prim_path=scene_prim_path)
        SingleGeometryPrim(prim_path=scene_prim_path, collision=True)
        add_update_semantics(prim=get_prim_at_path('/World/Pier/Mesh'),
                             type_label='reflectivity',
                             semantic_label='1.0')

        
        # Load the robot
        robot_prim_path = "/World/rob"
        # robot_usd_path = '/home/haoyu/isaacsim_assets/USD/BlueRov/BROV2-HEAVY_0.5down.usd'
        # add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        DynamicCuboid(prim_path=robot_prim_path, size=0.2)
        
        # Toggle rigid body and collider preset for rob
        self._rob = get_prim_at_path(robot_prim_path)
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(self._rob)
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        rob_rigidBody_API.GetLinearDampingAttr().Set(0.0)
        rob_rigidBody_API.GetAngularDampingAttr().Set(0.0)
        rob_rigid_prim = SingleRigidPrim(prim_path=robot_prim_path,
                                         mass=self._rob_mass)
        
        rob_xform_path = robot_prim_path + '/rob_xform'
        SingleXFormPrim(rob_xform_path)
        
        # Attach the front camera
        cam_prim_path = rob_xform_path + '/UW_Camera'
        self._cam = UW_Camera(
            prim_path=cam_prim_path,
            resolution=self._cam_res,
            translation=[0.2,0.0,0.0],
            )
        self._cam.set_focal_length(0.1 * self._cam_focal)
        self._cam.set_clipping_range(0.1, 100)
        
        
        # Attach the forward looking imaging sonar
        sonar_prim_path = rob_xform_path + '/ImagingSonar'
        self._sonar = ImagingSonarSensor(prim_path=sonar_prim_path,
                                         translation=[0.5, 0.0, 0.1],
                                         max_range= 100,
                                         range_res= 0.5,
                                         hori_res=3000)

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by t setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._rob, self._sonar, self._cam)

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the posiheirtion it was in when it was created in self._setup_scene().
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
