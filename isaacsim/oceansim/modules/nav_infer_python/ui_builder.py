# Omniverse import
import carb.settings
import numpy as np
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt
import omni.kit.commands
import carb
import omni.physx.scripts.utils as physicsBaseUtils

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, create_new_stage, open_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.gui.components import CollapsableFrame, Frame, StateButton, get_style, setup_ui_headers, StringField
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
import isaacsim.core.utils.prims as prims_utils
import os
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK
from isaacsim.core.utils.extensions import get_extension_path
from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path
import isaacsim.core.utils.viewports as viewports_utils
from omni.kit.viewport.utility import get_active_viewport


from .scenario import NavScenario


class UIBuilder:
    def __init__(self):
        self._ext_id = omni.kit.app.get_app().get_extension_manager().get_extension_id_by_module(__name__)
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link =  EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        self._extension_path = get_extension_path(self._ext_id)
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
        setup_ui_headers(
            ext_id=self._ext_id,
            file_path=self._file_path,
            title=self._title,
            doc_link=self._doc_link,
            overview=self._overview,
            info_collapsed=False
        )
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
        # Custom import
        self._scenario = NavScenario()
        # Robot parameters
        self._rob_mass = 5.0 # kg
        self._rob_angular_damping = 10.0
        self._rob_linear_damping = 10.0


        

    
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

        create_new_stage()
        add_reference_to_stage(prim_path="/Env", usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
        # add_reference_to_stage(prim_path="/Env", usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd")
    

        robot_prim_path = "/rob"
        robot_usd_path = get_oceansim_assets_path() + "/Bluerov/BROV_low.usd"
        rob_prim = add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        # Toggle rigid body and collider preset for robot, and set zero gravity to mimic underwater environment
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(rob_prim)
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        # Set damping of the robot
        rob_rigidBody_API.GetLinearDampingAttr().Set(self._rob_linear_damping)
        rob_rigidBody_API.GetAngularDampingAttr().Set(self._rob_angular_damping)
        physicsBaseUtils.set_local_space_velocities(rob_prim, True)

        # Set the mass for the robot to suppress a warning from inertia autocomputation
        yaw = np.random.uniform(low = -180, high = 180)
        rob_collider_prim = SingleGeometryPrim(prim_path=robot_prim_path,
                                               position=[-8.0, 10.0, 0.25],
                                               orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw]), degrees=True, extrinsic=False),
                                               collision=True)
        rob_collider_prim.set_collision_approximation('boundingCube')
        self._rob = SingleRigidPrim(prim_path=robot_prim_path,
                        mass=self._rob_mass,
                        )
        camera_prim_path = robot_prim_path + "/cam"
        self._cam = Camera(prim_path=camera_prim_path,
                           translation=[0.2, 0.0, 0.15],
                           resolution=(96, 96), # default nomad image size
                           )
        self._cam.set_clipping_range(near_distance=0.01)
        self._cam.set_projection_type("fisheyeKannalaBrandtK3")

        viewport_api = get_active_viewport()
        viewport_api.set_active_camera(camera_path=camera_prim_path)
        # set_camera_view(eye=np.array([-10, 12, 1]), target=rob_collider_prim.get_world_pose()[0])


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
        self._scenario.setup_scenario(self._rob, self._cam)

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
