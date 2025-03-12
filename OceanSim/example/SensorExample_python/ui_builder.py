# Omniverse import
import numpy as np
import os
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import carb
# Isaac sim import
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, create_new_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.gui.components import CollapsableFrame, Frame, StateButton, get_style, setup_ui_headers, CheckBox, xyz_plot_builder, combo_cb_xyz_plot_builder
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.core.utils.extensions import get_extension_path, get_extension_id, get_extension_path_from_name
# Custom import
from ...sensors.ImagingSonarSensor import ImagingSonarSensor
from ...sensors.UW_Camera import UW_Camera
from ...sensors.DVLsensor import DVLsensor
from ...sensors.BarometerSensor import BarometerSensor
from .scenario import MHL_Sensor_Example_Scenario
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK
from ...utils.assets_utils import get_OceanSim_assets_path
class UIBuilder():
    def __init__(self):

        self._ext_id = get_extension_id('OceanSim')
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link =  EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        
        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

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
        self._DVL_event_sub = None
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """

        setup_ui_headers(
            self._ext_id, self._file_path, self._title, self._doc_link, self._overview, info_collapsed=False
        )
        
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                sonar_check_box = CheckBox(
                    "Imaging Sonar",
                    default_value=False,
                    tooltip=" Click this checkbox to activate imaging sonar",
                    on_click_fn=self._on_sonar_checkbox_click_fn,
                )
                self._use_sonar = False
                self.wrapped_ui_elements.append(sonar_check_box)
                camera_check_box = CheckBox(
                    "Underwater Camera",
                    default_value=False,
                    tooltip=" Click this checkbox to activate underwater camera",
                    on_click_fn=self._on_camera_checkbox_click_fn,
                )
                self._use_camera = False
                self.wrapped_ui_elements.append(camera_check_box)

                DVL_check_box = CheckBox(
                    'DVL',
                    default_value=False,
                    tooltip=" Click this checkbox to activate DVL",
                    on_click_fn=self._on_DVL_checkbox_click_fn
                )
                self._use_DVL = False
                self.wrapped_ui_elements.append(DVL_check_box)

                baro_check_box = CheckBox(
                    "Barometer",
                    default_value=False,
                    tooltip='Click this checkbox to activate barometer',
                    on_click_fn=self._on_baro_checkbox_click_fn
                ) 
                self._use_baro = False
                self.wrapped_ui_elements.append(baro_check_box)


                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                # self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
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

        self.sensor_reading_frame = CollapsableFrame('Sensor Reading', collapsed=False)





    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters
        self._rob_mass = 5.0 # kg
        self._rob_angular_damping = 10.0
        self._rob_linear_damping = 10.0

        # Sensor
        self._sonar = None
        self._sonar_trans = np.array([0.5,0.0, 0.0])
        self._cam = None
        self._cam_trans = self._sonar_trans
        self._cam_focal_length = 21
        self._DVL = None
        self._DVL_trans = np.array([0,0,-0.1])
        self._baro = None
        self._water_surface = 1.43389 # Read from USD scene
        
        # Scenario
        self._scenario = MHL_Sensor_Example_Scenario()


    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        
        # add MHL scene as reference
        MHL_prim_path = '/World/mhl'
        MHL_usd_path = get_OceanSim_assets_path() + "/collected_MHL/mhl_water.usd"
        add_reference_to_stage(usd_path=MHL_usd_path, prim_path=MHL_prim_path)
        # Toggle MHL mesh's collider
        SingleGeometryPrim(prim_path=MHL_prim_path, collision=True)
        # apply a reflectivity of 1.0 for sonar simulation
        add_update_semantics(prim=get_prim_at_path(MHL_prim_path),
                             type_label='reflectivity',
                             semantic_label='1.0')
        
        
        # add bluerov robot as reference
        robot_prim_path = "/World/rob"
        robot_usd_path = get_OceanSim_assets_path() + "/Bluerov/BROV_low.usd"
        self._rob = add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        # Toggle rigid body and collider preset for robot, and set zero gravity to mimic underwater environment
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_prim_path))
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        # Set damping of the robot
        rob_rigidBody_API.GetLinearDampingAttr().Set(self._rob_linear_damping)
        rob_rigidBody_API.GetAngularDampingAttr().Set(self._rob_angular_damping)
        # Set the mass for the robot to suppress a warning from inertia autocomputation
        SingleRigidPrim(prim_path=robot_prim_path,
                        mass=self._rob_mass)
        
        # Load the rock
        rock_prim_path = '/World/rock'
        rock_usd_path = get_OceanSim_assets_path() + "/collected_rock/rock.usd"
        rock_prim = add_reference_to_stage(usd_path=rock_usd_path, prim_path=rock_prim_path)
        # apply a reflectivity of 2.0 for sonar simulation
        add_update_semantics(prim=rock_prim,
                             type_label='reflectivity',
                             semantic_label='2.0')
        # Toggle collider for the rock
        rock_collider_prim = SingleGeometryPrim(prim_path=rock_prim_path,
                           translation=np.array([1.0, 0.1, -1.5]),
                           orientation=euler_angles_to_quat(np.array([0.0,0.0,90]), degrees=True),
                           collision=True,
                           )
        # Set collision approximation using convexDecomposition to automatically compute inertia matrix
        rock_collider_prim.set_collision_approximation('convexDecomposition')
        # Toggle rigid body for the rock
        rock_rigid_prim = SingleRigidPrim(prim_path=rock_prim_path )
        


        if self._use_sonar:
            self._sonar = ImagingSonarSensor(prim_path=robot_prim_path + '/sonar',
                                            translation=self._sonar_trans,
                                            range_res=0.005,
                                            angular_res=0.25,
                                            )
            
        if self._use_camera:
            self._cam = UW_Camera(prim_path=robot_prim_path + '/UW_camera',
                                    resolution=[1920,1080],
                                    translation=self._cam_trans)
            self._cam.set_focal_length(0.1 * self._cam_focal_length)
            self._cam.set_clipping_range(0.1, 100)
            
        if self._use_DVL:
            self._DVL = DVLsensor(max_range=0.3)
            self._DVL.attachDVL(rigid_body_path=robot_prim_path,
                                location=self._DVL_trans)
            self._DVL.add_debug_lines()
            
        if self._use_baro:
            self._baro = BarometerSensor(prim_path=robot_prim_path + '/Baro',
                                        water_surface_z=self._water_surface)
            


    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by t setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._reset_scenario()
        self._add_extra_ui()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._rob, self._sonar, self._cam, self._DVL, self._baro)
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


    def _on_sonar_checkbox_click_fn(self, model):
        self._use_sonar = model
        print('Reload the scene for changes to take effect.')

    def _on_camera_checkbox_click_fn(self, model):
        self._use_camera = model
        print('Reload the scene for changes to take effect.')

    def _on_DVL_checkbox_click_fn(self, model):
        self._use_DVL = model
        print('Reload the scene for changes to take effect.')

    def _on_baro_checkbox_click_fn(self, model):
        self._use_baro = model
        print('Reload the scene for changes to take effect.')
    
    def _add_extra_ui(self):
        if self._use_DVL is True:
            with self.sensor_reading_frame:
                self._build_DVL_plot_frame()

    def _build_DVL_plot_frame(self):
        self._DVL_event_sub = None
        self._DVL_x_vel = []
        self._DVL_y_vel = []
        self._DVL_z_vel = []

        kwargs = {
            "label": "DVL reading xyz vel (m/s)",
            "on_clicked_fn": self.toggle_DVL_step,
            "data": [self._DVL_x_vel, self._DVL_y_vel, self._DVL_z_vel],
        }
        (
            self._DVL_plot,
            self._DVL_plot_value,
        ) = combo_cb_xyz_plot_builder(**kwargs)
    def toggle_DVL_step(self, val=None):
        print("You've cliked DVL time_series_plot_data:", val)
        if val:
            if not self._DVL_event_sub:
                self._DVL_event_sub = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_DVL_step)
                )
            else:
                self._DVL_event_sub = None
        else:
            self._DVL_event_sub = None

    def _on_DVL_step(self, e: carb.events.IEvent):
        x_vel = float(self._scenario._DVL_reading[0])
        y_vel = float(self._scenario._DVL_reading[1])
        z_vel = float(self._scenario._DVL_reading[2])

        self._DVL_plot_value[0].set_value(x_vel)
        self._DVL_plot_value[1].set_value(y_vel)
        self._DVL_plot_value[2].set_value(z_vel)

        self._DVL_x_vel.append(x_vel)
        self._DVL_y_vel.append(y_vel)
        self._DVL_z_vel.append(z_vel)
        if len(self._DVL_x_vel) > 50:
            self._DVL_x_vel.pop(0)
            self._DVL_y_vel.pop(0)
            self._DVL_z_vel.pop(0)

        self._DVL_plot[0].set_data(*self._DVL_x_vel)
        self._DVL_plot[1].set_data(*self._DVL_y_vel)
        self._DVL_plot[2].set_data(*self._DVL_z_vel)
