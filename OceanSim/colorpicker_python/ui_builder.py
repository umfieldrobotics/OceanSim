# Omniverse import
import numpy as np
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType

from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import warp as wp
# Isaac sim import
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.gui.components import CollapsableFrame, Frame, StateButton, get_style, combo_floatfield_slider_builder, Button
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid

# Custom import
from .scenario import MHL_colorpicker_Scenario
from ..utils.UWrenderer_utils import UW_render
from ..sensors.DVLsensor import DVLsensor

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
        self._is_annotator_loaded = False
        self._colorpicker_provider = ui.ByteImageProvider()  
        self._param = np.zeros(9)
        
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

        color_picker_frame = CollapsableFrame('Color Picker', collapsed=False)
        self._param_models = []
        params_labels = [                        
            "Backscatter_R", "Backscatter_G","Backscatter_B",
            "Backscatter_coeff_R", "Backscatter_coeff_G", "Backscatter_coeff_B",
            "Attenuation_coeff_R", "Attenuation_coeff_G", "Attenuation_coeff_B",
        ]
        params_types = [
            'float', 'float', 'float',
            'float', 'float', 'float',
            'float', 'float', 'float',
        ]
        params_default = [
            0.0, 0.31, 0.24,
            0.05, 0.05, 0.2,
            0.05, 0.05, 0.05
        ]

        with color_picker_frame:
            with ui.VStack(spacing=10):

                for i in range(9):
                    param_model, _ = combo_floatfield_slider_builder(
                        label=params_labels[i],
                        type=params_types[i],
                        default_val=params_default[i])
                    self._param_models.append(param_model)
                    param_model.add_value_changed_fn(self._on_color_param_changes)
                with ui.ZStack(height=500):
                    ui.Rectangle(style={"background_color": 0xFF000000})
                    ui.ImageWithProvider(self._colorpicker_provider)

                load_pic_button = Button(
                    label="Load the picture from active camera annotator",
                    text="Get picture",
                    on_click_fn=self._on_load_picture

                )
                
    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters

        self._rob_mass = 10 # kg


        # Camera parameters
        self._cam = None
        self._cam_res = (1920, 1080)
        self._cam_pose = ([0.5,0.0,0.0], [0.5,0.5,-0.5,-0.5])
        self._cam_focal = 6
        self._scenario = MHL_colorpicker_Scenario()


    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        # Open MHL scene
        open_stage("/home/haoyu-ma/projects/OceanSim_utils/demo/MHL_Water.usd")

        # Load the robot
        robot_prim_path = "/World/rob"
        robot_usd_path = '/home/haoyu-ma/projects/OceanSim_utils/assets/usd/BlueRov/BROV2-HEAVY_0.5down.usd'
        add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        # DynamicCuboid(prim_path=robot_prim_path, size=0.5, color=np.array([0.5,0.5,1]))
        # Load the rock
        # rock_prim_path = '/MHL/rock'
        # rock_usd_path = '/home/haoyu/isaacsim_assets/USD/3D model/rock/rock.usd'
        # add_reference_to_stage(usd_path=rock_usd_path, prim_path=rock_prim_path)
        
        # # Toggle rigid body and collider preset for rob and rock
        self._rob = get_prim_at_path(robot_prim_path)
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(self._rob)
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        rob_rigidBody_API.GetLinearDampingAttr().Set(0.0)
        rob_rigidBody_API.GetAngularDampingAttr().Set(0.0)
        rob_rigid_prim = SingleRigidPrim(prim_path=robot_prim_path,
                                         mass=self._rob_mass)
        
        # rock_collider_prim = SingleGeometryPrim(prim_path=rock_prim_path,
        #                    translation=np.array([0.0, 3.5, 0.5]),
        #                    orientation=euler_angles_to_quat(np.array([0.0,0.0,125]), degrees=True),
        #                    collision=True,
        #                    )
        # rock_collider_prim.set_collision_approximation('convexDecomposition')
        # rock_rigid_prim = SingleRigidPrim(prim_path=rock_prim_path)
        
        
        self._DVL = DVLsensor(elevation=30)
        self._DVL.attachDVL(rigid_body_path=get_prim_path(self._rob),
                            location=np.array([0.0,0.0,-0.1]))
        self._DVL.add_single_beam()
        self._DVL.add_debug_lines()
        # Attach the front camera
        cam_prim_path = robot_prim_path + '/Camera'
        self._cam = Camera(
            prim_path=cam_prim_path,
            resolution=self._cam_res,
            )
        self._cam.set_focal_length(0.1 * self._cam_focal)
        SingleXFormPrim(cam_prim_path).set_local_pose(translation=self._cam_pose[0], orientation=self._cam_pose[1])
        

        MHLMesh_prim_path = '/World/mhl_scaled/Mesh/mesh'
        SingleGeometryPrim(prim_path=MHLMesh_prim_path, collision=True)


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
        self._scenario.setup_scenario(self._rob, self._cam, self._DVL)

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
        self._scenario.update_scenario(step, self._param)

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
        self._scenario.save()

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


    def _on_load_picture(self):
        if not self._is_annotator_loaded:
            try:
                self._rgba_annot = self._scenario._ldr
                self._depth_annot = self._scenario._depth
                self._is_annotator_loaded = True
                print('Annotator is loaded.')
            except:
                print('Annnotator not created. Load the scece first.')
                return
        if self._is_annotator_loaded:
            try:
                self._raw_rgba = self._rgba_annot.get_data()
                self._depth_image = self._depth_annot.get_data()
                self._colorpicker_provider.set_bytes_data_from_gpu(self._raw_rgba.ptr, [self._raw_rgba.shape[1], self._raw_rgba.shape[0]])
                print('If image generated is distorted. Please run the scene and click again.')
            except:
                print('Image capture failed. Please run the scene and click again.')

        
    
    def _on_color_param_changes(self, model):
        for i, param_model in zip(range(9), self._param_models):
            self._param[i] = param_model.get_value_as_float()    
        backscatter_value = wp.vec3f(*self._param[0:3])
        atten_coeff = wp.vec3f(*self._param[6:9])
        backscatter_coeff = wp.vec3f(*self._param[3:6])
        uw_image = wp.zeros_like(self._raw_rgba)
        wp.launch(
            dim=(self._raw_rgba.shape[0], self._raw_rgba.shape[1]),
            kernel=UW_render,
            inputs=[
                self._raw_rgba,
                self._depth_image,
                backscatter_value,
                atten_coeff,
                backscatter_coeff
            ],
            outputs=[
                uw_image
            ]
        )  

        self._colorpicker_provider.set_bytes_data_from_gpu(uw_image.ptr, [uw_image.shape[1], uw_image.shape[0]])
