# Omniverse import
import numpy as np
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
import warp as wp
import yaml
from PIL import Image
import carb
import os
# Isaac sim import

from isaacsim.core.utils.stage import open_stage
from isaacsim.gui.components import CollapsableFrame, StateButton, IntField, get_style, combo_floatfield_slider_builder, Button, StringField, setup_ui_headers, str_builder, CheckBox, FloatField
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.core.utils.extensions import get_extension_path

from isaacsim.gui.property.array_widget import CustomMultiIntField
# Custom import
from .scenario import TerrainInstancer_Scenario
from isaacsim.oceansim.utils.UWrenderer_utils import UW_render
from isaacsim.oceansim.watersurface import WaterSurface
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK
import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import Gf, Sdf, UsdGeom
class UIBuilder:
    def __init__(self):
        self._ext_id = omni.kit.app.get_app().get_extension_manager().get_extension_id_by_module(__name__)
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link =  EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        self._extension_path = get_extension_path(self._ext_id)

        # UI frames created
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()
        # A flag indicating if the scenario is loaded at least once (helpful for UI module to see if scenario variables are created)

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
        for frame in self.frames:
            frame.cleanup()

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
        self.frames.append(world_controls_frame)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self.scene_path_field = str_builder(
                    label='Path to USD',
                    tooltip='Input the path to your USD scene file',
                    default_val="",
                    use_folder_picker=True,
                    folder_button_title='Select USD',
                    folder_dialog_title='Select USD scene to import',
                    on_clicked_fn=lambda x: print("Click load for instancing new usd mesh")
                )
                
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
        self.frames.append(run_scenario_frame)
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

        instancer_frame = CollapsableFrame("Instancer on Grid", collapsed=False)
        self.frames.append(instancer_frame)
        with instancer_frame:
            with ui.VStack(style=get_style(), spacing=5, height = 0):
                self._x_span_int_field = FloatField(
                    label="X span",
                    default_value=10.0,
                    lower_limit=0.01,
                    on_value_changed_fn=self._on_meshList_UI_change
                )
                self._y_span_int_field = FloatField(
                    label="Y span",
                    default_value=10.0,
                    lower_limit= 0.01,
                    on_value_changed_fn=self._on_meshList_UI_change

                )
                self._x_count_int_field = IntField(
                    label="X counte",
                    default_value=5,
                    lower_limit=1,
                    on_value_changed_fn=self._on_meshList_UI_change
                )
                self._y_count_int_field = IntField(
                    label="Y count",
                    default_value=5,
                    lower_limit=1,
                    on_value_changed_fn=self._on_meshList_UI_change
                )
                self.wrapped_ui_elements.append(self._x_count_int_field)
                self.wrapped_ui_elements.append(self._y_count_int_field)
                self.wrapped_ui_elements.append(self._x_span_int_field)
                self.wrapped_ui_elements.append(self._y_span_int_field)

    def _on_meshList_UI_change(self, value):
        self._set_instancer()

    def _set_instancer(self):

        x_span = [-self._x_span_int_field.get_value(), self._x_span_int_field.get_value()]
        y_span = [-self._y_span_int_field.get_value(), self._y_span_int_field.get_value()]
        num_instances = [self._x_count_int_field.get_value(), self._y_count_int_field.get_value()]
        x_pos = np.linspace(x_span[0], y_span[1], num_instances[0])
        y_pos = np.linspace(y_span[0], y_span[1], num_instances[1])
        z = 1.0
        xs, ys = np.meshgrid(x_pos, y_pos)
        mMeshIndices = []
        mPositions = []
        mOrientations = []
        mLinearVelocities = []
        mAngularVelocities = []
        for i in range(num_instances[1]):
            for j in range(num_instances[0]):
                mMeshIndices.append(0)
                mPositions.append(Gf.Vec3f(xs[i][j], ys[i][j], z))
                mOrientations.append(Gf.Quath(1.0, 0.0, 0.0, 0.0))
                mLinearVelocities.append(Gf.Vec3f(0.0))
                mAngularVelocities.append(Gf.Vec3f(0.0))

        if self.shapeList:
            print("set")
            self.shapeList.GetProtoIndicesAttr().Set(mMeshIndices)
            self.shapeList.GetPositionsAttr().Set(mPositions)
            self.shapeList.GetOrientationsAttr().Set(mOrientations)
            self.shapeList.GetVelocitiesAttr().Set(mLinearVelocities)
            self.shapeList.GetAngularVelocitiesAttr().Set(mAngularVelocities)            

    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters
        self.shapeList = None
        self._scenario = TerrainInstancer_Scenario()


    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        stage_utils.create_new_stage()
        stage_utils.open_stage(self.scene_path_field.get_value_as_string())
        # test_scene_usd_path = self.scene_path_field.get_value_as_string()
        stage = stage_utils.get_current_stage()
        defaultPrimPath = '/root'
        # stage_utils.add_reference_to_stage(test_scene_usd_path, defaultPrimPath)

        geomPointInstancerPath = defaultPrimPath + "/pointinstancer"
        MeshActorPath = defaultPrimPath + "/model"

        # stage_utils.add_reference_to_stage(test_scene_usd_path, MeshActorPath)

        self.shapeList = UsdGeom.PointInstancer.Define(stage, Sdf.Path(geomPointInstancerPath))
        meshList = self.shapeList.GetPrototypesRel()
        # add mesh reference to point instancer
        meshList.AddTarget(Sdf.Path(MeshActorPath))

        self._set_instancer()

                
                

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
        self._scenario.setup_scenario()

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



    
    
    
    def _on_save_traj(self):
        if self.save_dir_field.get_value() != "":
            save_dir = self.save_dir_field.get_value()
            npy_path = save_dir + f"{self.file_name_field.get_value()}.npy"

            try:
                np.save(arr = self._scenario.recorded_position, file=npy_path)
                print(f"Recorded Trajectories written to {npy_path}")
            except yaml.YAMLError as e:
                print(f"Error writing npy file: {e}")
        else:
            carb.log_error('Saving directory is empty.')


