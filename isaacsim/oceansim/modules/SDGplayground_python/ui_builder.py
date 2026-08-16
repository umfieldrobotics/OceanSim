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
import re
# Isaac sim import

from isaacsim.core.utils.stage import open_stage
from isaacsim.gui.components import CollapsableFrame, StateButton, IntField, get_style, combo_floatfield_slider_builder, Button, StringField, setup_ui_headers, str_builder, CheckBox, FloatField
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.core.utils.extensions import get_extension_path

from isaacsim.gui.property.array_widget import CustomMultiIntField, CustomMultiFloatField
# Custom import
from .scenario import SDGplayground_Scenario
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK
import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import Gf, Sdf, UsdGeom
from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path
from isaacsim.oceansim.utils.UWCam_sdg_utils import *
from isaacsim.oceansim.sensors.UW_Camera import UW_Camera
class UIBuilder:
    def __init__(self):
        self._ext_id = omni.kit.app.get_app().get_extension_manager().get_extension_id_by_module(__name__)
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link =  EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        self._extension_path = get_extension_path(self._ext_id)
        self._oceansim_assets_path = get_oceansim_assets_path()
        # UI frames created
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()
        # A flag indicating if the scenario is loaded at least once (helpful for UI module to see if scenario variables are created)

        # Run initialization for the provided example
        self._on_init()
        self._randomization_settings = {}
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
                    tooltip='Input the path to your terrain scene file',
                    default_val=self._oceansim_assets_path + "/sample_sdg/terrains_3x3/rocky_trail_8k/rocky_trail_8k.usd",
                    use_folder_picker=True,
                    folder_button_title='Select USD',
                    folder_dialog_title='Select USD scene to import',
                )
                self.object_folder_field = str_builder(
                    label="Object Folder",
                    tooltip="Directory containing object assets for randomization",
                    default_val=self._oceansim_assets_path + "/sample_sdg/objects/sea_urchin/",
                    use_folder_picker=True,
                    folder_button_title="Select Object Folder",
                    folder_dialog_title="Select folder with object assets",
                )
                self.object_folder_field.add_value_changed_fn(
                    lambda model, key="object_folder": self._on_field_change(
                        key, model.get_value_as_string()
                    )
                )
                self._on_field_change("object_folder", self.object_folder_field.get_value_as_string())
                with ui.HStack(style=get_style(), spacing=10, height=0):
                    self.object_count_field = IntField(
                        label="Count",
                        tooltip="Instances spawned per frame",
                        lower_limit=0,
                        default_value=7,
                    )
                self.object_count_field.set_on_value_changed_fn(
                    lambda value, key="object_count": self._on_field_change(key, value)
                )

                self.wrapped_ui_elements.extend(
                    [self.object_count_field]
                )
                self._on_field_change("object_count", self.object_count_field.get_value())

                self.distractor_folder_field = str_builder(
                    label="Distractor Folder",
                    tooltip="Directory containing distractor assets for randomization",
                    default_val=self._oceansim_assets_path + "/sample_sdg/objects/OceanRealm",
                    use_folder_picker=True,
                    folder_button_title="Select Distractor Folder",
                    folder_dialog_title="Select folder with distractor assets",
                )
                self.distractor_folder_field.add_value_changed_fn(
                    lambda model, key="distractor_folder": self._on_field_change(
                        key, model.get_value_as_string()
                    )
                )
                self._on_field_change(
                    "distractor_folder", self.distractor_folder_field.get_value_as_string()
                )
                with ui.HStack(style=get_style(), spacing=10, height=0):
                    self.distractor_count_field = IntField(
                        label="Count",
                        tooltip="Instances spawned per frame",
                        lower_limit=0,
                        default_value=2,
                    )
                self.distractor_count_field.set_on_value_changed_fn(
                    lambda value, key="distractor_count": self._on_field_change(key, value) 
                )

                self.wrapped_ui_elements.extend(
                    [
                        self.distractor_count_field,
                    ]
                )
                self._on_field_change("distractor_count", self.distractor_count_field.get_value())

                self.sampling_prim_field = StringField(
                    label="Sampling Mesh",
                    tooltip="USD prim path or mesh file used as the sampling surface",
                    default_value="/terrain/collider",
                )
                self.wrapped_ui_elements.append(self.sampling_prim_field)
                self.sampling_prim_field.set_on_value_changed_fn(
                    lambda value, key="sampling_prim": self._on_field_change(key, value)
                )
                self._on_field_change("sampling_prim", self.sampling_prim_field.get_value())
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
        
        
        randomization_control_frame = CollapsableFrame("Randomization Control", collapsed = False)
        self.frames.append(randomization_control_frame)
        with randomization_control_frame:
            with ui.VStack(style=get_style(), spacing=8, height=0):
                

                # Workspace bounds (min/max XYZ) for both target objects and distractors
                workspace_labels = ["Min X", "Min Y", "Min Z", "Max X", "Max Y", "Max Z"]

                def _build_workspace_grid(title, defaults, key_prefix, labels=None, columns=3):
                    labels = labels or workspace_labels
                    total_fields = len(defaults)
                    if total_fields == 0:
                        return []
                    rows = (total_fields + columns - 1) // columns
                    ui.Label(title, height=18, alignment=ui.Alignment.LEFT_CENTER)
                    fields = []
                    for row in range(rows):
                        with ui.HStack(spacing=10):
                            for col in range(columns):
                                idx = row * columns + col
                                if idx >= total_fields:
                                    break
                                label_name = labels[idx] if idx < len(labels) else f"Value {idx + 1}"
                                key_name = f"{key_prefix}_{self._normalize_label(label_name)}"
                                with ui.VStack(spacing=2, width=ui.Fraction(1 / columns)):
                                    ui.Label(label_name, height=16, alignment=ui.Alignment.LEFT_CENTER)
                                    field = ui.FloatField(precision=3, height=0)
                                    field.model.set_value(defaults[idx])
                                    field.tooltip = f"{label_name} for {title.lower()}"
                                    field.model.add_value_changed_fn(
                                        lambda model, key=key_name: self._on_field_change(
                                            key, model.get_value_as_float()
                                        )
                                    )
                                    self._on_field_change(key_name, defaults[idx])
                                    fields.append(field)
                    return fields

                obj_ws_defaults = [-2.5, -2.5, -1.0, 2.5, 2.5, 1.0]
                self._obj_ws_fields = _build_workspace_grid("Object Workspace", obj_ws_defaults, "obj_ws")

                distractor_ws_defaults = [-2.5, -2.5, -1.0, 2.5, 2.5, 1.0]
                self._distractor_ws_fields = _build_workspace_grid(
                    "Distractor Workspace", distractor_ws_defaults, "distractor_ws"
                )

                camera_ws_defaults = [-0.25, -0.25, 0.5, 0.25, 0.25, 1.75]
                self._camera_ws_fields = _build_workspace_grid("Camera Workspace", camera_ws_defaults, "camera_ws")

                camera_look_at_ws_defaults = [-1.5, -1.5, -1.0, 1.5, 1.5, 1.0]
                self._camera_look_at_ws_fields = _build_workspace_grid("Camera Look-at Workspace", camera_look_at_ws_defaults, "camera_look_at_ws")

                variation_labels = [
                    "Size Scale  +/-",
                    "Texture Scale  *//",
                    "Texture Bias  +/-",
                ]
                self._object_variation_fields = _build_workspace_grid(
                    "Object Variation",
                    [0.5, 10, 0.5],
                    "object_variation",
                    labels=variation_labels,
                )
                self._distractor_variation_fields = _build_workspace_grid(
                    "Distractor Variation",
                    [0.25, 10, 0.5],
                    "distractor_variation",
                    labels=variation_labels,
                )

                self._process_btn = Button(
                    text='Update Sampling',
                    label='Update',
                    tooltip="Click this button to regenerate the sampling points on the terrain based on the current sampling mesh and workspace settings. This does not respawn objects, so it can be used to adjust sampling without affecting current object placement.",
                    on_click_fn=self.process_sampling_mesh
                )
        
        
        color_picker_frame = CollapsableFrame('Color Picker', collapsed=False)
        self.frames.append(color_picker_frame)
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
            0.08, 0.42, 0.52,
            0.905, 0.961, 0.982,
            0.905*1.5, 0.961*1.5, 0.982*1.5,
        ]
        self._param = params_default
        with color_picker_frame:
            with ui.VStack(spacing=10):

                for i in range(9):
                    param_model, param_slider = combo_floatfield_slider_builder(
                        label=params_labels[i],
                        type=params_types[i],
                        default_val=params_default[i],
                        max=3.0)
                    self._param_models.append(param_model)
                    param_model.add_value_changed_fn(self._on_color_param_changes)
                    self._on_color_param_changes(param_model)
        
        run_scenario_frame = CollapsableFrame("Run Scenario", collapsed=False)
        self.frames.append(run_scenario_frame)
        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._spawn_button = Button(
                    label="Spawn Objects",
                    text="Spawn",
                    tooltip="Press this to clear and spawn objects",
                    on_click_fn= self._on_spawn,
                )
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
                self.save_dir_field = StringField(
                    label='Viewport PNG saving Path',
                    tooltip='Save the render parameter and reference pic into this directory',
                    use_folder_picker=True
                )
                self.wrapped_ui_elements.append(self.save_dir_field)
                save_viewport_button = Button(
                    text='Save viewport',
                    label='Save rendered image',
                    tooltip="Click this button to capture the current raw/rendered/depth image from viewport",
                    on_click_fn=self._on_save_viewport
                )
                self.wrapped_ui_elements.append(save_viewport_button)
    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters
        self._scenario = SDGplayground_Scenario()
        self._objects = []
        self._distractors = []
        self._cameras = []
        self._camera_properties =  {
            "focalLength": 24,
            "focusDistance": 400,
            "fStop": 0.0,
            "clippingRange": (0.001, 100),
        }

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        stage_utils.create_new_stage()

        self.load_terrain()
        self.load_objects()
        self.process_sampling_mesh(self._randomization_settings["sampling_prim"])
        self._on_spawn()

        stage = stage_utils.get_current_stage()
        create_dome_ligth(stage, "/Environment", intensity=1000.0)
        self._UW_cam = UW_Camera("/UW_Camera", resolution=(1920, 1080))
        for key, value in self._camera_properties.items():
            self._UW_cam.prim.GetAttribute(key).Set(value)
        self._cameras.append(self._UW_cam.prim)
                

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
        self._scenario.setup_scenario(self._UW_cam)

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
        self._scenario_state_btn.enabled= False
        self._reset_btn.enabled = False

    def _normalize_label(self, label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return normalized or "value"

    def _on_field_change(self, field_name, value):
        """Generic handler to keep UI-driven data in sync with runtime settings."""
        original_value = self._randomization_settings.get(field_name, None)
        self._randomization_settings[field_name] = value
        print(f"Field '{field_name}' changed from {original_value} to {value}.")

    def _on_color_param_changes(self, model):
        for i, param_model in zip(range(9), self._param_models):
            self._param[i] = param_model.get_value_as_float()
        if self._scenario._cam is not None: 
            self._scenario._cam._backscatter_value = wp.vec3f(*self._param[0:3])
            self._scenario._cam._atten_coeff = wp.vec3f(*self._param[6:9])
            self._scenario._cam._backscatter_coeff = wp.vec3f(*self._param[3:6]) 



    def _on_spawn(self):
        """Spawn objects and distractors on the terrain based on the current randomization settings."""
        sample_objects_on_points(self._obj_ws_points, self._objects, offset=(0, 0, 0.025))
        sample_objects_on_points(self._dist_ws_points, self._distractors)  
        object_size_var = self._randomization_settings["object_variation_size_scale"]
        distractor_size_var = self._randomization_settings["distractor_variation_size_scale"]
        perturb_object_poses(self._objects, scale_range=(1.0 - object_size_var, 1.0 + object_size_var))      
        # This bias from 0.25 instead of 1.0 is chosen based on the average size of the distractor assets, so that the randomization effect is more visually significant. Adjust as needed based on your specific assets.
        perturb_object_poses(self._distractors, scale_range=(max(0.1, 0.25 - distractor_size_var), 0.25 + distractor_size_var))
        randomize_UVTexture_scale_bias(self._object_uv_texture_shaders, 
                                       scale_range=(
                                           1.0 / self._randomization_settings["object_variation_texture_scale"],
                                           1.0 * self._randomization_settings["object_variation_texture_scale"],
                                       ),
                                       bias_range=(
                                           -self._randomization_settings["object_variation_texture_bias"],
                                           self._randomization_settings["object_variation_texture_bias"],
                                       ))
        randomize_UVTexture_scale_bias(self._distractor_uv_texture_shaders, 
                                       scale_range=(
                                           1.0 / self._randomization_settings["distractor_variation_texture_scale"],
                                           1.0 * self._randomization_settings["distractor_variation_texture_scale"],
                                       ),
                                       bias_range=(
                                           -self._randomization_settings["distractor_variation_texture_bias"],
                                           self._randomization_settings["distractor_variation_texture_bias"],
                                       ))
        randomize_camera_poses_rel_to_objs(self._cameras, 
                                           self._objects, 
                                           [self._randomization_settings[f"camera_look_at_ws_{bound}"] for bound in ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]], 
                                           [self._randomization_settings[f"camera_ws_{bound}"] for bound in ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]],
                                           look_at_offset=(-0.3, 0.3))
    
    
    def _on_save_viewport(self):
        if self._scenario_state_btn.enabled:
            if self.save_dir_field.get_value() != "":
                save_dir = self.save_dir_field.get_value()
                rendered_image = self._scenario._cam._uw_image.numpy()
                uw_image = Image.fromarray(rendered_image, 'RGBA')
                uw_image.save(save_dir + '/viewport_uw_rgba.png')
                print(f'viewport result written to {save_dir}.')
            else:

                carb.log_error('Saving directory is empty.')

        else:
            print('Load a scenario first.')


    def load_terrain(self):
        """
        Loads the terrain and initializes the workspace points.
        """
        stage = stage_utils.get_current_stage()
        stage_utils.add_reference_to_stage(self.scene_path_field.get_value_as_string(), 
                                               "/terrain")
        
        
    def load_objects(self):
        if not self._objects:
            try: 
                self._objects, _ = add_objects(objects_folder_path=self._randomization_settings["object_folder"], 
                                                override_semantic_mapping=None, 
                                                physics=True,
                                                count=self._randomization_settings["object_count"],
                                                )
                
                self._distractors, _ = add_distractor_from_UE(mapping={},
                                                UE_asset_folder=self._randomization_settings["distractor_folder"],
                                                root_path="SDG_distractors",
                                                name_prefix="distractor_",
                                                physics=True,
                                                num=30,
                                                count=self._randomization_settings["distractor_count"],
                                                )
            except Exception as e:
                print(f"Error loading objects: {e}")
                return

    def process_sampling_mesh(self, sampling_prim_path=None):
        """
        Processes the sampling mesh to extract points within the defined workspaces for objects and distractors.
        This function also traverse all the objects in the stage to get the UVtexture shader handles.
        """
        sampling_prim_path = sampling_prim_path or self._randomization_settings.get("sampling_prim")
        if not sampling_prim_path:
            print("Sampling prim path is not set; please update the sampling mesh path before processing.")
            return
        sampling_prim = stage_utils.get_current_stage().GetPrimAtPath(sampling_prim_path)
        if not sampling_prim.IsValid() or not sampling_prim.GetAttribute("points"):
            print("Invalid sampling prim or the prim is not a geometry or does not contain points attribute") 
            return
        points = UsdGeom.Mesh(sampling_prim).GetPointsAttr().Get()
        self._obj_ws_points = [
            point
            for point in points
            if self._randomization_settings["obj_ws_min_x"] <= point[0] <= self._randomization_settings["obj_ws_max_x"]
            and self._randomization_settings["obj_ws_min_y"] <= point[1] <= self._randomization_settings["obj_ws_max_y"]
            and self._randomization_settings["obj_ws_min_z"] <= point[2] <= self._randomization_settings["obj_ws_max_z"]
        ]
        self._dist_ws_points = [
            point
            for point in points
            if self._randomization_settings["distractor_ws_min_x"] <= point[0] <= self._randomization_settings["distractor_ws_max_x"]
            and self._randomization_settings["distractor_ws_min_y"] <= point[1] <= self._randomization_settings["distractor_ws_max_y"]
            and self._randomization_settings["distractor_ws_min_z"] <= point[2] <= self._randomization_settings["distractor_ws_max_z"]
        ]   


        objects_material_prims = get_material_prims(stage_utils.get_current_stage().GetPrimAtPath("/SDG_objects"))
        distractor_material_prims = get_material_prims(stage_utils.get_current_stage().GetPrimAtPath("/SDG_distractors"))
        self._object_uv_texture_shaders = list(chain.from_iterable([get_UsdUVTexture_shaders(prim) for prim in objects_material_prims]))
        self._distractor_uv_texture_shaders = list(chain.from_iterable([get_UsdUVTexture_shaders(prim) for prim in distractor_material_prims]))
