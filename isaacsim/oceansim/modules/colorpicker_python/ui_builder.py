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
from isaacsim.gui.components import CollapsableFrame, StateButton, get_style, combo_floatfield_slider_builder, Button, StringField, setup_ui_headers, str_builder
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.core.utils.extensions import get_extension_path


# Custom import
from .scenario import Colorpicker_Scenario
from isaacsim.oceansim.utils.UWrenderer_utils import UW_render
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK


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

        demo_image_path = self._extension_path + "/demo/demo_rgb.png"
        demo_depth_path = self._extension_path + '/demo/demo_depth.npy'
        demo_image = Image.open(demo_image_path).convert('RGBA')
        self._demo_rgba = wp.array(data=np.array(demo_image), dtype=wp.uint8, ndim=3)      
        self._demo_depth = wp.array(data=np.load(file=demo_depth_path),
                              dtype=wp.float32,
                              ndim=2)
        self._demo_res = [self._demo_rgba.shape[1], self._demo_rgba.shape[0]]
        self._demo_provider = ui.ByteImageProvider()
        self._demo_provider.set_bytes_data_from_gpu(self._demo_rgba.ptr, self._demo_res)  
        self._uw_image = None
        self._param = np.zeros(9)

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
                    folder_dialog_title='Select USD scene to import'
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
            0.0, 0.31, 0.24,
            0.05, 0.05, 0.2,
            0.05, 0.05, 0.05
        ]
        self._param = params_default
        with color_picker_frame:
            with ui.VStack(spacing=10):

                for i in range(9):
                    param_model, param_slider = combo_floatfield_slider_builder(
                        label=params_labels[i],
                        type=params_types[i],
                        default_val=params_default[i])
                    self._param_models.append(param_model)
                    param_model.add_value_changed_fn(self._on_color_param_changes)
                    self._on_color_param_changes(param_model)
                with ui.ZStack(height=300):
                    ui.Rectangle(style={"background_color": 0xFF000000})
                    ui.ImageWithProvider(self._demo_provider,
                                         style={'alignment': ui.Alignment.CENTER,
                                                "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT})
                self.save_dir_field = StringField(
                    label='YAML saving Path',
                    tooltip='Save the render parameter and reference pic into this directory',
                    use_folder_picker=True
                )
                
                self.wrapped_ui_elements.append(self.save_dir_field)
                self.file_name_field = StringField(
                    label='File name',
                    tooltip='Label your yaml file',
                    default_value='render_param_0'
                )
                save_button = Button(
                    text="Save param",
                    label='Save render params',
                    tooltip='Click this button to save the current render parameters',
                    on_click_fn=self._on_save_param
                    )
                save_viewport_button = Button(
                    text='Save viewport',
                    label='Save rendered image',
                    tooltip="Click this button to capture the current raw/rendered/depth image from viewport",
                    on_click_fn=self._on_save_viewport
                )
                
        self.wrapped_ui_elements.append(self.file_name_field)
        self.wrapped_ui_elements.append(save_button)
        self.wrapped_ui_elements.append(save_viewport_button)
                
    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters

        self._scenario = Colorpicker_Scenario()


    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        try: 
            open_stage(self.scene_path_field.get_value_as_string())
            print('USD scene is loaded.')
        except:
            print('Path is not valid or scene can not be opened. Default to current stage')



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



    
    def _on_color_param_changes(self, model):
        for i, param_model in zip(range(9), self._param_models):
            self._param[i] = param_model.get_value_as_float()    
        self._update_demo_render()



    def _update_demo_render(self):

            self._uw_image = wp.zeros_like(self._demo_rgba)
            wp.launch(
                dim=np.flip(self._demo_res),
                kernel=UW_render,
                inputs=[
                    self._demo_rgba,
                    self._demo_depth,
                    wp.vec3f(*self._param[0:3]),
                    wp.vec3f(*self._param[6:9]),
                    wp.vec3f(*self._param[3:6])
                ],
                outputs=[
                    self._uw_image
                ]
            )  
            
            self._demo_provider.set_bytes_data_from_gpu(self._uw_image.ptr, self._demo_res)
    
    def _on_save_param(self):
        if self.save_dir_field.get_value() != "":
            data = {
                "backscatter_value":self._param[0:3],
                'atten_coeff': self._param[6:9],
                'backscatter_coeff': self._param[3:6]
                }
            save_dir = self.save_dir_field.get_value()
            yaml_path = save_dir + f"{self.file_name_field.get_value()}.yaml"
            png_path = save_dir + f"{self.file_name_field.get_value()}.png"
            with open(yaml_path, 'w') as file:
                try:
                    yaml.dump(data, file, sort_keys=False)
                    output_demo_image = Image.fromarray(self._uw_image.numpy(), 'RGBA')
                    output_demo_image.save(png_path)
                    print(f"Underwater render parameters written to {yaml_path}")
                except yaml.YAMLError as e:
                    print(f"Error writing YAML file: {e}")
        else:
            carb.log_error('Saving directory is empty.')

    def _on_save_viewport(self):
        if self._scenario_state_btn.enabled:
            if self.save_dir_field.get_value() != "":
                save_dir = self.save_dir_field.get_value()
                raw_rgba = self._scenario.raw_rgba.numpy()
                depth = self._scenario.depth_image.numpy()
                rendered_image = self._scenario.uw_image.numpy()
                np.save(file=save_dir + '/viewport_depth.npy', arr=depth)
                raw_image = Image.fromarray(raw_rgba, 'RGBA')
                uw_image = Image.fromarray(rendered_image, 'RGBA')
                raw_image.save(save_dir + '/viewport_raw_rgba.png')
                uw_image.save(save_dir + '/viewport_uw_rgba.png')
                print(f'viewport result written to {save_dir}.')
            else:

                carb.log_error('Saving directory is empty.')

        else:
            print('Load a scenario first.')