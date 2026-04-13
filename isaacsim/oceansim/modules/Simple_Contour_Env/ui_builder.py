# Omniverse import
import os

import carb
import numpy as np
import omni.timeline
import omni.ui as ui

# Isaac sim import
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.extensions import get_extension_path
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage,
    get_current_stage,
    open_stage,
)
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components import (
    CheckBox,
    CollapsableFrame,
    StateButton,
    combo_cb_plot_builder,
    combo_cb_xyz_plot_builder,
    dropdown_builder,
    get_style,
    setup_ui_headers,
    str_builder,
)

# ROS2 integ
from isaacsim.oceansim.sensors import ros2_helpers  # move to utils
from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path
from omni.usd import StageEventType
from pxr import PhysxSchema

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_LINK, EXTENSION_TITLE

# Custom import
from .scenario import MHL_Sensor_Example_Scenario


class UIBuilder:
    def __init__(self):

        self._ext_id = (
            omni.kit.app.get_app()
            .get_extension_manager()
            .get_extension_id_by_module(__name__)
        )
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link = EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        self._extension_path = get_extension_path(self._ext_id)

        self._ctrl_mode = "Manual control"
        self._waypoints_path = self._extension_path + "/demo/demo_waypoints.txt"
        self._timeline = omni.timeline.get_timeline_interface()

        # UI frames created
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        self._DVL_event_sub = None
        self._baro_event_sub = None
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        for frame in self.frames:
            frame.cleanup()

    def build_ui(self):

        setup_ui_headers(
            ext_id=self._ext_id,
            file_path=self._file_path,
            title=self._title,
            doc_link=self._doc_link,
            overview=self._overview,
            info_collapsed=False,
        )

        sensor_choosing_frame = CollapsableFrame("Sensors", collapsed=False)
        self.frames.append(sensor_choosing_frame)
        with sensor_choosing_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                sonar_check_box = CheckBox(
                    "Imu",
                    default_value=False,
                    tooltip=" Click this checkbox to activate Imu",
                    on_click_fn=self._on_imu_checkbox_click_fn,
                )
                self._use_imu = False

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
                    "DVL",
                    default_value=False,
                    tooltip=" Click this checkbox to activate DVL",
                    on_click_fn=self._on_DVL_checkbox_click_fn,
                )
                self._use_DVL = False
                self.wrapped_ui_elements.append(DVL_check_box)

                baro_check_box = CheckBox(
                    "Barometer",
                    default_value=False,
                    tooltip="Click this checkbox to activate barometer",
                    on_click_fn=self._on_baro_checkbox_click_fn,
                )

                self._use_baro = False
                self.wrapped_ui_elements.append(baro_check_box)

        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)
        self.frames.append(world_controls_frame)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._USD_path_field = str_builder(
                    label="Path to USD",
                    default_val="",
                    tooltip="Select the USD file for the scene",
                    use_folder_picker=True,
                    folder_button_title="Select USD",
                    folder_dialog_title="Select the USD scene to test",
                )

                self._ctrl_mode_model = dropdown_builder(
                    label="Control Mode",
                    default_val=3,
                    items=[
                        "No control",
                        "Straight line",
                        "Waypoints",
                        "Manual control",
                    ],
                    tooltip="Select preferred control mode",
                    on_clicked_fn=self._on_ctrl_mode_dropdown_clicked,
                )

                self._load_btn = LoadButton(
                    "Load Button",
                    "LOAD",
                    setup_scene_fn=self._setup_scene,
                    setup_post_load_fn=self._setup_scenario,
                )
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button",
                    "RESET",
                    pre_reset_fn=None,
                    post_reset_fn=self._on_post_reset_btn,
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

        self.sensor_reading_frame = CollapsableFrame(
            "Sensor Reading", collapsed=False, visible=False
        )
        self.frames.append(self.sensor_reading_frame)
        self.waypoints_frame = CollapsableFrame(
            "Waypoints", collapsed=False, visible=False
        )
        self.frames.append(self.waypoints_frame)

    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Robot parameters
        self._rob_mass = 5.0  # kg
        self._rob_angular_damping = 10.0
        self._rob_linear_damping = 10.0

        # Sensor
        self._imu = None
        self._sonar = None
        self._sonar_trans = np.array([0.3, 0.0, 0.3])
        self._cam = None
        self._cam_trans = np.array([0.3, 0.0, 0.1])
        self._cam_focal_length = 21
        self._DVL = None
        self._DVL_trans = np.array([0, 0, -0.1])
        self._baro = None
        self._water_surface = 1.43389

        # Scenario
        self._scenario = MHL_Sensor_Example_Scenario()

    def _setup_scene(self):
        create_new_stage()
        if self._USD_path_field.get_value_as_string() != "":
            scene_prim_path = "/World/scene"
            add_reference_to_stage(
                usd_path=self._USD_path_field.get_value_as_string(),
                prim_path=scene_prim_path,
            )
            print("User USD scene is loaded.")
        else:
            print("USD path is empty. Default to simple_contour scene")

            # Simple contour scene
            scene_prim_path = "/World/simple_contour"
            scene_usd_path = (
                get_oceansim_assets_path()
                + "/simple_contour/simple_contour_extra_bright.usd"
            )  # simple_contour.usd for gloomier look
            add_reference_to_stage(usd_path=scene_usd_path, prim_path=scene_prim_path)
            SingleGeometryPrim(prim_path=scene_prim_path, collision=True)
            add_update_semantics(
                prim=get_prim_at_path(scene_prim_path),
                type_label="reflectivity",
                semantic_label="1.0",
            )

        # add bluerov robot as reference
        robot_prim_path = "/World/rob"
        robot_usd_path = get_oceansim_assets_path() + "/Bluerov/BROV_low.usd"
        self._rob = add_reference_to_stage(
            usd_path=robot_usd_path, prim_path=robot_prim_path
        )
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(
            get_prim_at_path(robot_prim_path)
        )
        rob_rigidBody_API.CreateDisableGravityAttr(True)
        rob_rigidBody_API.GetLinearDampingAttr().Set(self._rob_linear_damping)
        rob_rigidBody_API.GetAngularDampingAttr().Set(self._rob_angular_damping)
        rob_collider_prim = SingleGeometryPrim(
            prim_path=robot_prim_path, collision=True
        )
        rob_collider_prim.set_collision_approximation("boundingCube")
        SingleRigidPrim(
            prim_path=robot_prim_path,
            mass=self._rob_mass,
            translation=np.array([-8.0, -3.25, 1.4]),
        )

        set_camera_view(
            eye=np.array([5, 0.6, 0.4]), target=rob_collider_prim.get_world_pose()[0]
        )

        if self._use_imu:
            from isaacsim.oceansim.sensors.ImuSensor_ROS import ImuSensor_ROS

            self._imu = ImuSensor_ROS(
                prim_path=robot_prim_path + "/imu",
                name="Imu",
                frequency=60,
                translation=np.array([0, 0, 0]),
            )

        if self._use_sonar:
            from isaacsim.oceansim.sensors.ImagingSonarSensor_ROS import (
                ImagingSonarSensor_ROS,
            )

            self._sonar = ImagingSonarSensor_ROS(
                prim_path=robot_prim_path + "/sonar",
                translation=self._sonar_trans,
                orientation=euler_angles_to_quat(
                    np.array([0.0, 45, 0.0]), degrees=True
                ),
                range_res=0.005,
                angular_res=0.25,
                hori_res=4000,
            )

        if self._use_camera:
            from isaacsim.oceansim.sensors.UW_Camera_ROS import UW_Camera_ROS

            self._cam = UW_Camera_ROS(
                prim_path=robot_prim_path + "/UW_camera",
                resolution=[1920, 1080],
                translation=self._cam_trans,
                orientation=euler_angles_to_quat(
                    np.array([0.0, 45, 0.0]), degrees=True
                ),
            )
            self._cam.set_focal_length(0.1 * self._cam_focal_length)
            self._cam.set_clipping_range(0.1, 100)
            approx_freq = 30

        if self._use_DVL:
            from isaacsim.oceansim.sensors.DVLSensor_ROS import DVLSensor_ROS

            self._DVL = DVLSensor_ROS(max_range=10)
            self._DVL.attachDVL(
                rigid_body_path=robot_prim_path, translation=self._DVL_trans
            )
            self._DVL.add_debug_lines()

        if self._use_baro:
            from isaacsim.oceansim.sensors.BarometerSensor_ROS import (
                BarometerSensor_ROS,
            )

            self._baro = BarometerSensor_ROS(
                prim_path=robot_prim_path + "/Baro",
                water_surface_z=self._water_surface,
                frame_id="baro",
            )

    def _setup_scenario(self):
        self._reset_scenario()
        self._add_extra_ui()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(
            self._imu,
            self._rob,
            self._sonar,
            self._cam,
            self._DVL,
            self._baro,
            self._ctrl_mode,
        )

    def _on_post_reset_btn(self):
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

    def _on_imu_checkbox_click_fn(self, model):
        self._use_imu = model
        print("Reload the scene for changes to take effect.")

    def _on_sonar_checkbox_click_fn(self, model):
        self._use_sonar = model
        print("Reload the scene for changes to take effect.")

    def _on_camera_checkbox_click_fn(self, model):
        self._use_camera = model
        print("Reload the scene for changes to take effect.")

    def _on_DVL_checkbox_click_fn(self, model):
        self._use_DVL = model
        print("Reload the scene for changes to take effect.")

    def _on_baro_checkbox_click_fn(self, model):
        self._use_baro = model
        print("Reload the scene for changes to take effect.")

    def _on_manual_ctrl_cb_click_fn(self, model):
        self._manual_ctrl = model
        print("Reload the scene for changes to take effect.")

    def _on_ctrl_mode_dropdown_clicked(self, model):
        self._ctrl_mode = model
        print(f"Ctrl mode: {model}. Reload the scene for changes to take effect.")

    def _add_extra_ui(self):
        with self.sensor_reading_frame:
            with ui.VStack(spacing=5, height=0):
                if self._use_DVL is True:
                    self._build_DVL_plot()
                    self.sensor_reading_frame.visible = True
                if self._use_baro is True:
                    self._build_baro_plot()
                    self.sensor_reading_frame.visible = True
                if not self._use_baro and not self._use_DVL:
                    self.sensor_reading_frame.visible = False
        with self.waypoints_frame:
            if self._ctrl_mode == "Waypoints":
                self._build_waypoints_filepicker()
                self.waypoints_frame.visible = True
            else:
                self.waypoints_frame.visible = False

    def _build_waypoints_filepicker(self):
        self._waypoints_path_field = str_builder(
            label="Path to waypoints",
            default_val=self._waypoints_path,
            tooltip="Select the txt files containing the waypoint data",
            use_folder_picker=True,
            folder_button_title="Select txt",
            folder_dialog_title="Select the txt file containing the waypoint",
        )
        self._scenario.setup_waypoints(
            waypoint_path=self._waypoints_path,
            default_waypoint_path=self._extension_path + "/demo/demo_waypoints.txt",
        )
        self._waypoints_path_field.add_value_changed_fn(
            self._on_waypoints_path_changed_fn
        )

    def _on_waypoints_path_changed_fn(self, model):
        self._waypoints_path = model.get_value_as_string()
        self._scenario.setup_waypoints(
            waypoint_path=model.get_value_as_string(),
            default_waypoint_path=self._extension_path + "/demo/demo_waypoints.txt",
        )

    def _build_DVL_plot(self):
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
        print("DVL DAQ: ", val)
        if val:
            if not self._DVL_event_sub:
                self._DVL_event_sub = (
                    omni.kit.app.get_app()
                    .get_update_event_stream()
                    .create_subscription_to_pop(self._on_DVL_step)
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

    def _build_baro_plot(self):
        self._baro_event_sub = None
        self._baro_data = []

        kwargs = {
            "label": "Barometer reading (Pa)",
            "on_clicked_fn": self.toggle_baro_step,
            "data": self._baro_data,
            "min": 101325.0,
            "max": 101325.0 + 50000,
        }
        self._baro_plot, self._baro_plot_value = combo_cb_plot_builder(**kwargs)

    def toggle_baro_step(self, val=None):
        print("Barometer DAQ: ", val)
        if val:
            if not self._baro_event_sub:
                self._baro_event_sub = (
                    omni.kit.app.get_app()
                    .get_update_event_stream()
                    .create_subscription_to_pop(self._on_baro_step)
                )
            else:
                self._baro_event_sub = None
        else:
            self._baro_event_sub = None

    def _on_baro_step(self, e: carb.events.IEvent):
        baro = float(self._scenario._baro_reading)
        self._baro_plot_value.set_value(baro)
        self._baro_data.append(baro)
        if len(self._baro_data) > 50:
            self._baro_data.pop(0)
        self._baro_plot.set_data(*self._baro_data)
