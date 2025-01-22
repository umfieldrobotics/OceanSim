# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import weakref

import omni
import omni.ui as ui
from omni.isaac.core.utils.prims import delete_prim, get_prim_at_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.ui.menu import make_menu_item_description
from omni.isaac.ui.ui_utils import btn_builder, combo_cb_scrolling_frame_builder, get_style, setup_ui_headers
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

EXTENSION_NAME = "Ultrasonic Info"


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        self._ext_id = ext_id

        # The extension acquires the ULTRASONIC interface at startup.  It will be released during extension shutdown.  We
        # create a ULTRASONIC prim using our schema, and then we interact with / query that prim using the python API found
        # in ultrasonic/bindings
        self._ul = _range_sensor.acquire_ultrasonic_sensor_interface()
        self.ultrasonic = None
        self._timeline = omni.timeline.get_timeline_interface()

        self._menu_items = [
            MenuItemDescription(
                name="Sensors",
                sub_menu=[
                    make_menu_item_description(ext_id, "Ultrasonic", lambda a=weakref.proxy(self): a._menu_callback())
                ],
            )
        ]
        add_menu_items(self._menu_items, "Isaac Examples")

        self._build_ui()

    def _build_ui(self):
        self._window = omni.ui.Window(
            EXTENSION_NAME, width=600, height=800, visible=False, dockPreference=omni.ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                title = "Read an Ultrasonic Sensor Data Stream"
                doc_link = "https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/ext_omni_isaac_range_sensor.html"

                overview = "This sample demonstrates the ULTRASONIC python API for Isaac Sim. It shows how to create an Ultrasonic Sensor, set its properties, and read data streaming from it. "
                overview += "First press the 'Load Sensor' button and then press PLAY to simulate."
                overview += "\n\nPress the 'Open in IDE' button to view the source code."
                overview += "\nNote: The buttons above only work with an Ultrasonic sensor made by the 'Load Sensor' button; not existing ones in the stage."

                setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                frame = ui.CollapsableFrame(
                    title="Command Panel",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        dict = {
                            "label": "Load Sensor",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Loads an Ultrasonic Sensor and sets its properties",
                            "on_clicked_fn": self._on_spawn_ultrasonic_button,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Load Scene",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Loads a obstacles for the Ultrasonic sensor to sense",
                            "on_clicked_fn": self._on_spawn_obstacles_button,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Show Data Stream",
                            "type": "checkbox_scrolling_frame",
                            "default_val": [False, "No Data To Display"],
                            "tooltip": "Show incoming data from an active Sensor",
                        }
                        self._info_cb, self._info_label = combo_cb_scrolling_frame_builder(**dict)

                self._envelope_frame = ui.CollapsableFrame(
                    title="Envelopes Panel",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )

    def on_shutdown(self):
        # Perform cleanup once the sample closes
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None
        self._editor_event_subscription = None

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    async def _spawn_ultrasonic_function(self, task):
        # Wait for stage clear to complete before creating ULTRASONIC
        done, pending = await asyncio.wait({task})
        if task in done:
            stage = omni.usd.get_context().get_stage()

            # Set up axis to z.  The ULTRASONIC extension scans in the XZ plane, which is assumed to be perpendicular to the
            # rotational plane, and so to use the ULTRASONIC as it is currently written, Z must be up.
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)

            # Create the PhysicsScene.  The ultrasonic is going to execute line trace calls in PhysX, and return a value based
            # on how far it travels before colliding with an object that is using the PhysX collision API.  Because of this,
            # to use the ULTRASONIC extension, you MUST have a physics scene defined
            UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))

            # List of poses that define the emitter prims
            origin = Gf.Vec3d(4.8, 6.4, 0.0)

            emitter_poses = [
                ((0, 0, 75.0), Gf.Vec3d(3.844, 0.9384, 0.525)),
                ((0, 0, 30.0), Gf.Vec3d(4.046, 0.7735, 0.56)),
                ((0, 0, 11.8), Gf.Vec3d(4.172, 0.3256, 0.591)),
                ((0, 0, -11.8), Gf.Vec3d(4.172, -0.3256, 0.591)),
                ((0, 0, -30.0), Gf.Vec3d(4.046, -0.7735, 0.561)),
                ((0, 0, -75.0), Gf.Vec3d(3.844, -0.9384, 0.525)),
                ((0, 0, 99.2), Gf.Vec3d(-1.454, 0.9352, 0.5367)),
                ((0, 0, 150.0), Gf.Vec3d(-1.789, 0.788, 0.558)),
                ((0, 0, 175.5), Gf.Vec3d(-1.887, 0.36, 0.6249)),
                ((0, 0, -175.5), Gf.Vec3d(-1.887, -0.36, 0.6249)),
                ((0, 0, -150.0), Gf.Vec3d(-1.789, -0.788, 0.558)),
                ((0, 0, -99.2), Gf.Vec3d(-1.454, -0.9352, 0.5367)),
            ]

            adjacency = [
                [0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5],
                [6, 7],
                [6, 7, 8],
                [7, 8, 9],
                [8, 9, 10],
                [9, 10, 11],
                [10, 11],
            ]
            emitters = []
            for i in range(len(emitter_poses)):
                pose = emitter_poses[i]
                adjacent = adjacency[i]
                result, emitter_prim = omni.kit.commands.execute(
                    "RangeSensorCreateUltrasonicEmitter",
                    path="/World/UltrasonicEmitter",
                    per_ray_intensity=0.4,
                    yaw_offset=0.0,
                    adjacency_list=adjacent,
                )
                emitter_prim.GetPrim().GetAttribute("xformOp:translate").Set((origin + pose[1]))
                emitter_prim.GetPrim().GetAttribute("xformOp:rotateXYZ").Set(pose[0])
                emitters.append(emitter_prim)
            emitter_paths = [emitter.GetPath() for emitter in emitters]

            result, group_1 = omni.kit.commands.execute(
                "RangeSensorCreateUltrasonicFiringGroup",
                path="/World/UltrasonicFiringGroup_0",
                emitter_modes=[(0, 1), (3, 0), (4, 1), (7, 0), (8, 1), (11, 0)],
                receiver_modes=[
                    (0, 1),
                    (1, 1),
                    (2, 0),
                    (3, 0),
                    (3, 1),
                    (4, 0),
                    (4, 1),
                    (5, 1),
                    (6, 0),
                    (7, 0),
                    (7, 1),
                    (8, 0),
                    (8, 1),
                    (9, 1),
                    (10, 0),
                    (11, 0),
                ],
            )

            result, group_2 = omni.kit.commands.execute(
                "RangeSensorCreateUltrasonicFiringGroup",
                path="/World/UltrasonicFiringGroup_1",
                emitter_modes=[(1, 1), (2, 0), (5, 1), (6, 0), (9, 1), (10, 0)],
                receiver_modes=[
                    (0, 1),
                    (1, 0),
                    (1, 1),
                    (2, 0),
                    (2, 1),
                    (3, 0),
                    (4, 1),
                    (5, 1),
                    (6, 0),
                    (7, 0),
                    (8, 1),
                    (9, 0),
                    (9, 1),
                    (10, 0),
                    (10, 1),
                    (11, 0),
                ],
            )
            self.ultrasonicPath = "/World/UltrasonicArray"

            result, self.ultrasonic = omni.kit.commands.execute(
                "RangeSensorCreateUltrasonicArray",
                path=self.ultrasonicPath,
                # Min and max range for the ULTRASONIC.  This defines the starting and stopping locations for the linetrace
                min_range=0.4,
                max_range=4.5,
                # These attributes affect drawing the ultrasonic in the viewport.  High Level Of Detail (HighLod) = True will draw
                # all rays.  If false it will only draw horizontal rays.  Draw Ultrasonic Points = True will draw the actual
                # ULTRASONIC rays in the viewport.
                draw_points=False,
                draw_lines=True,
                # Horizontal and vertical resolution in degrees.  Rays will be fired on the bin boundries defined by the
                # resolution.  If your FOV is 45 degrees and your resolution is 15 degrees, you will get rays at
                # 0, 15, 30, and 45 degrees.
                horizontal_fov=90.0,  # set wedge vertical extent in degrees
                vertical_fov=15.0,  # set wedge horizontal extent in degrees
                horizontal_resolution=0.3,
                vertical_resolution=0.5,
                num_bins=224,
                emitter_prims=emitter_paths,
                firing_group_prims=[group_1.GetPath(), group_2.GetPath()],
            )

            # we want to make sure we can see the ultrasonic we made, so we set the camera position and look target
            set_camera_view(eye=[20.00, 10.00, 5.00], target=[5.00, 5.00, 0.0], camera_prim_path="/OmniverseKit_Persp")

    def _on_spawn_ultrasonic_button(self):
        # wait for new stage before creating ultrasonic
        task = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        asyncio.ensure_future(self._spawn_ultrasonic_function(task))
        # refresh data stream box
        self._info_label.text = ""

        self._editor_event_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_editor_step)
        )

    def _on_editor_step(self, step):
        if self._info_cb.get_value_as_bool():
            if self._timeline.is_playing():
                self._get_info_function()
        else:
            self._info_label.text = ""

    def _on_spawn_obstacles_button(self):
        stage = omni.usd.get_context().get_stage()
        self.CubePath = "/World/Cube"
        self.CylinderPath = "/World/Cylinder"
        offset = Gf.Vec3f(-0.4636036, 7.2820291, 0.618376)
        offset_cylinder = Gf.Vec3f(3.8492474, 2.0546415, 0.6868243)
        size = 1.00
        cylinder_height = 2.00
        radius = 0.10

        # Define a light so we can see the obstacle better
        if get_prim_at_path("/DistantLight"):
            delete_prim("/DistantLight")
        distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(500)
        distantLight.AddRotateXYZOp().Set((-36, 36, 0))

        # To create a cube, we first define our geometry at our chosen path.  Then, becuase
        # we will need the primitive later, we query the prim from the stage. If the prim already exists, skip creation
        if stage.GetPrimAtPath(self.CubePath):
            return

        cylinderGeom = UsdGeom.Cylinder.Define(stage, self.CylinderPath)
        cubeGeom = UsdGeom.Cube.Define(stage, self.CubePath)
        cubePrim = stage.GetPrimAtPath(self.CubePath)
        cylinderPrim = stage.GetPrimAtPath(self.CylinderPath)

        # Remember!  Attributes do not exist until they are created.  Here we set the value to the non defualt at
        # creation.  Note that moving the cube to a different location involves adding a translation operation to
        # our primitive.
        cubeGeom.CreateSizeAttr(size)
        cylinderGeom.CreateHeightAttr(cylinder_height)
        cylinderGeom.CreateRadiusAttr(radius)
        cubeGeom.AddTranslateOp().Set(offset)
        cylinderGeom.AddTranslateOp().Set(offset_cylinder)

        # In order for our cube to interact with the ULTRASONIC, it needs to be able to colide with our physX line traces.
        # to do this, we give our cube the collision API, and set it's material and collision group.
        UsdPhysics.CollisionAPI.Apply(cubePrim)
        UsdPhysics.CollisionAPI.Apply(cylinderPrim)

    def _draw_envelope_frame(self):
        envelope_arr = self._ul.get_envelope_array(self.ultrasonicPath)

        with self._envelope_frame:
            with ui.VStack():
                ui.Label("Inspect Envelopes:", height=0)
                ui.Label("Mouse over the plot to see the associated envelope values per bin", height=0)
                for i in range(envelope_arr.shape[0]):
                    with ui.HStack():
                        ui.Label(f"{i}", width=15)
                        ui.Spacer(width=5)
                        ui.Plot(
                            ui.Type.HISTOGRAM,
                            0.0,
                            600.0,
                            *(envelope_arr[i].tolist()),
                            height=50,
                            style={"color": 0xFFFFFFFF},
                        )
                    ui.Spacer(height=1)

    def _get_info_function(self, val=False):
        if not self.ultrasonic:
            return
        maxDepth = self.ultrasonic.GetMaxRangeAttr().Get()
        self._info_label.text = ""

        # The ULTRASONIC itself exists as a C++ object.  In order to retrieve data from this object we need to call
        # C++ code, but this is handled for us through the use of python bindings.  Here we get the depth value of
        # each ray, and the spherical coordinates of each ray in (azimuth, zenith).
        depth = self._ul.get_depth_data(self.ultrasonicPath, 5)
        zenith = self._ul.get_zenith_data(self.ultrasonicPath)
        azimuth = self._ul.get_azimuth_data(self.ultrasonicPath)

        self._draw_envelope_frame()

        # most of the below is string formatting in order to display our data in a nice table within our GUI.
        tableString = ""
        numCols = len(zenith)
        rowString = ""
        for i in range(numCols):
            rowString += "{" + str(i + 2) + ":." + str(5) + "f}   "
        rowString = "{0:16}  {1:10}" + rowString + "\n"

        tableString += rowString.format("Azimuth \ Zenith", " | ", *zenith)
        tableString += "-" * len(tableString) + "\n"
        for row, cols in enumerate(depth):
            # The data on the c++ side is stored as uint16.  in order to get our depth values into centimeters, we
            # must first convert from uint16 into float on [0,1], and then scale to the maximum distance.
            entry = [ray * maxDepth / 65535.0 for ray in cols]
            tableString += rowString.format("{0:.5f}".format(azimuth[row]), " | ", *entry)

        self._info_label.text = tableString

    def _set_ultrasonicarray_path(self):
        def get_selected_path():
            selectedPrims = omni.usd.get_context().get_selection().get_selected_prim_paths()

            if len(selectedPrims) > 0:
                curr_prim = selectedPrims[-1]
            else:
                curr_prim = None
            return curr_prim

        self.ultrasonicPath = get_selected_path()
