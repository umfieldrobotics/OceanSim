import carb
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from isaacsim.core.nodes.scripts.utils import set_target_prims
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.sensors.camera import Camera

enable_extension("isaacsim.ros2.bridge")


def to_ros_stamp(sim_time: float) -> tuple[int, int]:
    """Convert a simulation time (seconds) to a ROS 2 stamp (sec, nanosec)."""
    sec = int(sim_time)
    nanosec = int(round((sim_time - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    elif nanosec < 0:
        sec -= 1
        nanosec += 1_000_000_000
    return sec, nanosec


###### Camera helper functions for setting up publishers. ########


# Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera_publishing.html
def publish_camera_info(camera: Camera, freq, topic_name=None):
    from isaacsim.ros2.bridge import read_camera_info

    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    if topic_name is None:
        topic_name = "ImagingSonar/camera_info"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[
        -1
    ]  # This matches what the TF tree is publishing.
    writer = rep.writers.get("ROS2PublishCameraInfo")
    camera_info, _ = read_camera_info(render_product_path=render_product)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_pointcloud_from_depth(camera: Camera, freq, topic_name=None):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    if topic_name is None:
        topic_name = "ImagingSonar/pointcloud"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[
        -1
    ]  # This matches what the TF tree is publishing.

    # Note, this pointcloud publisher will convert the Depth image to a pointcloud using the Camera intrinsics.
    # This pointcloud generation method does not support semantic labeled objects.
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_depth(camera: Camera, freq, topic_name=None):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    if topic_name is None:
        topic_name = "ImagingSonar/depth_raw"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[
        -1
    ]  # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


# Not currently using this for the UW camera, see the omnigraph node in UW_Camera_ROS instead
def publish_rgb(camera: Camera, freq, topic_name=None):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    if topic_name is None:
        topic_name = "RGBCamera/image_raw"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[
        -1
    ]  # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.Rgb.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_camera_tf(camera: Camera):
    camera_prim = camera.prim_path

    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    try:
        # Generate the camera_frame_id. OmniActionGraph will use the last part of
        # the full camera prim path as the frame name, so we will extract it here
        # and use it for the pointcloud frame_id.
        camera_frame_id = camera_prim.split("/")[-1]

        # Generate an action graph associated with camera TF publishing.
        ros_camera_graph_path = "/CameraTFActionGraph"

        # If a camera graph is not found, create a new one.
        if not is_prim_path_valid(ros_camera_graph_path):
            (ros_camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": ros_camera_graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("IsaacClock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("RosPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                        (
                            "IsaacClock.outputs:simulationTime",
                            "RosPublisher.inputs:timeStamp",
                        ),
                    ],
                },
            )

        # Generate 2 nodes associated with each camera: TF from world to ROS camera convention, and world frame.
        og.Controller.edit(
            ros_camera_graph_path,
            {
                og.Controller.Keys.CREATE_NODES: [
                    (
                        "PublishTF_" + camera_frame_id,
                        "isaacsim.ros2.bridge.ROS2PublishTransformTree",
                    ),
                    (
                        "PublishRawTF_" + camera_frame_id + "_world",
                        "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF_" + camera_frame_id + ".inputs:topicName", "/tf"),
                    # Note if topic_name is changed to something else besides "/tf",
                    # it will not be captured by the ROS tf broadcaster.
                    (
                        "PublishRawTF_" + camera_frame_id + "_world.inputs:topicName",
                        "/tf",
                    ),
                    (
                        "PublishRawTF_"
                        + camera_frame_id
                        + "_world.inputs:parentFrameId",
                        camera_frame_id,
                    ),
                    (
                        "PublishRawTF_"
                        + camera_frame_id
                        + "_world.inputs:childFrameId",
                        camera_frame_id + "_world",
                    ),
                    # Static transform from ROS camera convention to world (+Z up, +X forward) convention:
                    (
                        "PublishRawTF_" + camera_frame_id + "_world.inputs:rotation",
                        [0.5, -0.5, 0.5, 0.5],
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        ros_camera_graph_path + "/OnTick.outputs:tick",
                        "PublishTF_" + camera_frame_id + ".inputs:execIn",
                    ),
                    (
                        ros_camera_graph_path + "/OnTick.outputs:tick",
                        "PublishRawTF_" + camera_frame_id + "_world.inputs:execIn",
                    ),
                    (
                        ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                        "PublishTF_" + camera_frame_id + ".inputs:timeStamp",
                    ),
                    (
                        ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                        "PublishRawTF_" + camera_frame_id + "_world.inputs:timeStamp",
                    ),
                ],
            },
        )
    except Exception as e:
        carb.log_error(f"Failed to setup camera TF publishers: {e}")

    # Add target prims for the USD pose. All other frames are static.
    set_target_prims(
        primPath=ros_camera_graph_path + "/PublishTF_" + camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )
    return


###################################################################


# Omnigraph setup
class OmniHandler:
    def __init__(
        self,
        name: str = "Oceansim",
        dvl_message_package: str = "msgs",
        dvl_message_subfolder: str = "msg",
        dvl_message_name: str = "Dvl",
        use_camera: bool = False,
        use_sonar: bool = False,
        use_imu: bool = False,
        use_dvl: bool = False,
        use_baro: bool = False,
    ):
        self._og_graph = None
        self._rgb_node = None
        self._depth_node = None
        self._pointcloud_node = None
        self._sonar_node = None
        self._imu_node = None
        self._dvl_node = None
        self._baro_node = None
        self._name = name
        self._dvl_message_package = dvl_message_package
        self._dvl_message_subfolder = dvl_message_subfolder
        self._dvl_message_name = dvl_message_name
        self._use_camera = use_camera
        self._use_sonar = use_sonar
        self._use_imu = use_imu
        self._use_dvl = use_dvl
        self._use_baro = use_baro
        self._setup_ros_graph()

    @staticmethod
    def _add_sensor_publisher(
        *,
        node_definitions,
        execution_connections,
        attribute_values,
        publisher_node_name,
        publisher_node_type,
        publisher_input_values,
    ):
        node_definitions.append((publisher_node_name, publisher_node_type))
        execution_connections.append(
            (
                "on_tick.outputs:tick",
                f"{publisher_node_name}.inputs:execIn",
            )
        )
        attribute_values.extend(
            (
                f"{publisher_node_name}.inputs:{input_name}",
                input_value,
            )
            for input_name, input_value in publisher_input_values.items()
        )

    def _setup_ros_graph(self):
        """Creates a standalone OmniGraph to drive the internal C++ ROS Bridge."""
        try:
            keys = og.Controller.Keys
            graph_path = f"/UW_Publisher_{self._name}"

            if omni.usd.get_context().get_stage().GetPrimAtPath(graph_path):
                omni.kit.commands.execute("DeletePrims", paths=[graph_path])

            publisher_node_definitions = [
                ("on_tick", "omni.graph.action.OnTick")
            ]
            publisher_execution_connections = []
            publisher_attribute_values = []

            # https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2PublishImage.html
            if self._use_camera:
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="uw_rgb_publisher",
                    publisher_node_type="isaacsim.ros2.bridge.ROS2PublishImage",
                    publisher_input_values={
                        "topicName": "RGBCamera/image",
                        "frameId": self._name,
                        "encoding": "rgba8",
                    },
                )
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="uw_depth_publisher",
                    publisher_node_type="isaacsim.ros2.bridge.ROS2PublishImage",
                    publisher_input_values={
                        "topicName": "DepthImage",
                        "frameId": self._name,
                        "encoding": "32FC1",
                    },
                )
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="uw_pointcloud_publisher",
                    publisher_node_type=(
                        "isaacsim.ros2.bridge.ROS2PublishPointCloud"
                    ),
                    publisher_input_values={
                        "topicName": "RGBCamera/pointcloud",
                        "frameId": self._name,
                    },
                )

            if self._use_sonar:
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="multibeam_sonar_publisher",
                    publisher_node_type="isaacsim.ros2.bridge.ROS2PublishImage",
                    publisher_input_values={
                        "topicName": "ImagingSonar/image",
                        "frameId": self._name,
                        "encoding": "rgba8",
                    },
                )

            if self._use_imu:
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="imu_publisher",
                    publisher_node_type="isaacsim.ros2.bridge.ROS2PublishImu",
                    publisher_input_values={
                        "topicName": "IMU",
                        "frameId": self._name,
                    },
                )

            if self._use_dvl:
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="dvl_publisher",
                    publisher_node_type="isaacsim.ros2.bridge.ROS2Publisher",
                    publisher_input_values={
                        "topicName": "DVL",
                        "queueSize": 10,
                        "messagePackage": self._dvl_message_package,
                        "messageSubfolder": self._dvl_message_subfolder,
                        "messageName": self._dvl_message_name,
                    },
                )

            if self._use_baro:
                self._add_sensor_publisher(
                    node_definitions=publisher_node_definitions,
                    execution_connections=publisher_execution_connections,
                    attribute_values=publisher_attribute_values,
                    publisher_node_name="baro_publisher",
                    publisher_node_type="isaacsim.ros2.bridge.ROS2Publisher",
                    publisher_input_values={
                        "topicName": "Barometer",
                        "queueSize": 10,
                        "messagePackage": "sensor_msgs",
                        "messageSubfolder": "msg",
                        "messageName": "FluidPressure",
                    },
                )

            if not publisher_execution_connections:
                carb.log_info(
                    f"[{self._name}] No ROS 2 sensor publishers are enabled"
                )
                return

            self._og_graph, _, _, _ = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: publisher_node_definitions,
                    keys.CONNECT: publisher_execution_connections,
                    keys.SET_VALUES: publisher_attribute_values,
                },
            )

            if self._use_camera:
                self._rgb_node = og.Controller.node(
                    f"{graph_path}/uw_rgb_publisher"
                )
                self._depth_node = og.Controller.node(
                    f"{graph_path}/uw_depth_publisher"
                )
                self._pointcloud_node = og.Controller.node(
                    f"{graph_path}/uw_pointcloud_publisher"
                )
            if self._use_sonar:
                self._sonar_node = og.Controller.node(
                    f"{graph_path}/multibeam_sonar_publisher"
                )
            if self._use_imu:
                self._imu_node = og.Controller.node(
                    f"{graph_path}/imu_publisher"
                )
            if self._use_dvl:
                self._dvl_node = og.Controller.node(
                    f"{graph_path}/dvl_publisher"
                )
            if self._use_baro:
                self._baro_node = og.Controller.node(
                    f"{graph_path}/baro_publisher"
                )
            carb.log_info(
                f"[{self._name}] Internal ROS 2 Bridge Graph initialized at {graph_path}"
            )

        except Exception as e:
            carb.log_error(f"[{self._name}] Failed to setup ROS Graph: {e}")
