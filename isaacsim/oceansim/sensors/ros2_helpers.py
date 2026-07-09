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
def publish_camera_info(camera: Camera, freq):
    from isaacsim.ros2.bridge import read_camera_info

    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_camera_info"
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


def publish_pointcloud_from_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_pointcloud"  # Set topic name to the camera's name
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


def publish_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_depth"
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
def publish_rgb(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_rgb"
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
    ):
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
        self._setup_ros_graph()

    def _setup_ros_graph(self):
        """Creates a standalone OmniGraph to drive the internal C++ ROS Bridge."""
        try:
            keys = og.Controller.Keys
            graph_path = f"/UW_Publisher_{self._name}"

            if pd := omni.usd.get_context().get_stage().GetPrimAtPath(graph_path):
                omni.kit.commands.execute("DeletePrims", paths=[graph_path])
            # https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2PublishImage.html
            (
                self._og_graph,
                [
                    rgb_pub_node,
                    depth_pub_node,
                    pointcloud_pub_node,
                    sonar_pub_node,
                    imu_pub_node,
                    dvl_pub_node,
                    baro_pub_node,
                    _,
                ],
                _,
                _,
            ) = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("uw_rgb_publisher", "isaacsim.ros2.bridge.ROS2PublishImage"),
                        ("uw_depth_publisher", "isaacsim.ros2.bridge.ROS2PublishImage"),
                        (
                            "uw_pointcloud_publisher",
                            "isaacsim.ros2.bridge.ROS2PublishPointCloud",
                        ),
                        (
                            "multibeam_sonar_publisher",
                            "isaacsim.ros2.bridge.ROS2PublishImage",
                        ),
                        ("imu_publisher", "isaacsim.ros2.bridge.ROS2PublishImu"),
                        ("dvl_publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
                        ("baro_publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
                        ("on_tick", "omni.graph.action.OnTick"),
                    ],
                    keys.CONNECT: [
                        ("on_tick.outputs:tick", "uw_rgb_publisher.inputs:execIn"),
                        ("on_tick.outputs:tick", "uw_depth_publisher.inputs:execIn"),
                        (
                            "on_tick.outputs:tick",
                            "uw_pointcloud_publisher.inputs:execIn",
                        ),
                        (
                            "on_tick.outputs:tick",
                            "multibeam_sonar_publisher.inputs:execIn",
                        ),
                        ("on_tick.outputs:tick", "imu_publisher.inputs:execIn"),
                        ("on_tick.outputs:tick", "dvl_publisher.inputs:execIn"),
                        ("on_tick.outputs:tick", "baro_publisher.inputs:execIn"),
                    ],
                    keys.SET_VALUES: [
                        ("uw_rgb_publisher.inputs:topicName", f"{self._name}/rgb"),
                        ("uw_rgb_publisher.inputs:frameId", self._name),
                        (
                            "uw_rgb_publisher.inputs:encoding",
                            "rgba8",
                        ),  # switch back to rgb8 if not working
                        ("uw_depth_publisher.inputs:topicName", f"{self._name}/depth"),
                        ("uw_depth_publisher.inputs:frameId", self._name),
                        ("uw_depth_publisher.inputs:encoding", "32FC1"),
                        (
                            "uw_pointcloud_publisher.inputs:topicName",
                            f"{self._name}/pointcloud",
                        ),
                        ("uw_pointcloud_publisher.inputs:frameId", self._name),
                        (
                            "multibeam_sonar_publisher.inputs:topicName",
                            f"{self._name}/sonar_image",
                        ),
                        ("multibeam_sonar_publisher.inputs:frameId", self._name),
                        (
                            "multibeam_sonar_publisher.inputs:encoding",
                            "rgba8",
                        ),  # switch back to rgb8 if not working
                        ("imu_publisher.inputs:topicName", f"{self._name}/imu"),
                        ("imu_publisher.inputs:frameId", self._name),
                        ("dvl_publisher.inputs:topicName", f"{self._name}/dvl"),
                        ("dvl_publisher.inputs:queueSize", 10),
                        (
                            "dvl_publisher.inputs:messagePackage",
                            self._dvl_message_package,
                        ),
                        (
                            "dvl_publisher.inputs:messageSubfolder",
                            self._dvl_message_subfolder,
                        ),
                        ("dvl_publisher.inputs:messageName", self._dvl_message_name),
                        ("baro_publisher.inputs:topicName", f"{self._name}/baro"),
                        ("baro_publisher.inputs:queueSize", 10),
                        ("baro_publisher.inputs:messagePackage", "sensor_msgs"),
                        ("baro_publisher.inputs:messageSubfolder", "msg"),
                        ("baro_publisher.inputs:messageName", "FluidPressure"),
                    ],
                },
            )
            self._rgb_node = rgb_pub_node
            self._depth_node = depth_pub_node
            self._pointcloud_node = pointcloud_pub_node
            self._sonar_node = sonar_pub_node
            self._imu_node = imu_pub_node
            self._dvl_node = dvl_pub_node
            self._baro_node = baro_pub_node
            carb.log_info(
                f"[{self._name}] Internal ROS 2 Bridge Graph initialized at {graph_path}"
            )

        except Exception as e:
            carb.log_error(f"[{self._name}] Failed to setup ROS Graph: {e}")
