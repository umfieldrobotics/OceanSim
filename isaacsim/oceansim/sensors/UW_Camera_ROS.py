import numpy as np
import omni.graph.core as og
import omni.timeline

from isaacsim.oceansim.sensors.UW_Camera import UW_Camera


class UW_Camera_ROS(UW_Camera):
    """Underwater camera wrapper that publishes RGB, depth, and pointcloud via OmniGraph."""

    def __init__(
        self,
        prim_path,
        name="UW_Camera_ROS",
        frequency=None,
        dt=None,
        resolution=None,
        position=None,
        orientation=None,
        translation=None,
        render_product_path=None,
        og_node=None,
        depth_og_node=None,
        pointcloud_og_node=None,
    ):
        super().__init__(
            prim_path=prim_path,
            name=name,
            frequency=frequency,
            dt=dt,
            resolution=resolution,
            position=position,
            orientation=orientation,
            translation=translation,
            render_product_path=render_product_path,
        )
        self._og_node = og_node
        self._depth_og_node = depth_og_node
        self._pointcloud_og_node = pointcloud_og_node

    def initialize(
        self,
        UW_param: np.ndarray = np.array(
            [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05]
        ),
        depth_noise_sigma: float = 0.01,
        max_range: float = 20.0,
        viewport: bool = True,
        writing_dir: str = None,
        UW_yaml_path: str = None,
        physics_sim_view=None,
        og_node=None,
        depth_og_node=None,
        pointcloud_og_node=None,
        depth_topic_name=None,
        pointcloud_topic_name=None,
    ):
        if og_node is not None:
            self._og_node = og_node
        if depth_og_node is not None:
            self._depth_og_node = depth_og_node
        if pointcloud_og_node is not None:
            self._pointcloud_og_node = pointcloud_og_node

        super().initialize(
            UW_param=UW_param,
            depth_noise_sigma=depth_noise_sigma,
            max_range=max_range,
            viewport=viewport,
            writing_dir=writing_dir,
            UW_yaml_path=UW_yaml_path,
            physics_sim_view=physics_sim_view,
        )

        if self._depth_og_node:
            frame_id = self.prim_path.split("/")[-1]
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:frameId")
            ).set(frame_id)
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:topicName")
            ).set(
                depth_topic_name
                if depth_topic_name is not None
                else "DepthImage"
            )
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:encoding")
            ).set("32FC1")

        if self._pointcloud_og_node:
            from isaacsim.ros2.bridge import read_camera_info

            frame_id = self.prim_path.split("/")[-1]
            camera_info, _ = read_camera_info(
                render_product_path=self._render_product_path
            )
            self._camera_k = np.asarray(camera_info.k, dtype=np.float32).reshape(3, 3)
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:frameId")
            ).set(frame_id)
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:topicName")
            ).set(
                pointcloud_topic_name
                if pointcloud_topic_name is not None
                else "RGBCamera/pointcloud"
            )

    def render(self):
        super().render()
        self._publish_ros()

    def _publish_ros(self):
        if self._uw_frame is None:
            return

        sim_time = omni.timeline.get_timeline_interface().get_current_time()

        if self._og_node:
            width = self._res[0] if self._res is not None else self._uw_frame.shape[1]
            height = self._res[1] if self._res is not None else self._uw_frame.shape[0]
            og.Controller.attribute(self._og_node.get_attribute("inputs:width")).set(
                width
            )
            og.Controller.attribute(self._og_node.get_attribute("inputs:height")).set(
                height
            )
            og.Controller.attribute(
                self._og_node.get_attribute("inputs:timeStamp")
            ).set(sim_time)
            og.Controller.attribute(self._og_node.get_attribute("inputs:data")).set(
                self._uw_frame
            )

        if self._depth_og_node and self._degraded_depth_frame is not None:
            depth_bytes = self._degraded_depth_frame.view(np.uint8).reshape(-1)
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:timeStamp")
            ).set(sim_time)
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:width")
            ).set(self._degraded_depth_frame.shape[1])
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:height")
            ).set(self._degraded_depth_frame.shape[0])
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:bufferSize")
            ).set(self._degraded_depth_frame.nbytes)
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:data")
            ).set(depth_bytes)

        if self._pointcloud_og_node and self._camera_k is not None:
            pointcloud = self._build_pointcloud()
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:timeStamp")
            ).set(sim_time)
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:bufferSize")
            ).set(pointcloud.nbytes)
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:data")
            ).set(pointcloud)
