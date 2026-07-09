import omni.graph.core as og
import omni.timeline
from isaacsim.sensors.physics import IMUSensor


class ImuSensor_ROS(IMUSensor):
    """OceanSim IMU wrapper that publishes IMU data via OmniGraph ROS2PublishImu."""

    def __init__(
        self,
        prim_path,
        name="Imu",
        frequency=None,
        translation=None,
        og_node=None,
    ):
        self._og_node = og_node
        super().__init__(
            prim_path=prim_path,
            name=name,
            frequency=frequency,
            translation=translation,
        )

    def initialize(self, physics_sim_view=None, og_node=None):
        if og_node is not None:
            self._og_node = og_node
        super().initialize(physics_sim_view)

    def read(self):
        # imu api: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.sensors.physics/docs/index.html#isaacsim.sensors.physics.IMUSensor
        # graph node attributes: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2PublishImu.html
        imu_data = self.get_current_frame()
        if self._og_node is None:
            return imu_data

        sim_time = float(omni.timeline.get_timeline_interface().get_current_time())
        if self._og_node.get_attribute_exists("inputs:timeStamp"):
            og.Controller.attribute(
                self._og_node.get_attribute("inputs:timeStamp")
            ).set(sim_time)
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:angularVelocity")
        ).set(imu_data["ang_vel"])
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:linearAcceleration")
        ).set(imu_data["lin_acc"])
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:orientation")
        ).set(imu_data["orientation"])
        return imu_data
