import numpy as np
import omni.graph.core as og
import omni.timeline

from isaacsim.oceansim.sensors.BarometerSensor import BarometerSensor
from isaacsim.oceansim.sensors.ros2_helpers import to_ros_stamp


class BarometerSensor_ROS(BarometerSensor):
    """Barometer wrapper that publishes sensor_msgs/msg/FluidPressure via ROS2Publisher."""

    def __init__(
        self,
        prim_path,
        name="baro",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        water_density: float = 1000.0,
        g: float = 9.81,
        noise_cov: float = 0.0,
        water_surface_z: float = 0.0,
        atmosphere_pressure: float = 101325.0,
        og_node=None,
        frame_id: str = "baro",
    ):
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            water_density=water_density,
            g=g,
            noise_cov=noise_cov,
            water_surface_z=water_surface_z,
            atmosphere_pressure=atmosphere_pressure,
        )

        self._og_node = og_node
        self._frame_id = frame_id
        self._last_payload = None

    def initialize(self, og_node=None):
        if og_node is not None:
            self._og_node = og_node

    def _compute_variance(self) -> float:
        sqrt_cov = np.asarray(self._mvn_press.get_sqrt_cov(), dtype=np.float64)
        if sqrt_cov.shape != (1, 1):
            return 0.0
        variance = float(sqrt_cov[0, 0] ** 2)
        if not np.isfinite(variance) or variance < 0.0:
            return 0.0
        return variance

    def _build_payload(self) -> dict:
        sim_time = float(omni.timeline.get_timeline_interface().get_current_time())
        stamp_sec, stamp_nanosec = to_ros_stamp(sim_time)
        pressure = float(self.get_pressure())
        return {
            "header_stamp_sec": stamp_sec,
            "header_stamp_nanosec": stamp_nanosec,
            "header_frame_id": self._frame_id,
            "timestamp": sim_time,
            "fluid_pressure": pressure,
            "variance": self._compute_variance(),
        }

    def _set_og_attr(self, attr_name: str, value):
        attr = self._og_node.get_attribute(attr_name)
        og.Controller.attribute(attr).set(value)

    def _publish_payload(self, payload: dict):
        if self._og_node is None:
            return
        self._set_og_attr("inputs:header:stamp:sec", payload["header_stamp_sec"])
        self._set_og_attr("inputs:header:stamp:nanosec", payload["header_stamp_nanosec"])
        self._set_og_attr("inputs:header:frame_id", payload["header_frame_id"])
        self._set_og_attr("inputs:fluid_pressure", payload["fluid_pressure"])
        self._set_og_attr("inputs:variance", payload["variance"])

    def read(self):
        payload = self._build_payload()
        self._last_payload = payload
        self._publish_payload(payload)
        return payload["fluid_pressure"]
