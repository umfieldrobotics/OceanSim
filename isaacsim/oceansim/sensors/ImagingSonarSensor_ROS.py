import omni.graph.core as og
import omni.timeline

from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor


class ImagingSonarSensor_ROS(ImagingSonarSensor):
    """Imaging sonar wrapper that publishes sonar images via OmniGraph ROS2PublishImage."""

    def __init__(
        self,
        prim_path,
        name="ImagingSonar_ROS",
        frequency=None,
        dt=None,
        position=None,
        orientation=None,
        translation=None,
        render_product_path=None,
        physics_sim_view=None,
        min_range: float = 0.2,
        max_range: float = 3.0,
        range_res: float = 0.008,
        hori_fov: float = 130.0,
        vert_fov: float = 20.0,
        angular_res: float = 0.5,
        hori_res: int = 3000,
        og_node=None,
    ):
        super().__init__(
            prim_path=prim_path,
            name=name,
            frequency=frequency,
            dt=dt,
            position=position,
            orientation=orientation,
            translation=translation,
            render_product_path=render_product_path,
            physics_sim_view=physics_sim_view,
            min_range=min_range,
            max_range=max_range,
            range_res=range_res,
            hori_fov=hori_fov,
            vert_fov=vert_fov,
            angular_res=angular_res,
            hori_res=hori_res,
        )
        self._og_node = og_node

    def sonar_initialize(
        self,
        output_dir: str = None,
        viewport: bool = True,
        include_unlabelled=False,
        if_array_copy: bool = True,
        og_node=None,
    ):
        if og_node is not None:
            self._og_node = og_node
        super().sonar_initialize(
            output_dir=output_dir,
            viewport=viewport,
            include_unlabelled=include_unlabelled,
            if_array_copy=if_array_copy,
        )

    def _on_sonar_frame(self, sonar_image):
        if self._og_node is None:
            return
        sonar_vis_frame = sonar_image.numpy()
        sim_time = omni.timeline.get_timeline_interface().get_current_time()
        og.Controller.attribute(self._og_node.get_attribute("inputs:timeStamp")).set(
            sim_time
        )
        og.Controller.attribute(self._og_node.get_attribute("inputs:width")).set(
            sonar_vis_frame.shape[1]
        )
        og.Controller.attribute(self._og_node.get_attribute("inputs:height")).set(
            sonar_vis_frame.shape[0]
        )
        og.Controller.attribute(self._og_node.get_attribute("inputs:data")).set(
            sonar_vis_frame
        )
