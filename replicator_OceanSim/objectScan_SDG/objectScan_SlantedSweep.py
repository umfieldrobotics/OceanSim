import asyncio
import os

import numpy as np
import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
from omni.replicator.core.scripts.functional import write_image, write_json, write_np
import random
WRITE_THREADS = 16
QUEUE_SIZE = 5000

rep.settings.carb_settings("/omni/replicator/backend/writeThreads", WRITE_THREADS)
rep.settings.carb_settings("/omni/replicator/backend/queueSize", QUEUE_SIZE)


## Writer ##
class ScanWriter(Writer):
    def __init__(
        self,
        output_dir,
    ):
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators.append(AnnotatorRegistry.get_annotator("camera_params"))
        self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_camera"))
        self.annotators.append(AnnotatorRegistry.get_annotator("pointcloud"))
        # self.annotators.append(AnnotatorRegistry.get_annotator("rgb")) 


        self._frame_id = 0
    
    
    def write(self, data:dict):
        self._process_frame_data(data)
        self._write_frame_data()
        print(f"Writing frame [{self._frame_id}] data to {self.backend.output_dir} ..")

        self._frame_id += 1
    
    
    
    
    # Process the render product data and store it in the selected format, return the number of objects in the frame
    def _process_frame_data(self, data: dict):
        # Store the frame data for writing to disk
        self._frame_data = {}

        # Get and process the camera parameters annotator data in the selected format

        # Store the camera information in the
        self._frame_data["camera_data"] = self._process_camera_parameters(data["camera_params"])

        self._frame_data["depth_data"] = data["distance_to_camera"]

        self._frame_data['pcl_data'] = data["pointcloud"]["data"]

        # self._frame_data["rgb_data"] = data["rgb"]
    
    
    # Get the camera parameters from the annotator data
    def _process_camera_parameters(self, camera_params) -> dict:
        camera_data = {}
        camera_data["aperture"] = camera_params["cameraAperture"].tolist()
        camera_data["aperture_offset"] = camera_params["cameraApertureOffset"].tolist()
        camera_data["focal_length"] = float(camera_params["cameraFocalLength"])
        camera_data["resolution"] = camera_params["renderProductResolution"].tolist()
        camera_data["meters_per_scene_unit"] = float(camera_params["metersPerSceneUnit"])

        # OV only supports square pixels, so the pixel size is the same in both x and y directions
        # https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#cameras
        pixel_size = camera_params["cameraAperture"][0] / camera_params["renderProductResolution"][0]
        camera_data["intrinsics"] = {
            "fx": camera_params["cameraFocalLength"] / pixel_size,
            "fy": camera_params["cameraFocalLength"] / pixel_size,
            "cx": camera_params["renderProductResolution"][0] / 2.0 + camera_params["cameraApertureOffset"][0],
            "cy": camera_params["renderProductResolution"][1] / 2.0 + camera_params["cameraApertureOffset"][1],
        }
        camera_data["camera_view_matrix"] = np.round(camera_params["cameraViewTransform"], 5).reshape(4, 4).tolist()
        camera_data["camera_projection_matrix"] = np.round(camera_params["cameraProjection"], 5).reshape(4, 4).tolist()

        return camera_data

    # Write the processed data to disk
    def _write_frame_data(self):
        # Write camera params to as a JSON file
        file_path_cameraParams = f"cameraParams/cameraParams_{self._frame_id}.json"
        self.backend.schedule(write_json, path=file_path_cameraParams, data=self._frame_data["camera_data"])

        # # Write rgb to png image
        # file_path_rgb = f"rgb/rgb_{self._frame_id}.png"
        # self.backend.schedule(write_image, path=file_path_rgb, data=self._frame_data["rgb_data"])

        # Write depth data to npy
        file_path_depth = f"depth/depth_{self._frame_id}.npy"
        self.backend.schedule(write_np, path=file_path_depth, data=self._frame_data["depth_data"])

        # Write pointcloud to npy
        file_path_pcl = f"pcl/pcl_{self._frame_id}.npy"
        self.backend.schedule(write_np, path=file_path_pcl, data=self._frame_data["pcl_data"])
    
    
    def on_final_frame(self):
        self._frame_id = 0


# Register this writer
rep.WriterRegistry.register(ScanWriter)


## Scene ##

RT_SUBFRAMES = 2
z0 = 1.3

x_span = [-0.55, 0.55]
y_span = [-0.42, 0.42]

scale_factor = 1.5
x = np.arange(scale_factor * x_span[0], scale_factor * x_span[1], 0.1)
y = np.arange(scale_factor * y_span[0], scale_factor * y_span[1], 0.1)
xs, ys = np.meshgrid(x, y, indexing='ij')

cam = rep.create.camera(clipping_range=[0.01, 24])
rp = rep.create.render_product(cam, (1024, 1024))
writer = rep.WriterRegistry.get("ScanWriter")

rep.create.light(light_type="dome")

out_dir = '/home/haoyu-ma/Desktop' + "/_out_custom_event"
print(f"Writing data to {out_dir}")
writer.initialize(output_dir=out_dir)
writer.attach(rp)

def move_cam(cam_xform_prim, translate, orientation):

    cam_xform_prim.GetAttribute('xformOp:translate').Set((translate[0], translate[1], translate[2]))

    cam_xform_prim.GetAttribute('xformOp:rotateXYZ').Set((orientation[0], orientation[1], orientation[2]))

rep.trigger.register(move_cam)



async def run_scan_async(xs, ys, z):
    cam_xform_prim = rep.get.xform(path_pattern="Camera_Xform").get_output_prims()['prims'][0]

    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            for k in range(4):
                translation = [xs[i,j], ys[i,j], z]
                orientation = [0, -45, 90*k]
                rep.trigger.move_cam(cam_xform_prim, translation, orientation)

                # step the simulation to write one frame of data
                await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)


    # Wait until all the data is saved to disk
    await rep.orchestrator.wait_until_complete_async()



asyncio.ensure_future(run_scan_async(xs, ys, z0))
