import asyncio
import os

import numpy as np
import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
from omni.replicator.core.scripts.functional import write_image, write_json, write_np

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
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb")) 


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

        self._frame_data["rgb_data"] = data["rgb"]
    
    
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

        # Write rgb to png image
        file_path_rgb = f"rgb/rgb_{self._frame_id}.png"
        self.backend.schedule(write_image, path=file_path_rgb, data=self._frame_data["rgb_data"])

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

    
r0 = 1
elevation = [45, -45] # deg
num_azi = 10

cam = rep.create.camera(clipping_range=[0.01, 8])
rp = rep.create.render_product(cam, (1024, 1024))
writer = rep.WriterRegistry.get("ScanWriter")

rep.create.light(light_type="dome")

out_dir = os.getcwd() + "/_out_custom_event"
print(f"Writing data to {out_dir}")
writer.initialize(output_dir=out_dir)
writer.attach(rp)



async def run_scan_async(cam, r0, num_azi, elevation):
    azi = np.linspace(-180, 180, num_azi).tolist()

    for i in range(len(elevation)):
        for j in range(len(azi)):
            with cam:
                rep.modify.pose_orbit(
                    barycentre=(0,0,0),
                    distance=r0,
                    azimuth=azi[j],
                    elevation=elevation[i],
                    look_at_barycentre=True,
                    )
                
            # step the simulation to write one frame of data
            await rep.orchestrator.step_async(rt_subframes=8)


    # Wait until all the data is saved to disk
    await rep.orchestrator.wait_until_complete_async()



asyncio.ensure_future(run_scan_async(cam, r0, num_azi, elevation))
