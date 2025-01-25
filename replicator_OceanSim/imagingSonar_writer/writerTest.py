import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry

class writerTest(Writer):
    def __init__(
        self,
        output_dir,
    ):
        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        
        # Annotate with an rgb reading for ground truth
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb")) 
        # Annotate with a point cloud reading for sonar computation
        self.annotators.append(AnnotatorRegistry.get_annotator("pointcloud"))
        # Annotate with a camera info for sonar computation
        self.annotators.append(AnnotatorRegistry.get_annotator("camera_params"))
        self._frame_id = 0



    def write(self, data: dict):
        for annotator in data.keys():
            # If there are multiple render products the data will be stored in subfolders
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            if len(annotator_split) > 1:
                multi_render_prod = True
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            # rgb for gt
            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                filename_rgb = f"{render_product_path}rgb_{self._frame_id}.png"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_rgb} ..")
                self.backend.write_image(filename_rgb, data[annotator])

            # world positions
            if annotator.startswith("pointcloud"):
                if multi_render_prod:
                    render_product_path += "pointcloud/"
                filename_pcl = f"{render_product_path}pcl_{self._frame_id}.npy"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_pcl} ..")
                self.backend.write_array(filename_pcl, data[annotator]["data"])

                filename_normals = f"{render_product_path}normals_{self._frame_id}.npy"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_normals} ..")
                self.backend.write_array(filename_normals, data[annotator]["info"]["pointNormals"])

            # camera positions
            if annotator.startswith("camera_params"):
                if multi_render_prod:
                    render_product_path += "cameraViewTransform/"
                filename_viewTransform = f"{render_product_path}viewTransform_{self._frame_id}.npy"
                filename_cameraParam = f"{render_product_path}cameraParam_{self._frame_id}.npy"

                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_viewTransform} ..")
                self.backend.write_array(filename_viewTransform, data[annotator]['cameraViewTransform'])
                self.backend.write_array(filename_cameraParam, data[annotator])
        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0

WriterRegistry.register(writerTest)