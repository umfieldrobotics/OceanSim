import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts.functional import write_image, write_npy

class ImagingSonarWriter_Pt(Writer):
    def __init__(
        self,
        output_dir,
    ):
        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        
        # Annotate with an rgb reading for ground truth
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb")) 
        # Annotate with a world pos reading for sonar computation
        self.annotators.append(AnnotatorRegistry.get_annotator("PtWorldPos"))
        
        self._frame_id = 0

    def write(self, data: dict):
        for annotator in data.keys():
            # If there are multiple render products the data will be stored in subfolders
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            # rgb
            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                filename = f"{render_product_path}rgb_{self._frame_id}.png"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename} ..")
                self.backend.write_image(filename, data[annotator])

            # # semantic_segmentation
            # if annotator.startswith("normals"):
            #     if multi_render_prod:
            #         render_product_path += "normals/"
            #     filename = f"{render_product_path}normals_{self._frame_id}.png"
            #     print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename} ..")
            #     colored_data = ((data[annotator] * 0.5 + 0.5) * 255).astype(np.uint8)
            #     self.backend.write_image(filename, colored_data)

            # world positions
            if annotator.startswith("PtWorldPos"):
                if multi_render_prod:
                    render_product_path += "PtWorldPos/"
                filename = f"{render_product_path}PtWorldPos_{self._frame_id}.npy"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename} ..")
                self.backend.write_array(filename, data[annotator])
        
        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0

WriterRegistry.register(ImagingSonarWriter_Pt)