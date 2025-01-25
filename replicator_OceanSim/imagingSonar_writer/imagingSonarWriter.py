import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

## Need to upgrade numpy in isaac interface to use np.atan2()!!!!!!!!!!!!!!!!
class imagingSonarWriter(Writer):
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
        render_product_list = []
        # TODO # 
        # can use a list\dict to separately contain datas belongs to the different render product
        filename_rgb = f"rgb_{self._frame_id}.png"
        print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_rgb} ..")
        self.backend.write_image(filename_rgb, data['rgb'])

        pcl = np.array(data['pointcloud']["data"])
        normals = np.array(data['pointcloud']["info"]["pointNormals"])
        viewTransform = np.array(data['camera_params']['cameraViewTransform'])
        sonar_map = self.make_sonar_data(pcl,  normals, viewTransform )
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(sonar_map[:,0], sonar_map[:,1], c=sonar_map[:,2], cmap='jet')
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        filename_sonar = f"sonar_{self._frame_id}.png"

        self.backend.write_image(filename_rgb, image_array)
        print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_sonar} ..")

        # for annotator in data.keys():
        #     # If there are multiple render products the data will be stored in subfolders
        #     annotator_split = annotator.split("-")
        #     render_product_path = ""
        #     multi_render_prod = False
        #     if len(annotator_split) > 1:
        #         multi_render_prod = True
        #         render_product_name = annotator_split[-1]
        #         render_product_path = f"{render_product_name}/"

        #     # rgb for gt
        #     if annotator.startswith("rgb"):
        #         if multi_render_prod:
        #             render_product_path += "rgb/"
        #         filename_rgb = f"{render_product_path}rgb_{self._frame_id}.png"
        #         print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_rgb} ..")
        #         self.backend.write_image(filename_rgb, data[annotator])

        #     # world positions
        #     if annotator.startswith("pointcloud"):
        #         if multi_render_prod:
        #             render_product_path += "pointcloud/"
        #         filename_pcl = f"{render_product_path}pcl_{self._frame_id}.npy"
        #         print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_pcl} ..")
        #         self.backend.write_array(filename_pcl, data[annotator]["data"])

        #         filename_normals = f"{render_product_path}normals_{self._frame_id}.npy"
        #         print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_normals} ..")
        #         self.backend.write_array(filename_normals, data[annotator]["info"]["pointNormals"])

        #     # camera positions
        #     if annotator.startswith("camera_params"):
        #         if multi_render_prod:
        #             render_product_path += "cameraViewTransform/"
        #         filename_viewTransform = f"{render_product_path}viewTransform_{self._frame_id}.npy"
        #         filename_cameraParam = f"{render_product_path}cameraParam_{self._frame_id}.npy"

        #         print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_viewTransform} ..")
        #         self.backend.write_array(filename_viewTransform, data[annotator]['cameraViewTransform'])
        #         self.backend.write_array(filename_cameraParam, data[annotator])
        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0



    def make_sonar_data(self, pcl:np.ndarray, normals:np.ndarray, viewTransform:np.ndarray):

        max_range = 8
        base_intensity = 255
        reflectivity = 1
        attenuation = 0.01


        normals = np.delete(arr=normals, obj=3, axis=1)
        viewTransform = viewTransform.reshape(4,4).T
        render_trans = -(np.transpose(viewTransform)[:3,3])
        render_rot = np.transpose(viewTransform)[:3,:3]
        dist = np.linalg.norm(pcl-render_trans, axis=1)
        directs = pcl - render_trans
        unit_directs = directs/np.linalg.norm(directs)

        theta = np.arccos(np.sum(unit_directs * normals, axis=1))
        intensity = base_intensity * reflectivity * np.cos(theta) * (1/max_range)**2 * np.exp(-attenuation * 2 * dist)

        # Pre-multiplication to produce transform with respect to world frame
        pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
        # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
        pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


        pcl_spher_local = self.cartesian_to_spherical(pcl_local)
        sonar_map = self.bin_intensity(1024, 1024, pcl_spher_local, intensity)
        sonar_map = np.array([sonar_map[:,0] * np.cos(sonar_map[:,1]), 
                            sonar_map[:,0] * np.sin(sonar_map[:,1]),
                            sonar_map[:,2]]).T
        
        return sonar_map

        
        
    def cartesian_to_spherical(self, cart_coords):
        """
        Convert N x 3 Cartesian coordinates to spherical coordinates.
        
        Parameters:
            cart_coords (numpy.ndarray): N x 3 array of Cartesian coordinates (x, y, z)
            
        Returns:
            numpy.ndarray: N x 3 array of spherical coordinates (r, theta, phi)
        """
        x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.atan2(y, x)
        phi = np.arccos(z / r)
        return np.vstack((r, theta, phi)).T

    def spherical_to_cartesian(self, sph_coords):
        """
        Convert N x 3 spherical coordinates to Cartesian coordinates.
        
        Parameters:
            sph_coords (numpy.ndarray): N x 3 array of spherical coordinates (r, theta, phi)
            
        Returns:
            numpy.ndarray: N x 3 array of Cartesian coordinates (x, y, z)
        """
        r, theta, phi = sph_coords[:, 0], sph_coords[:, 1], sph_coords[:, 2]
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.vstack((x, y, z)).T


    def bin_intensity(self, num_r_bins, num_azi_bins, pcl, intensity):
        min_r = pcl[:,0].min()
        max_r = pcl[:,0].max()
        min_azi = pcl[:,1].min()
        max_azi = pcl[:,1].max()
        r_bins = np.linspace(min_r, max_r, num_r_bins, endpoint=True)
        azi_bins = np.linspace(min_azi, max_azi, num_azi_bins, endpoint=True)

        intensity_binned, r_edges, azi_edges, _ = binned_statistic_2d(pcl[:,0], pcl[:,1], intensity, statistic='mean', bins=[r_bins, azi_bins])
        r_mid = (r_edges[:-1] + r_edges[1:]) / 2  
        azi_mid = (azi_edges[:-1] + azi_edges[1:]) / 2
        r, azi = np.meshgrid(r_mid, azi_mid, indexing='ij')
        
        return np.stack((r, azi, intensity_binned), axis=-1).reshape(-1,3)


WriterRegistry.register(imagingSonarWriter)