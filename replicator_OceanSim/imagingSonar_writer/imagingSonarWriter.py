import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import time

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

        start_time = time.time()
        render_product_list = []
        # TODO # 
        # can use a list\dict to separately contain datas belongs to the different render product
        
        
        # save ground truth rgb
        filename_rgb = f"rgb_{self._frame_id}.png"
        print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_rgb} ..")
        self.backend.write_image(filename_rgb, data['rgb'])

        # get point cloud, normals, and camera pose for computing sonar data
        pcl = np.array(data['pointcloud']["data"])
        normals = np.array(data['pointcloud']["info"]["pointNormals"])
        viewTransform = np.array(data['camera_params']['cameraViewTransform'])
        sonar_data = self.make_sonar_data(pcl,  normals, viewTransform )
        
        # convert sonar data to a plot and save as rgba
        sonar_map = self.make_sonar_map(sonar_data)
        
        # save sonar map
        filename_sonar = f"sonar_{self._frame_id}.png"
        self.backend.write_image(filename_sonar, sonar_map)
        print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename_sonar} ..")

        print(f'Writting 1 frame of annotated data takes {(time.time() - start_time):.2f} sec')
        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0



    def make_sonar_data(self, pcl:np.ndarray, normals:np.ndarray, viewTransform:np.ndarray):
        def arctan_with_quadrants(y, x):
            # Compute arctan for the ratio y/x
            angle = np.arctan(np.divide(y, x, where=x != 0))  # Avoid division by zero with `where`
            
            # Adjust angles based on the quadrant
            angle = np.where((x > 0), angle, angle + np.pi)  # Quadrants II and III
            angle = np.where((x < 0) & (y < 0), angle - 2 * np.pi, angle)  # Quadrant III correction
            angle = np.where((x == 0) & (y > 0), np.pi / 2, angle)  # Positive y-axis
            angle = np.where((x == 0) & (y < 0), -np.pi / 2, angle)  # Negative y-axis

            return angle
        def cartesian_to_spherical(cart_coords):
            x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = arctan_with_quadrants(y, x)
            phi = np.arccos(z / r)
            return np.vstack((r, theta, phi)).T


        def bin_intensity(num_r_bins, num_azi_bins, pcl, intensity):
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

        max_range = 4
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
        intensity = base_intensity * reflectivity * np.abs(np.cos(theta)) * (1/max_range)**2 * np.exp(-attenuation * 2 * dist)

        # Pre-multiplication to produce transform with respect to world frame
        pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
        # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
        pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


        pcl_spher_local = cartesian_to_spherical(pcl_local)
        sonar_map = bin_intensity(1024, 1024, pcl_spher_local, intensity)
        sonar_map = np.array([sonar_map[:,0] * np.cos(sonar_map[:,1]), 
                            sonar_map[:,0] * np.sin(sonar_map[:,1]),
                            sonar_map[:,2]]).T
        
        return sonar_map
    
    def make_sonar_map(self, sonar_data:np.ndarray):
        
        fig = plt.figure(dpi=600)
        ax1 = fig.add_subplot(1,1,1)
        sonar_plot = ax1.scatter(sonar_data[:,0], sonar_data[:,1], c=sonar_data[:,2], cmap='jet', s=0.05, marker='.')
        fig.colorbar(mappable=sonar_plot, ax=ax1)
        fig.canvas.draw()
        
        return np.array(fig.canvas.renderer.buffer_rgba())

        
        


WriterRegistry.register(imagingSonarWriter)