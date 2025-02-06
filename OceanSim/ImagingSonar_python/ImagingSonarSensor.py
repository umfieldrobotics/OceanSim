from omni.isaac.sensor import Camera
from omni.isaac.core.prims import BaseSensor
import omni.replicator.core as rep
import isaacsim.core.utils.rotations as rotations_utils

from omni.replicator.core.scripts.functional import write_np
from scipy.stats import binned_statistic_2d
import numpy as np


class ImagingSonarSensor:

    def __init__(self, prim_path : str, 
                 trans : list[float]= [0.0, 0.0, 0.0], 
                 orients: list[float] = rotations_utils.euler_angles_to_quat(np.array([0,0,0]))
                 ):
        
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = 200 # m
        self.min_range = 0.2 # m
        self.range_res = 0.008 # m
        self.update_rate = 40 # Hz (max update rate)
        self.hori_fov = 130 # degree
        self.vert_fov = 20 # degree
        self.num_beams = 256 # (max number of beams)
        self.angular_res = 2 # degree
        self.beam_separation = 0.5 # degree

        # TODO!!!!!!!!!!!!!!!!!! Tomorrow
        # Export the sonar data and test if the HFOV and VFOV are really this degrees
        
        
        # TODO 
        # Needs to figure out how to get resolution in pixels 
        # given above angular res, beam separations...
        self.resolution = (1024, 1024)
        sensor_xform_prim_path = prim_path + '/sensor_xform'
        self.sonar = BaseSensor(
            prim_path=sensor_xform_prim_path,
            )
        self.camera_prim_path = sensor_xform_prim_path + '/Camera'
        self.camera = Camera(
            prim_path=self.camera_prim_path,
            translation=trans,
            orientation=orients
            )
        self.camera.initialize()

        self.focal_length = self.camera.get_focal_length()
        self.hori_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.hori_fov/2))
        self.vert_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.vert_fov/2))


        self.camera.set_clipping_range(
            near_distance=self.min_range,
            far_distance=self.max_range
        )

        self.camera.set_horizontal_aperture(float(self.hori_aper))
        self.camera.set_vertical_aperture(float(self.vert_aper))
    

    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # For now, data is generated per simulation tick
    def initialize(self, output_dir : str = None):
        self.writing = False
        self.id = 0
        self.rp = rep.create.render_product(
            camera=self.camera_prim_path,
            resolution=self.resolution
            )

        self.pointcloud_annot = rep.AnnotatorRegistry.get_annotator(
            name="pointcloud",
            init_params={"includeUnlabelled": True},
            do_array_copy=True
            )
        
        self.cameraParams_annot = rep.AnnotatorRegistry.get_annotator(
            name="CameraParams",
            do_array_copy=True
            )
        # do_array_copy: If True, retrieve a copy of the data array. 
        # This is recommended for workflows using asynchronous
        # backends to manage the data lifetime. 
        # Can be set to False to gain performance if the data is 
        # expected to be used immediately within the writer. Defaults to True.

        self.pointcloud_annot.attach(self.rp)
        self.cameraParams_annot.attach(self.rp)

        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})


    def make_sonar_data(self):

        max_range = self.max_range
        base_intensity = 255
        reflectivity = 1
        attenuation = 0.1


        pcl=self.pointcloud_annot.get_data()['data'],
        normals=self.pointcloud_annot.get_data()['info']['pointNormals'],
        viewTransform=self.cameraParams_annot.get_data()['cameraViewTransform']

        def cartesian_to_spherical(cart_coords):
            x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
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

        

        normals = np.delete(arr=normals, obj=3, axis=1)
        viewTransform = viewTransform.reshape(4,4).T
        render_trans = -(viewTransform[:3,:3].T @ viewTransform[:3,3])
        dist = np.linalg.norm(pcl-render_trans, axis=1)
        directs = pcl - render_trans
        unit_directs = directs/np.linalg.norm(directs)

        theta = np.arccos(np.sum(unit_directs * normals, axis=1))
        # Formula to calculate the intensity 
        intensity = base_intensity * reflectivity * np.abs(np.cos(theta)) * (1/max_range)**2 * np.exp(-attenuation * 2 * dist)
        # Pre-multiplication to produce transform with respect to world frame
        pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
        # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
        pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        # Convert to spherical coordinates for binning in sperhical r and azi
        pcl_spher_local = cartesian_to_spherical(pcl_local)
        # Binning the intensity to collapse into 2D
        sonar_data = bin_intensity(1024, 1024, pcl_spher_local, intensity)
        # TODO Look at why there are so many nan values in intensities
        sonar_data = sonar_data[~np.isnan(sonar_data[:,2])]
        # Normalized the intensity
        normalized_intensity = (sonar_data[:,2] - sonar_data[:,2].min()) / (sonar_data[:,2].max() - sonar_data[:,2].min())
        # Convert back to cartesian corrdiantes and map normalized intensity to 0-255
        sonar_data = np.array([sonar_data[:,0] * np.cos(sonar_data[:,1]), 
                            sonar_data[:,0] * np.sin(sonar_data[:,1]),
                            np.round(normalized_intensity * 255)
                            ]).T
        
        if self.writing:
            self.backend.schedule(
                fn=write_np, 
                path=f"sonar_data_{self.id}.npy", 
                data=sonar_data)
        
        self.id += 1

        return sonar_data


    def close(self):
        if self.writing:
            self.backend.wait_until_done()
        
        self.pointcloud_annot.detach(self.rp)
        self.cameraParams_annot.detach(self.rp)
        self.rp.clear()
        