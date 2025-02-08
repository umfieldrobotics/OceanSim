from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import isaacsim.core.utils.rotations as rotations_utils
import numpy as np
from omni.replicator.core.scripts.functional import write_np
from scipy.stats import binned_statistic_2d
# TODO #
# Implement 'pointSemantic': dnarray[...],  #  shape=(<num_points>), dtype=uint8), from PointCloud annotator 
# to take into account of material property
class ImagingSonarSensor:

    def __init__(self, prim_path : str, 
                 trans : list[float]= [0.0, 0.0, 0.0], 
                 orients: list[float] = rotations_utils.euler_angles_to_quat(np.array([0,0,0]))
                 ):
        
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = 5 # m (max is 200 m in datasheet )
        self.min_range = 0.2 # m (min is 0.2 m in datasheet)
        self.range_res = 0.008 # m (datasheet is 0.008 m)
        self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!
        self.hori_fov = 130 # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = 20 # degree (vert_fov is 20 degrees in datasheet)
        self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        self.angular_res = 0.1 # degree (datasheet is 2 deg)
        self.beam_separation = 0.5 # degree


        # Create range bins
        self.r_bins = np.arange(self.min_range, self.max_range, self.range_res)
        #Create azimuthal bins
        self.azi_bins = np.arange(np.deg2rad(90-self.hori_fov/2), np.deg2rad(90+self.hori_fov/2), np.deg2rad(self.angular_res))

        # We introduce this factor to adjust raycast density
        # (Equivalently, adjust the beam_separation) 
        # Increase this value by 1 will quadraple the total number of raycasts
        self.ray_factor = 10

        self.AR = self.hori_fov / self.vert_fov
        self.hori_res = int(self.ray_factor * (self.hori_fov / self.beam_separation))
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam vertical separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.

        print(f'resolution: {self.hori_res} x {self.vert_res}')
        print(f'bin: {self.azi_bins.shape[0]} x {self.r_bins.shape[0]}')
        self.camera_prim_path = prim_path + '/Camera'
        self.camera = Camera(
            prim_path=self.camera_prim_path,
            translation=trans,
            orientation=orients,
            resolution=(self.hori_res, self.vert_res)
            )
        self.camera.set_clipping_range(
            near_distance=self.min_range,
            far_distance=self.max_range
        )
        # This is a bug. Needs to call initialize() before changing aperture
        # https://forums.developer.nvidia.com/t/error-when-setting-a-cameras-vertical-horizontal-aperture/271314
        self.camera.initialize()

        # Assume the default focal length to compute the desired horizontal aperture
        # The reason why we are doing this is because Isaac sim will fix vertical aperture
        # given aspect ratio for mandating square pixles
        # https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5
        focal_length = self.camera.get_focal_length()
        horizontal_aper = 2 * focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.camera.set_horizontal_aperture(horizontal_aper)
        # Notice if you would like to observe sonar view from linked viewport.
        # Only horizontal fov is displayed correctly while the vertical fov is
        # followed by your viewport aspect ratio settings.
        
        # Future: maybe able to increase W or H resolution and cut out data points outside of the view
        # But this method requires us to think about math about combinations 
        # of f and A_h will enable us to cut out the least number of points.


    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # For now, data is generated per simulation tick
    def initialize(self, output_dir : str = None):
        self.writing = False
        self.scan_data = {}
        self.id = 0
        self.rp = rep.create.render_product(
            camera=self.camera_prim_path,
            resolution=(self.hori_res, self.vert_res)
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



    def scan(self):
        self.scan_data['pcl'] = self.pointcloud_annot.get_data()['data']
        self.scan_data['normals'] = self.pointcloud_annot.get_data()['info']['pointNormals']
        self.scan_data['viewTransform'] = self.cameraParams_annot.get_data()['cameraViewTransform']



    def make_sonar_data(self):

        reflectivity = 1
        attenuation = 0.3

        gau_noise_param = 0.2
        ray_noise_param = 0.15
        intensity_offset = 0.2
        intensity_gain = 1.5

        if self.scan_data['pcl'].size != 0:
            pcl = self.scan_data['pcl']
            normals=self.scan_data['normals'][:,:3]
            viewTransform=self.scan_data['viewTransform']
        else:
            return
        
        def cartesian_to_spherical(cart_coords):
            x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)
            return np.vstack((r, theta, phi)).T


        viewTransform = viewTransform.reshape(4,4).T
        render_trans = -(viewTransform[:3,:3].T @ viewTransform[:3,3])
        dist = np.linalg.norm(pcl-render_trans, axis=1)
        directs = pcl - render_trans
        unit_directs = directs/np.linalg.norm(directs)

        cos_theta = np.sum(unit_directs * normals, axis=1)


        # TODO#
        # Implement clustering raycasting 
        # Calculate the reflected vector
        # self.reflects = unit_directs - 2 * cos_theta * normals
        # print(np.linalg.norm(self.reflects))

        # Formula to calculate the intensity 
        intensity =  reflectivity * np.abs(cos_theta) * np.exp(-attenuation * dist)
        # Pre-multiplication to produce transform with respect to world frame
        pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
        # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
        pcl_local = pcl_local[:,:3] @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        # Convert to spherical coordinates for binning in sperhical r and azi
        pcl_spher_local = cartesian_to_spherical(pcl_local)

        # Binning the intensity to collapse into 2D (bins without data is nan )
        intensity_binned, r_edges, azi_edges, _ = binned_statistic_2d(
            x=pcl_spher_local[:,0], 
            y=pcl_spher_local[:,1], 
            values=intensity, 
            statistic='mean', 
            bins=[self.r_bins, self.azi_bins]
            )
        # Calculate the mid point of these bins
        r_mid = (r_edges[:-1] + r_edges[1:]) / 2  
        azi_mid = (azi_edges[:-1] + azi_edges[1:]) / 2
        # Use the mid points of those bins to create new mesh grid
        r, azi = np.meshgrid(r_mid, azi_mid, indexing='ij')
        # Normalize the intensity
        normalized_intensity = intensity_binned / np.nanmax(intensity_binned)
        # Apply offset
        normalized_intensity += intensity_offset
        # Apply gain
        normalized_intensity *= intensity_gain
        # Set all nan intensity to 0
        normalized_intensity = np.nan_to_num(normalized_intensity, nan=0)
        # Calculate noise
        gau_noise = np.random.normal(loc=0, scale=gau_noise_param, size=r.shape)
        ray_noise = np.random.rayleigh(scale=ray_noise_param, size=r.shape)
        range_dependent_ray_noise = r**2/self.max_range**2*(1 + np.exp(-np.abs(azi)))*ray_noise 
        # Apply the noise
        normalized_intensity = normalized_intensity * (0.5 + gau_noise) + range_dependent_ray_noise
        # Clip the intensity
        normalized_intensity = np.clip(normalized_intensity, 0, 1)
        # Make the sonar data
        sonar_data = np.stack((r, azi, normalized_intensity), axis=-1).reshape(-1,3)

        # Convert back to cartesian corrdiantes and map normalized intensity to 0-255
        sonar_data = np.array([sonar_data[:,0] * np.cos(sonar_data[:,1]), 
                            sonar_data[:,0] * np.sin(sonar_data[:,1]),
                            sonar_data[:,2]
                            ]).T

        if self.writing:
            self.backend.schedule(
                fn=write_np, 
                path=f"sonar_data_{self.id}.npy", 
                data=sonar_data)
            self.backend.schedule(
                fn=write_np,
                path=f"pcl_{self.id}.npy",
                data=pcl_local
            )
            self.backend.schedule(
                fn=write_np,
                path=f"intensity_{self.id}.npy",
                data=intensity
            )
        
        self.id += 1

        return sonar_data


    def close(self):
        self.pointcloud_annot.detach(self.rp)
        self.cameraParams_annot.detach(self.rp)

        self.pointcloud_annot = None
        self.cameraParams_annot = None
