import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
from scipy.stats import binned_statistic_2d
import json


with open('cameraParams_1.json', 'r') as file:
    data = json.load(file)



pcl = np.load('pcl_1.npy')
depth = np.load('depth_1.npy')
viewTransform = np.array(data['camera_view_matrix']).T
print(pcl.shape)
print(depth.shape)

pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
# Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcl_local)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

o3d.visualization.draw_geometries([frame, point_cloud])


# cmap = plt.get_cmap('jet')
# colors = cmap(intensity/np.max(intensity))[:,:3]

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pcl)
# point_cloud.colors = o3d.utility.Vector3dVector(colors)
# point_cloud.normals = o3d.utility.Vector3dVector(normals)
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame()


# o3d.visualization.draw_geometries([frame, point_cloud])
