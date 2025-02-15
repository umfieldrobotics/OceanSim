# import numpy as np
# import matplotlib.pyplot as plt
# import open3d as o3d
# import json


###############################################################################
# Viz function for viewing object scan result
# id = 154
# object = 'rig'



# with open(f'{object}/cameraParams/cameraParams_{id}.json', 'r') as file:
#     data = json.load(file)

# pcl = np.load(f'{object}/pcl/pcl_{id}.npy')
# depth = np.load(f'{object}/depth/depth_{id}.npy')
# viewTransform = np.array(data['camera_view_matrix']).T
# # rgb = plt.imread(f'{object}/rgb/rgb_{id}.png')

# pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
# # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
# # pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
# pcl_local = np.delete(pcl_local, obj=3, axis=1)

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pcl_local)
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

# o3d.visualization.draw_geometries([frame, point_cloud])

# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(depth)
# ax1.set_title('depth')

# # ax2 = fig.add_subplot(1,2,2)
# # ax2.imshow(rgb)
# # ax2.set_title('rgb')
# plt.show()



##########################################################
# Viz function for converting raw sonar map data to videos
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
# from io import BytesIO


# input_folder = "/home/haoyu-ma/Desktop/MHL_replica"  # Folder containing sonar_data_{id}.npy files
# output_video = input_folder + "/output_video.mp4"  # Output video file
# files = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')] , key=lambda x: int(x.split("_")[2].split(".")[0]))
# # Parameters
# num_frames = len(files)  # Number of frames (scatter plots)
# fps = 8  # Frames per second
# frame_size = (800, 800)  # Width x height of the video frames
# print(files)
# # Create a video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
# video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

# colormap = plt.get_cmap('gray')
# # Generate scatter plots and create video frames in memory
# for i in range(num_frames):
#     # Generate random data for the scatter plot (replace this with your data)
#     sonar_data = np.load(input_folder + '/' + files[i])
#     # Create a scatter plot
#     fig = plt.figure(figsize=(8, 8))
#     sonar_data_flat = sonar_data[:,:,2].squeeze()
#     plt.imshow(sonar_data_flat, cmap='gray', aspect=0.5)
#     ax = plt.gca()  
#     # ax.invert_yaxis()
#     ax.invert_xaxis()
#     ax.set_facecolor("black")

#     buffer = BytesIO()
#     plt.savefig(buffer, format="png", dpi=100)
#     plt.close()
    
#     # Read the buffer into a NumPy array
#     buffer.seek(0)
#     file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
#     frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    

    
#     # Write the frame to the video
#     video_writer.write(frame)

# # Release the video writer
# video_writer.release()

# print(f"Video saved to {output_video}")


##################################################
# Viz function for a single plot of a sonar map

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

input_folder = "/home/haoyu-ma/Desktop/MHL_replica"  # Folder containing sonar_data_{id}.npy files

id = 10
colormap = plt.get_cmap('gray')  # You can use "plasma", "inferno", "magma", etc.

sonar_data = np.load(input_folder + f"/sonar_data_{id}.npy")
sonar_data_flat = sonar_data[:,:,2].squeeze()

sonar_data = sonar_data.reshape(-1,3)
# Create a scatter plot
fig = plt.figure(figsize=(8, 8))
plt.scatter(sonar_data[:,0], sonar_data[:,1], c=sonar_data[:,2], cmap=colormap, s=0.1, marker='o')
plt.colorbar(label="Intensity Value")
plt.ylim([0,sonar_data[:,1].max()])
ax = plt.gca()  
ax.set_facecolor("black")
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.imshow(sonar_data_flat, cmap='gray', aspect=0.5)
ax = plt.gca()  
# ax.invert_yaxis()
ax.invert_xaxis()
ax.set_facecolor("black")
plt.show()

pcl = np.load(input_folder + f"/pcl_local_{id}.npy")
intensity = np.load(input_folder + f"/intensity_{id}.npy")
intensity_normalized = intensity  / intensity.max()
colors = colormap(intensity_normalized)[:, :3]  # Extract RGB (ignore alpha channel)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcl)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

o3d.visualization.draw_geometries([frame, point_cloud])
