import numpy as np
import matplotlib.pyplot as plt

saved_path = '/home/haoyu-ma/Desktop/viz_test'

# # single ultrasonic test
# azi = np.load(saved_path+'/azi.npy')
# zen =        np.load(saved_path+'/zen.npy')
# envelope_array =        np.load(saved_path+'/envelope_array.npy')
# envelope = np.load(saved_path+'/envelope.npy')
# depth =        np.load(saved_path+'/depth.npy')
# intensity =         np.load(saved_path+'/intensity.npy')
# num_rows =       np.load(saved_path+'/num_rows.npy')
# num_cols =       np.load(saved_path+'/num_cols.npy')

# print(f'azi:{azi}')
# print(f'zen:{zen}')
# print(f'depth:{depth}')
# print(f'intensity_of_sensor_0;{intensity}')
# print(f'num_rows:{num_rows}')
# print(f'num_cols:{num_cols}')
# print(f'size of envelope array:{envelope_array.shape}')
# print(f'sum_of_envelope_array[0]:{np.sum(envelope_array[0])}')
# print(f'sum_of_envelope_array[1]:{np.sum(envelope_array[1])}')
# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax1.scatter(np.arange(0,len(envelope_array[0]),1), envelope_array[0])
# ax1.grid(True)
# ax1.set_title('array_0')

# ax2 = fig.add_subplot(2,2,4)
# ax2.scatter(np.arange(0,len(envelope_array[1]),1), envelope_array[1])
# ax2.grid(True)
# ax2.set_title('array_1')

# plt.show()




# Multi ultrasonic sensor test
rotations = np.load(saved_path+'/rotations.npy')
envelope_array = np.load(saved_path+'/envelope_array.npy')
print(f'envelope_shape:{envelope_array.shape}')
print(f'sum of whole envelope;{np.sum(envelope_array[:,:])}')
rotations_rad = np.deg2rad(rotations)
r = np.linspace(0,envelope_array.shape[1],envelope_array.shape[1])
sonar_map = np.zeros([envelope_array.shape[0], envelope_array.shape[1], 3])
for i in range(envelope_array.shape[0]):
    for j in range(envelope_array.shape[1]):
        sonar_map[i,j,:] = np.array([r[j] * np.cos(rotations_rad[i]), r[j] * np.sin(rotations_rad[i]), envelope_array[i,j]])
sonar_map = sonar_map.reshape(-1,3)
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(sonar_map[:,0], sonar_map[:,1], c=sonar_map[:,2], cmap='jet',s=0.01)

ax2 = fig.add_subplot(1,2,2)
ax2.scatter(r, envelope_array[98,:])
plt.show()
# # Load your data (replace 'saved_path' with your actual path)
# sonar_map = np.load(saved_path + '/sonar_map.npy')
# cart_coor = np.load(saved_path + '/cart_coord.npy')
# envelope = np.load(saved_path + '/envelope.npy')
# # Create a figure
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(projection='3d')
# # Plot Cartesian coordinates
# ax1.scatter(cart_coor[:, 0], cart_coor[:, 1], cart_coor[:, 2], c=cart_coor[:, 2], cmap='jet',s=0.5)



# plt.figure(figsize=(12, 6))

# plt.scatter(sonar_map[:,0], sonar_map[:,1], c=sonar_map[:,2], cmap='jet')


# plt.figure(figsize=(12, 6))
# plt.scatter(np.arange(0,len(envelope[0]),1), envelope[0])

# plt.show()


