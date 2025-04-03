# Build Your Own Digital Twin in OceanSim
<!-- (../../media/oceansim_digital_twin.gif) -->
![OceanSim Framework](../../media/oceansim_digital_twin.gif)

In this documentation, we show an example of building a digital twin for OceanSim. We hope this example can help you understand how to build your own digital twin in OceanSim.

## 3D Scan of the Environment
We use a ZED stereo camera to scan the environment. The ZED camera is a stereo camera that can capture depth information and RGB images. We use the camera to scan the environment, and generate a folder of RGB images, which can be used to create a 3D model of the environment.

The next step is to use the RGB images to create a 3D model of the environment. Among various 3D reconstruction softwares, we use [Metashape](https://www.agisoft.com/) to create a 3D model of the environment. The process is as follows:
1. Import the RGB images into Metashape.
2. Align the images to create a sparse point cloud.
3. Build a dense point cloud from the sparse point cloud.
4. Build a mesh from the dense point cloud.
5. Build texture for the model.

We recommend checking the online tutorials of Metashape for more details.

Please note the 3D model reconstructed by Metashape is not metric-scaled. To recover the correct metric scale, during the 3D scan, please also measure some reference distances between noticeable markers in the environment. In Metashape, you can use the `Scale Bar` tool to set the scale of the model. Please follow the steps in this [video](https://youtu.be/lp5eIOUJxCE?si=E3ZoLXriJAfuRdWU&t=359).

## Importing the 3D Scan
We tested exporting the 3D model from Metashape as a `.obj` file and importing it into NVIDIA Isaac Sim. If you prefer `.usd` format, we recommend doing the conversion in Isaac Sim for better compatibility.

Then you should be able to use this 3D model in OceanSim. Please refer to the provided [example](running_example.md) and modify it to fit your needs.