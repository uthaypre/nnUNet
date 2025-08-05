# import vtkmodules.all as vtk

# def generate_mesh_from_nifti(nii_path: str,
#                              iso_value: float = 300.0,
#                              is_labelmap: bool = False) -> vtk.vtkPolyData:
#     # 1. Read .nii or .nii.gz
#     reader = vtk.vtkNIFTIImageReader()
#     reader.SetFileName(nii_path)
#     reader.Update()

#     image = reader.GetOutput()

#     # Optional: convert to float & smooth (improves mesh quality)
#     # Especially useful before contouring a mask as a float fraction
#     cast = vtk.vtkImageCast()
#     cast.SetInputData(image)
#     cast.SetOutputScalarTypeToFloat()
#     cast.Update()

#     smooth = vtk.vtkImageGaussianSmooth()  # optional
#     smooth.SetInputData(cast.GetOutput())
#     smooth.SetStandardDeviations(1.0, 1.0, 1.0)
#     smooth.Update()

#     # 2. Extract surface via marching cubes
#     if is_labelmap:
#         contour = vtk.vtkDiscreteMarchingCubes()
#         contour.SetInputConnection(smooth.GetOutputPort())
#         contour.GenerateValues(1, iso_value, iso_value)  # only label==iso_value
#     else:
#         contour = vtk.vtkMarchingCubes()  # or vtkFlyingEdges3D()
#         contour.SetInputConnection(smooth.GetOutputPort())
#         contour.SetValue(0, iso_value)

#     contour.ComputeNormalsOn()
#     contour.Update()

#     # 3. Transform mesh into NIfTI’s anatomical space if necessary
#     transform = vtk.vtkTransform()
#     if reader.GetQFormMatrix():
#         transform.Concatenate(reader.GetQFormMatrix())
#     elif reader.GetSFormMatrix():
#         transform.Concatenate(reader.GetSFormMatrix())

#     tf_filter = vtk.vtkTransformPolyDataFilter()
#     tf_filter.SetInputConnection(contour.GetOutputPort())
#     tf_filter.SetTransform(transform)
#     tf_filter.Update()

#     return tf_filter.GetOutput()

# def show_polydata(polydata: vtk.vtkPolyData):
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputData(polydata)
#     mapper.ScalarVisibilityOff()

#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetColor(1.0, 0.7, 0.3)

#     ren = vtk.vtkRenderer()
#     ren.AddActor(actor)
#     ren.SetBackground(0.1, 0.1, 0.2)

#     renWin = vtk.vtkRenderWindow()
#     renWin.AddRenderer(ren)
#     renWin.SetSize(800, 800)

#     iren = vtk.vtkRenderWindowInteractor()
#     iren.SetRenderWindow(renWin)

#     renWin.Render()
#     iren.Initialize()
#     iren.Start()

# # Example use:
# mesh = generate_mesh_from_nifti("/mnt/d/projectsD/datasets/AbdomenCT-1K/images/Case_00001_0000.nii.gz", iso_value=300.0)
# show_polydata(mesh)
# #####################################################################################################
# # from tqdm import tqdm
# # import os
# # from random import randint

# # import numpy as np
# # import pandas as pd

# # import nibabel as nib
# # import pydicom as pdm
# # import nilearn as nl
# # import nilearn.plotting as nlplt
# # # import nrrd
# # # import h5py

# # import matplotlib.pyplot as plt
# # from matplotlib import cm
# # import matplotlib.animation as anim

# # import imageio
# # from skimage.transform import resize
# # from skimage.util import montage

# # # from IPython.display import Image as show_gif

# # import warnings
# # warnings.simplefilter("ignore")

# # sample_filename = "/mnt/d/projectsD/datasets/AbdomenCT-1K/images/Case_00001_0000.nii.gz"
# # sample_filename_mask = "/mnt/d/projectsD/datasets/AbdomenCT-1K/labels/Case_00001.nii.gz"

# # sample_img = nib.load(sample_filename)
# # sample_img = np.asanyarray(sample_img.dataobj)
# # sample_mask = nib.load(sample_filename_mask)
# # sample_mask = np.asanyarray(sample_mask.dataobj)
# # print("img shape ->", sample_img.shape)
# # print("mask shape ->", sample_mask.shape)
# # # slice_n = 100
# # # fig, ax = plt.subplots(2, 3, figsize=(25, 15))

# # # ax[0, 0].imshow(sample_img[slice_n, :, :])
# # # ax[0, 0].set_title(f"image slice number {slice_n} along the x-axis", fontsize=18, color="red")
# # # ax[1, 0].imshow(sample_mask[slice_n, :, :])
# # # ax[1, 0].set_title(f"mask slice {slice_n} along the x-axis", fontsize=18, color="red")

# # # ax[0, 1].imshow(sample_img[:, slice_n, :])
# # # ax[0, 1].set_title(f"image slice number {slice_n} along the y-axis", fontsize=18, color="red")
# # # ax[1, 1].imshow(sample_mask[:, slice_n, :])
# # # ax[1, 1].set_title(f"mask slice number {slice_n} along the y-axis", fontsize=18, color="red")

# # # ax[0, 2].imshow(sample_img[:, :, slice_n])
# # # ax[0, 2].set_title(f"image slice number {slice_n} along the z-axis", fontsize=18, color="red")
# # # ax[1, 2].imshow(sample_mask[:, :, slice_n])
# # # ax[1, 2].set_title(f"mask slice number {slice_n}along the z-axis", fontsize=18, color="red")
# # # fig.tight_layout()
# # # plt.show()

# # # import matplotlib.pyplot as plt
# # # from mpl_toolkits.mplot3d import proj3d

# # # fig = plt.figure(figsize=(8, 8))
# # # ax = fig.add_subplot(111, projection='3d')

# # # ax.scatter(sample_img[:, 0], sample_mask[:, 1], sample_mask[:, 2], c=sample_mask.flatten(), cmap='viridis', alpha=0.5)
# # # plt.show()


# # ###############################################################################################
# # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# # import numpy as np
# # from skimage import measure

# # def plot_3d(image, threshold=-300): 
# #     p = image.transpose(2,1,0)
# #     verts, faces, normals, values = measure.marching_cubes(p, threshold)
# #     fig = plt.figure(figsize=(10, 10))
# #     ax = fig.add_subplot(111, projection='3d')
# #     mesh = Poly3DCollection(verts[faces], alpha=0.1)
# #     face_color = [0.5, 0.5, 1]
# #     mesh.set_facecolor(face_color)
# #     ax.add_collection3d(mesh)
# #     ax.set_xlim(0, p.shape[0])
# #     ax.set_ylim(0, p.shape[1])
# #     ax.set_zlim(0, p.shape[2])

# #     plt.show()
# # plot_3d(sample_img, threshold=300)

# # # from vtkplotter import *

# # # volume = load(mydicomdir) #returns a vtkVolume object
# # # show(volume, bg='white')

##############################################

import open3d as o3d
import SimpleITK as sitk
import numpy as np
from skimage import measure

ct_image = sitk.ReadImage("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/imagesTs/BDMAP_00000005_0000.nii.gz")  # or DICOM folder
ct_mask = sitk.ReadImage("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/labelsTs/BDMAP_00000005.nii.gz")  # or DICOM folder
ct_array = sitk.GetArrayFromImage(ct_image)  # shape: (slices, height, width)
print("CT image shape:", ct_array.shape)

mask_array = sitk.GetArrayFromImage(ct_mask)  # shape: (slices, height, width)
print("CT mask shape:", mask_array.shape)
# 2. Get voxel spacing for scaling
# spacing = ct_image.GetSpacing()  # (x, y, z)
# spacing = spacing[::-1]  # reverse to match numpy array order

# # 3. Apply thresholding (HU value, e.g., for bone or soft tissue)
# threshold = 40  # Adjust based on your use case
# verts, faces, _, _ = measure.marching_cubes(ct_array, level=threshold, spacing=spacing)

# # 4. Create Open3D TriangleMesh
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(verts)
# mesh.triangles = o3d.utility.Vector3iVector(faces)

# mesh_pcd = mesh.sample_points_uniformly(number_of_points=10000)  # Optional: sample points for visualization
# mesh_pcd = mesh_pcd.voxel_down_sample(voxel_size=0.0006)  # Optional: downsample for performance


# # 5. Optional: clean and visualize
# # mesh.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh])

import nibabel as nib

mask = nib.load("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/labelsTs/BDMAP_00000005.nii.gz")
data = mask.get_fdata()
# print(data.shape)
label = 20
mask = (data == label).astype(np.uint8)

verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
organ_mesh = o3d.geometry.TriangleMesh()
organ_mesh.vertices = o3d.utility.Vector3dVector(verts)
organ_mesh.triangles = o3d.utility.Vector3iVector(faces)
organ_mesh.compute_vertex_normals()
# Visualize the mesh   
o3d.visualization.draw_geometries([organ_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
