import open3d as o3d
import SimpleITK as sitk
import numpy as np
from skimage import measure
import nibabel as nib

# ct_image = sitk.ReadImage("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/imagesTs/BDMAP_00000005_0000.nii.gz")
# ct_mask = sitk.ReadImage("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/labelsTs/BDMAP_00000005.nii.gz")
# ct_array = sitk.GetArrayFromImage(ct_image)  # shape: (slices, height, width)
# print("CT image shape:", ct_array.shape)

# mask_array = sitk.GetArrayFromImage(ct_mask)
# print("CT mask shape:", mask_array.shape)

# spacing = ct_image.GetSpacing()  # (x, y, z) -> for scaling
# spacing = spacing[::-1]  # reverse to match numpy array order


# threshold = 40  # HU value for bone/soft tissue (~40)
# verts, faces, _, _ = measure.marching_cubes(ct_array, level=threshold, spacing=spacing)


# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(verts)
# mesh.triangles = o3d.utility.Vector3iVector(faces)

# mesh_pcd = mesh.sample_points_uniformly(number_of_points=10000)  
# mesh_pcd = mesh_pcd.voxel_down_sample(voxel_size=0.0006)  



# # mesh.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh])



# mask = nib.load("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/labelsTs/BDMAP_00000005.nii.gz")
# mask = nib.load("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset335_Lap2Ct-ct/imagesTs/input_ct_001.nii.gz")
mask = nib.load("/mnt/d/projectsD/datasets/LAP2CT/ct/input_ct_001.nii.gz")
# ct = nib.load("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/imagesTs/input_ct_001_0000.nii.gz").get_fdata()
# pcd_ct = o3d.geometry.PointCloud()
# print("ct shape:", ct.shape, type(ct))
# threshold = 40  # Adjust based on your CT values
# coords = np.where(ct > threshold)  # Returns (z, y, x) indices

# # Convert indices to 3D coordinates
# points = np.column_stack((coords[2], coords[1], coords[0]))
# pcd_ct.points = o3d.utility.Vector3dVector(points)  # Reshape to (N, 3) for point cloud

data = mask.get_fdata()
# o3d.visualization.draw_geometries([pcd_ct])
# # To show single organ
# label_no = 20
# mask = (data == label_no).astype(np.uint8) # to show only single organ
# verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)# to show only single organ

# # To show all organs 
# data = np.array(data.astype(np.uint8))  
# verts, faces, _, _ = measure.marching_cubes(data, level=0.5)


# organ_mesh = o3d.geometry.TriangleMesh()
# organ_mesh.vertices = o3d.utility.Vector3dVector(verts)
# organ_mesh.triangles = o3d.utility.Vector3iVector(faces)
# organ_mesh.compute_vertex_normals()
# # Visualize the mesh   
# o3d.visualization.draw_geometries([organ_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)

class_map = {1: 'aorta', 2: 'gall_bladder', 3: 'kidney_left', 4: 'kidney_right', 5: 'liver',
                 6: 'pancreas', 7: 'postcava', 8: 'spleen', 9: 'stomach', 10: 'adrenal_gland_left',
                 11: 'adrenal_gland_right', 12: 'bladder', 13: 'celiac_trunk', 14: 'colon', 15: 'duodenum',
                 16: 'esophagus', 17: 'femur_left', 18: 'femur_right', 19: 'hepatic_vessel', 20: 'intestine',
                 21: 'lung_left', 22: 'lung_right', 23: 'portal_vein_and_splenic_vein',
                 24: 'prostate', 25: 'rectum'}
organ_colors = {
    0: [0.0, 0.0, 0.0],          # 000000 - black
    1: [1.0, 0.882, 0.0],        # FFE100 - yellow
    2: [0.941, 0.627, 1.0],      # F0A0FF - light pink
    3: [0.0, 0.459, 0.863],      # 0075DC - blue
    4: [0.6, 0.247, 0.0],        # 993F00 - brown
    5: [0.298, 0.0, 0.361],      # 4C005C - dark purple
    6: [0.098, 0.098, 0.098],    # 191919 - dark gray
    7: [0.0, 0.361, 0.192],      # 005C31 - dark green
    8: [0.173, 0.812, 0.282],    # 2BCE48 - bright green
    9: [1.0, 0.8, 0.6],          # FFCC99 - light orange
    10: [0.502, 0.502, 0.502],   # 808080 - gray
    11: [0.584, 1.0, 0.710],     # 94FFB5 - light green
    12: [0.561, 0.486, 0.0],     # 8F7C00 - olive
    13: [0.616, 0.8, 0.0],       # 9DCC00 - yellow green
    14: [0.761, 0.0, 0.533],     # C20088 - magenta
    15: [0.0, 0.2, 0.502],       # 003380 - dark blue
    16: [1.0, 0.643, 0.020],     # FFA405 - orange
    17: [1.0, 0.659, 0.733],     # FFA8BB - light pink
    18: [0.259, 0.4, 0.0],       # 426600 - dark green
    19: [1.0, 0.0, 0.063],       # FF0010 - red
    20: [0.369, 0.945, 0.949],   # 5EF1F2 - cyan
    21: [0.0, 0.6, 0.561],       # 00998F - teal
    22: [0.878, 1.0, 0.4],       # E0FF66 - light yellow
    23: [0.459, 0.039, 1.0],     # 740AFF - purple
    24: [0.6, 0.0, 0.0],         # 990000 - dark red
    25: [1.0, 1.0, 0.502],       # FFFF80 - light yellow
}

organ_collection = {}
organ_bool = []
for label, organ in class_map.items():
    organ_mask = (data == label).astype(np.uint8)  # Create a mask for the specific organ
        # Check if organ exists in the data
    if np.sum(organ_mask) == 0:
        print(f"Skipping {organ} (label {label}) - not present in this scan")
        organ_bool.append(False)
        organ_collection[label] = None
        continue
    verts, faces, _, _ = measure.marching_cubes(organ_mask, level=0.5)  # Extract the mesh for the organ

    organ_mesh = o3d.geometry.TriangleMesh()
    organ_mesh.vertices = o3d.utility.Vector3dVector(verts)
    organ_mesh.triangles = o3d.utility.Vector3iVector(faces)
    organ_mesh.compute_vertex_normals()
    organ_mesh.paint_uniform_color(organ_colors[label])  # Assign a random color to each organ
    organ_collection[label] = organ_mesh  # Store the organ mesh
    organ_bool.append(True)
    print(f"Organ: {organ}, label: {label}")
print("Organ collection keys:", organ_collection)
print("Organ bool:", organ_bool)
# Visualize each organ mesh
o3d.visualization.draw_geometries([organ for label,organ in organ_collection.items() if organ is not None],)


# # save liver point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector((data == 5).astype(np.uint8))  # Example for liver
# o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

# save liver mesh
# o3d.io.write_triangle_mesh("/mnt/d/projectsD/datasets/CTL-REG/input/01/model/test_liver.obj", organ_collection[5])  # Save the liver mesh as an example