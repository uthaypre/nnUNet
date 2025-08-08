import open3d as o3d
import SimpleITK as sitk
import numpy as np
from skimage import measure
import cv2
import nibabel as nib
# https://www.open3d.org/html/tutorial/Advanced/surface_reconstruction.html
endo_image = cv2.imread("/mnt/d/projectsD/datasets/depth_val_test/image_21534_0000.png")
endo_mask = cv2.imread("/mnt/d/projectsD/datasets/depth_val_test/image_21534.png")
depth_image = cv2.imread("/mnt/d/projectsD/datasets/depth_val_test/output/image_21534_0000_depth.png")
endo_mask = cv2.cvtColor(endo_mask, cv2.COLOR_BGR2GRAY)
# print(np.unique(endo_mask))
labels = {
	"background": 0,
        "abdominal_wall": 1,
        "colon": 2,
        "liver": 3,
        "pancreas": 4,
        "small_intestine": 5,
        "spleen": 6,
        "stomach": 7
    }
organ_colors = {
    0: [0.0, 0.0, 0.0],          # 000000 - black
    1: [1.0, 0.314, 0.020],     # FF5005 - red orange - abdominal_wall
    2: [0.761, 0.0, 0.533],     # C20088 - magenta - colon
    3: [0.298, 0.0, 0.361],      # 4C005C - dark purple - liver
    4: [0.098, 0.098, 0.098],    # 191919 - dark gray - pancreas
    5: [0.369, 0.945, 0.949],   # 5EF1F2 - cyan - small_intestine
    6: [0.173, 0.812, 0.282],    # 2BCE48 - bright green -spleen
    7: [1.0, 0.8, 0.6],          # FFCC99 - light orange - stomach
}
colored_mask = np.zeros((endo_mask.shape[0], endo_mask.shape[1], 3))
for label, color in organ_colors.items():
    if label == 0:
        continue  # Skip background
    colored_mask[endo_mask == label] = color  # Create a binary mask for the organ

colored_mask = (colored_mask * 255).astype(np.uint8)  # Convert to uint8 for visualization
print("endo_image shape:", endo_image.shape)
print("endo_mask shape:", endo_mask.shape)
print("depth_image shape:", depth_image.shape)

depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
print("depth_gray shape:", depth_gray.shape)
print("depth_gray dtype:", type(depth_gray))
depth_float = depth_gray.astype(np.float32)
def get_intrinsics(H,W, fov = 55.0):
    f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov))
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

depth_3d = o3d.geometry.PointCloud.create_from_depth_image(
    depth=o3d.geometry.Image(depth_float),
    intrinsic=o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=0.5 * depth_image.shape[1] / np.tan(0.5 * np.deg2rad(55.0)),
        fy=0.5 * depth_image.shape[0] / np.tan(0.5 * np.deg2rad(55.0)),
        cx=depth_image.shape[1] / 2.0,
        cy=depth_image.shape[0] / 2.0))



# mesh = o3d.geometry.TetraMesh.create_from_point_cloud(depth_3d)
# o3d.visualization.draw_geometries([depth_3d])

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=o3d.geometry.Image(colored_mask),
    depth=o3d.geometry.Image(depth_float),
    depth_trunc=1000.0,
    convert_rgb_to_intensity=False)
seg_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                    intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    width=depth_image.shape[1],
                    height=depth_image.shape[0],
                    fx=0.5 * depth_image.shape[1] / np.tan(0.5 * np.deg2rad(55.0)),
                    fy=0.5 * depth_image.shape[0] / np.tan(0.5 * np.deg2rad(55.0)),
                    cx=depth_image.shape[1] / 2.0,
                    cy=depth_image.shape[0] / 2.0))
# o3d.visualization.draw_geometries([seg_point_cloud])



# Filter out each organ point cloud
organ_pc_collection = []
for organ in labels.keys():
    organ_mask = (endo_mask == labels[organ]).astype(np.uint8)  # Create a mask for the specific organ
    # print("where id this",organ_mask.shape)
    single_colored_mask = np.zeros((organ_mask.shape[0], organ_mask.shape[1], 3))
    single_colored_mask[organ_mask == 1] = organ_colors[labels[organ]]  # Assign color to the organ mask
    single_colored_mask = (single_colored_mask * 255).astype(np.uint8)
    organ_depth = depth_float * organ_mask  # Apply the organ mask to the depth image
    single_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color=o3d.geometry.Image(single_colored_mask),
                    depth=o3d.geometry.Image(organ_depth),
                    depth_trunc=1000.0,
                    convert_rgb_to_intensity=False)
    single_seg_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(single_rgbd_image,
                    intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    width=depth_image.shape[1],
                    height=depth_image.shape[0],
                    fx=0.5 * depth_image.shape[1] / np.tan(0.5 * np.deg2rad(55.0)),
                    fy=0.5 * depth_image.shape[0] / np.tan(0.5 * np.deg2rad(55.0)),
                    cx=depth_image.shape[1] / 2.0,
                    cy=depth_image.shape[0] / 2.0))

    # Add the organ point cloud to the collection
    organ_pc_collection.append(single_seg_point_cloud)

# Visualize the point clouds for each organ
o3d.visualization.draw_geometries([organ_pc_collection[0]], mesh_show_back_face=True)