import numpy as np
import open3d as o3d
import cv2 

def depth_to_point_cloud(depth_image, intrinsic, extrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    height, width = depth_image.shape
    points = []

    for v in range(height):
        for u in range(width):
            Z_c = depth_image[v, u]
            if Z_c > 0:  # Ignore zero depth values
                x = (u - cx) * Z_c / fx
                y = (v - cy) * Z_c / fy
                # Note: The y coordinate may need to be negated if using a different convention
                point_camera = np.array([x, y, Z_c, 1.0])
                point_world = (extrinsic @ point_camera)
                points.append(point_world[:3])
    return points

def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# Example usage
if __name__ == "__main__":
    path = "/scratch2/yuxili/BlenderProc/output_eval/00004f89-9aa5-43c2-ae3c-129586be8aaa/"
    # Example depth image (replace with actual depth image data)
    rgb = cv2.imread(path + "room_empty.png")
    rgb_image = o3d.geometry.Image(rgb)
    depth = np.load(path + "room_depth.npy")
    depth_image = o3d.geometry.Image(depth)

    # Camera intrinsic matrix (example values)
    intrinsic = np.loadtxt(path + "camera_intrinsic_matrix.txt")
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_matrix.set_intrinsics(
        width=depth.shape[0],
        height=depth.shape[0],
        fx = intrinsic[0, 0],
        fy = intrinsic[1, 1],
        cx = intrinsic[0, 2],
        cy = intrinsic[1, 2],
    )

    # Camera extrinsic matrix (example values, identity matrix here)
    # extrinsic = np.loadtxt(path + "cam2world_matrix.txt")
    # extrinsic[:3, :3] = 0
    # np.fill_diagonal(extrinsic[:3, :3], 1)
    # extrinsic[3, 3] = 1
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_scale=1, depth_trunc=6)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_matrix, np.eye(4))


    # # Create Open3D point cloud
    # point_cloud = create_point_cloud(points)
    o3d.io.write_point_cloud(path + "room.ply", pcd)