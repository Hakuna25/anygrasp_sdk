import os
import argparse
import shutil
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from collections import Counter
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()
    
    # get data
    colors_f = np.array(Image.open(os.path.join(data_dir, 'rgb_1.png')), dtype=np.float32) / 255.0
    depths_f = np.array(Image.open(os.path.join(data_dir, 'depth_1.png')))
    colors_h = np.array(Image.open(os.path.join(data_dir, 'rgb_0.png')), dtype=np.float32) / 255.0
    depths_h = np.array(Image.open(os.path.join(data_dir, 'depth_0.png')))
    # hand camera intrinsics
    fx_h, fy_h = 909.08, 908.376 # (1280, 720)
    cx_h, cy_h = 633.573, 361.852
    fx_f, fy_f = 909.182, 907.714 # (1280, 720)
    cx_f, cy_f = 637.532, 364.387
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.25, 0.25
    ymin, ymax = -0.4, 0.4
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get front camera point cloud
    xmap, ymap = np.arange(depths_f.shape[1]), np.arange(depths_f.shape[0])
    xmap_f, ymap_f = np.meshgrid(xmap, ymap)
    points_z = depths_f / scale
    points_x = (xmap_f - cx_f) / fx_f * points_z
    points_y = (ymap_f - cy_f) / fy_f * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points_f = points[mask].astype(np.float32)
    colors_f = colors_f[mask].astype(np.float32)
    
    fc_base = np.array([[-0.025185470710454363, 0.9003537485256276, -0.43442930331751733, 0.8003658631290567],
                            [0.9990845637502204, 0.007637667199582072, -0.04209157297821219, 0.014761293894194942],
                            [-0.034579279071787865, -0.4350917070636938, -0.8997218903101533, 0.8697237283025128],# -0.02
                            [0.0, 0.0, 0.0, 1.0],], dtype=np.float32)

    # offset
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array([0.015, 0.01, 0.0], dtype=np.float32)
    T_total = T @ fc_base

    num_points = points_f.shape[0]
    points_hom = np.hstack([points_f, np.ones((num_points, 1), dtype=np.float32)])
    points_hom_transformed = (T_total @ points_hom.T).T
    points_f_transformed = points_hom_transformed[:, :3]

    # print(f'Front camera point cloud: min:{points_f.min(axis=0)}, max:{points_f.max(axis=0)}')
    # gg_f, cloud_f = anygrasp.get_grasp(points_f_transformed, colors_f, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
    # o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.io.write_point_cloud("point_cloud_f.pcd", cloud_f)
    
    # get hand camera point cloud
    xmap, ymap = np.arange(depths_h.shape[1]), np.arange(depths_h.shape[0])
    xmap_h, ymap_h = np.meshgrid(xmap, ymap)
    points_z = depths_h / scale
    points_x = (xmap_h - cx_h) / fx_h * points_z
    points_y = (ymap_h - cy_h) / fy_h * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points_h = points[mask].astype(np.float32)
    colors_h = colors_h[mask].astype(np.float32)
    hc_base = np.load('hc2base.npy').astype(np.float32)
    num_points = points_h.shape[0]
    points_hom = np.hstack([points_h, np.ones((num_points, 1), dtype=np.float32)])
    points_hom_transformed = (hc_base @ points_hom.T).T
    points_h_transformed = points_hom_transformed[:, :3]
    print(f'Hand camera point cloud: min:{points_h.min(axis=0)}, max:{points_h.max(axis=0)}')

    merged_points = np.vstack([points_h_transformed, points_f_transformed])
    merged_colors = np.vstack([colors_h, colors_f])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # merged_cloud = merge_pointclouds(cloud_f, cloud_h)
    
    # convert merged_points to hand camera coordinate
    hc_base_inv = np.linalg.inv(hc_base)
    num_points = merged_points.shape[0]
    points_hom = np.hstack([merged_points, np.ones((num_points, 1), dtype=np.float32)])
    points_in_hand_hom = (hc_base_inv @ points_hom.T).T
    points_in_hand = points_in_hand_hom[:, :3]
    np.save('merged_points.npy', points_in_hand)
    gg_m, cloud_m = anygrasp.get_grasp(points_in_hand, merged_colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # hc_base = np.load('hc2base.npy')
    # cloud_h.transform(hc_base)
    if len(gg_m) == 0:
        print('No Grasp detected after collision detection!')
    
    gg_m = gg_m.nms().sort_by_score()
    gg_pick = gg_m[0:10]
    grippers = gg_pick.to_open3d_geometry_list()
        
    gripper_mesh_dir = "gripper_mesh"
    if os.path.exists(gripper_mesh_dir):
        shutil.rmtree(gripper_mesh_dir) 
    os.makedirs(gripper_mesh_dir)
    # save TriangleMesh list
    for i, mesh in enumerate(grippers):
        o3d.io.write_triangle_mesh(f"gripper_mesh/mesh_{i}.ply", mesh) 
    # breakpoint()
    cloud_m.transform(hc_base)
    o3d.io.write_point_cloud("merged_point_cloud.pcd", cloud_m)
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)
    pc_list = []
    for i in range(len(gg_pick)):
        T_gripper = np.eye(4)
        gripper_idx = i
        T_gripper[:3, :3] = gg_pick[gripper_idx].rotation_matrix
        T_gripper[:3, 3] = gg_pick[gripper_idx].translation
        R_gripper = np.array([[ 0, 0, 1, 0 ],
                            [ 0, 1, 0, 0 ],
                            [-1, 0, 0, 0 ],
                            [ 0, 0, 0, 1 ]])
        gripper_pose = T_gripper @ R_gripper
        # check angle condition between gripper and e.e.
        ee = np.array([[-0.01106385, -0.99964285,  0.02432601,0.07624674],
                        [ 0.99967782, -0.01161345, -0.02256943, -0.03747215],
                        [ 0.02284388 , 0.02406847 , 0.99944928, -0.09101364],
                        [ 0.00000000e+00  ,0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
        ee_pose = np.linalg.inv(ee)
        x_o = gripper_pose[:3, 0]
        x_ee = ee_pose[:3, 0]
        cos_theta = np.dot(x_o, x_ee) / np.linalg.norm(x_o) * np.linalg.norm(x_ee)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        Rpi = np.eye(4)
        Rpi[0, 0] = -1
        Rpi[1, 1] = -1
        if theta > np.pi / 2:
            gripper_pose = gripper_pose @ Rpi
        if i == 0:
            np.savez('coor.npz', T_gripper = gripper_pose, hc_base = hc_base)
        partial_cloud, com = get_partial_pc(cloud_m, hc_base, gripper_pose)
        pc = np.asarray(partial_cloud.points).reshape(3, -1)
        if pc.shape[1] < 1024:
            # Calculate how many repeats we need so that the total number of columns is at least 1024
            num_repeats = int(np.ceil(1024 / pc.shape[1]))
            # Duplicate the points and then slice to get exactly 1024 points
            pc_dup = np.tile(pc, num_repeats)[:, :1024]
        else:
            # For the case that the point cloud already has 1024 or more points,
            # you might want to simply take the first 1024 points
            # or use a subsampling scheme.
            pc_dup = pc[:, :1024]
        pc_list.append(pc_dup)
        if i == 0:
            com_return = com.reshape(1, 3)
    # breakpoint()
    R = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    pc = np.array(pc_list)
    
    # breakpoint()
    np.savez('pc_com.npz', com = com_return, pc = pc)
    # if len(gg_f) == 0:
    #     print('No Grasp detected after collision detection!')

    # gg_f = gg_f.nms().sort_by_score()
    # gg_pick = gg_f[0:20]
    # print(gg_pick.scores)
    # print('grasp score:', gg_pick[0].score)

    # # visualization
    # if cfgs.debug:
    #     trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    #     cloud.transform(trans_mat)
    #     grippers = gg.to_open3d_geometry_list()
    #     for gripper in grippers:
    #         gripper.transform(trans_mat)
    #     o3d.visualization.draw_geometries([*grippers, cloud])
    #     o3d.visualization.draw_geometries([grippers[0], cloud])
    #     o3d.io.write_point_cloud("point_cloud.pcd", cloud)  # 保存为 PCD 文件
    
def get_partial_pc(cloud, hc_base, gripper_hc):
    
    # 获取所有点的颜色（浮点数，范围 [0, 1]）
    colors = np.asarray(cloud.colors)
    # 为了便于统计，将颜色乘 255 并转换成整数
    colors_int = (colors * 255).astype(np.uint8)
    # 转换为元组列表（hashable类型）进行计数
    color_tuples = [tuple(color) for color in colors_int]
    # 统计各颜色出现次数，取出现次数最多的作为地面颜色
    color_counter = Counter(color_tuples)
    ground_color_tuple, count = color_counter.most_common(1)[0]
    print("Detected ground color (most common):", ground_color_tuple, "count:", count)
    ground_color_f = np.array(ground_color_tuple, dtype=np.float32) / 255.0

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # base_frame.transform(base_transform)
    # ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])    
    # ee_frame.transform(np.linalg.inv(base_camera) @ ee_camera)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    camera_frame.transform(hc_base)
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    gripper_tf = hc_base @ gripper_hc
    gripper_frame.transform(gripper_tf)
    
    # 提取中心点（平移部分）
    center = gripper_tf[:3, 3]

    # 以 gripper_tf 的 y 和 z 轴确定平面方向
    # 注意：假设 gripper_tf 的前三列已经是正交且归一化的
    y_axis = gripper_tf[:3, 1]
    z_axis = gripper_tf[:3, 2]

    # 确保 y_axis 和 z_axis 归一化
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 计算 x 轴为 y_axis 和 z_axis 的叉积（构成右手坐标系）
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # [x, y, z]
    R = np.column_stack((x_axis, y_axis, z_axis))
    # Bounding box size
    extent = np.array([0.03, 0.15, 0.2])
    full_extent = np.array([0.25, 0.25, 0.25])  
    L = extent[2]
    # Calculate offset (0.5-0.2)
    offset = L * 0.3

    # bounding box offset along z axis
    new_center = center + offset * z_axis
    # Create Oriented Bounding Box
    obb = o3d.geometry.OrientedBoundingBox(new_center, R, extent)
    full_obb = o3d.geometry.OrientedBoundingBox(new_center, R, full_extent)
    # --- 从点云中过滤在包围盒内的部分 ---
    partial_cloud = cloud.crop(obb)
    full_cloud = cloud.crop(full_obb)
    # breakpoint()
    cropped_colors = np.asarray(full_cloud.colors)
    cropped_colors_diff = np.linalg.norm(cropped_colors - ground_color_f, axis=1)
    is_object = cropped_colors_diff >= 0.12
    cropped_points = np.asarray(full_cloud.points)
    object_points = cropped_points[is_object]
    object_colors = cropped_colors[is_object]
    object_cloud = o3d.geometry.PointCloud()
    object_cloud.points = o3d.utility.Vector3dVector(object_points)
    object_cloud.colors = o3d.utility.Vector3dVector(object_colors)
    if len(object_points) == 0:
        print("None object points detected!")
    else:
        com = np.mean(object_points, axis=0)
        # breakpoint()
        print("Object's center of mass:", com)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(com)
        sphere.paint_uniform_color([0, 0, 1])
    indices = obb.get_point_indices_within_bounding_box(cloud.points)
    partial_cloud = cloud.select_by_index(indices)
    partial_cloud = partial_cloud.paint_uniform_color([1, 0, 0])
    # 可视化原始点云与提取的局部点云（用不同颜色显示）
    object_cloud.colors = o3d.utility.Vector3dVector()
    object_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([cloud, partial_cloud, obb, full_obb, sphere, object_cloud])
    # partial_cloud.colors = o3d.utility.Vector3dVector()
    partial_cloud.paint_uniform_color([0, 1, 0])
    # Create a visualizer object
    # o3d.visualization.draw_geometries([cloud, base_frame, camera_frame, gripper_frame, obb, partial_cloud],)
    return partial_cloud, com

def merge_pointclouds(pc1, pc2):
    """
    Merge two Open3D pointcloud objects.
    
    Args:
        pc1 (open3d.geometry.PointCloud): The first pointcloud.
        pc2 (open3d.geometry.PointCloud): The second pointcloud.
        
    Returns:
        open3d.geometry.PointCloud: The merged pointcloud.
    """
    merged_pc = o3d.geometry.PointCloud()
    
    # Merge the point coordinates.
    points1 = np.asarray(pc1.points)
    points2 = np.asarray(pc2.points)
    merged_points = np.vstack((points1, points2))
    merged_pc.points = o3d.utility.Vector3dVector(merged_points)
    
    # If both pointclouds have colors, merge the colors.
    if pc1.has_colors() and pc2.has_colors():
        colors1 = np.asarray(pc1.colors)
        colors2 = np.asarray(pc2.colors)
        merged_colors = np.vstack((colors1, colors2))
        merged_pc.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # If both pointclouds contain normals, merge them.
    if pc1.has_normals() and pc2.has_normals():
        normals1 = np.asarray(pc1.normals)
        normals2 = np.asarray(pc2.normals)
        merged_normals = np.vstack((normals1, normals2))
        merged_pc.normals = o3d.utility.Vector3dVector(merged_normals)
    
    return merged_pc


if __name__ == '__main__':
    
    demo('./example_data/')