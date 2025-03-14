import os
import argparse
import torch
import numpy as np
import open3d as o3d
import shutil
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.05, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    # colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.3, 0.3 #-0.19, 0.12
    ymin, ymax = -0.2, 0.2 #0.02, 0.15
    zmin, zmax = 0.0, 0.5 #0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # # get point cloud
    # xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    # xmap, ymap = np.meshgrid(xmap, ymap)
    # points_z = depths / scale
    # points_x = (xmap - cx) / fx * points_z
    # points_y = (ymap - cy) / fy * points_z

    # # set your workspace to crop point cloud
    # mask = (points_z > 0) & (points_z < 1)
    # points = np.stack([points_x, points_y, points_z], axis=-1)
    
    # points = points[mask].astype(np.float32)
    # colors = colors[mask].astype(np.float32)
    
    points = np.load("warp_points.npy").astype(np.float32)
    colors = np.zeros(points.shape).astype(np.float32)
    breakpoint()
    print(points.min(axis=0), points.max(axis=0))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
    breakpoint()
    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)
    
    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        
        for gripper in grippers:
            gripper.transform(trans_mat)
            
        gripper_mesh_dir = "gripper_mesh"
        if os.path.exists(gripper_mesh_dir):
            shutil.rmtree(gripper_mesh_dir) 
        os.makedirs(gripper_mesh_dir)
        # save TriangleMesh list
        for i, mesh in enumerate(grippers):
            o3d.io.write_triangle_mesh(f"gripper_mesh/mesh_{i}.ply", mesh)  # save to ply file
        breakpoint()
        # save PointCloud
        o3d.io.write_point_cloud("point_cloud.pcd", cloud)  # 保存为 PCD 文件

        # Visualize the geometry
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        # Add the geometries to the visualizer
        for gripper in grippers:
            vis.add_geometry(gripper)
        vis.add_geometry(cloud)

        # Capture and save the image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("output_image.png")

        # Close the visualizer
        vis.destroy_window()
            
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])


if __name__ == '__main__':
    
    demo('./example_data/')
