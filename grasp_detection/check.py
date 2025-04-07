import open3d as o3d
import os
import numpy as np

if __name__ == "__main__":
    grippers = []
    folder_path = "gripper_mesh/"
    files = os.listdir(folder_path)
    ply_files = [f for f in files if f.endswith(".ply")]

    for i in range(len(ply_files)): 
        mesh = o3d.io.read_triangle_mesh(f"gripper_mesh/mesh_{i}.ply")
        grippers.append(mesh)
    cloud = o3d.io.read_point_cloud("point_cloud.pcd")
    
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # 构造齐次变换矩阵 (4x4)
    T_gripper = np.eye(4)

    T_gripper[:4, :4] = np.array([[-0.08158159 , 0.    ,      0.99666667 , 0.0041005 ],
                                [ 0.      ,   -1.   ,       0.   ,      -0.00163571],
                                [ 0.99666667, 0.     ,     0.08158159 , 0.441881  ],
                                [ 0.      ,    0.    ,      0.       ,   1.        ]]) 
    R_gripper = np.array([
                        [ 0, 0, 1, 0 ],
                        [ 0, 1, 0, 0 ],
                        [-1, 0, 0, 0 ],
                        [ 0, 0, 0, 1 ]
                    ])
    gripper_frame.transform((T_gripper@R_gripper))
    ee = np.array([[-0.01106385, -0.99964285,  0.02432601,0.07624674],
                [ 0.99967782, -0.01161345, -0.02256943, -0.03747215],
                [ 0.02284388 , 0.02406847 , 0.99944928, -0.09101364],
                [ 0.00000000e+00  ,0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
    ee_frame.transform(np.linalg.inv(ee))
    base= np.eye(4)
    base[:3, 3] = [0, 0, 0.5]  # 平移向量
    base[:3, :3] = np.array([[-1.0, 0, 0],
                             [0, 0, -1],
                            [0, -1, 0]])  # 旋转矩阵
    
    base_frame.transform(base)
    # o3d.visualization.draw_geometries([*grippers, cloud, camera_frame, gripper_frame, base_frame, ee_frame])
    o3d.visualization.draw_geometries([grippers[0], cloud, camera_frame, gripper_frame, ee_frame])