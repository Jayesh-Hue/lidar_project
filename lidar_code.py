import os
import cv2
import json
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from scipy.spatial import Delaunay

# ==== FUNCTION DEFINITIONS ====

def load_lidar_points(lidar_path):
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    xyz_lidar = points[:, :3]
    return np.hstack((xyz_lidar, np.ones((xyz_lidar.shape[0], 1))))

def transform_lidar_to_camera(xyz_hom, T_velo_cam):
    xyz_cam = (T_velo_cam @ xyz_hom.T).T
    return xyz_cam[xyz_cam[:, 2] > 0]  # Keep points in front

def project_to_image_plane(xyz_cam, P_rect):
    uv_hom = (P_rect @ xyz_cam.T).T
    uv = uv_hom[:, :2] / uv_hom[:, 2, np.newaxis]
    return uv, xyz_cam[:, 2]

def normalize_depth(depth):
    depth_min, depth_max = np.percentile(depth, [5, 95])
    depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
    return (depth_norm * 255).astype(np.uint8)

def draw_colored_points(image, uv, colors):
    overlay = image.copy()
    for i, (u, v) in enumerate(uv.astype(int)):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(overlay, (u, v), 2, tuple(int(c) for c in colors[i]), -1)
    return cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

def create_3d_bbox(corners, color):
    lines = [[0,2],[2,7],[7,5],[5,0],[0,1],[2,3],[6,7],[5,4],[1,3],[3,6],[6,4],[4,1]]
    bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    bbox.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return bbox

def is_inside_box(points, box_corners):
    hull = Delaunay(box_corners)
    return hull.find_simplex(points) >= 0

# ==== SETUP ====
os.environ['KITTI360_PATH'] = r'C:\Sumitsai\RWU\Lidar\lidar_project\KITTI-360_sample'
kitti360Path = os.environ.get('KITTI360_DATASET', os.path.join(os.path.dirname(os.path.realpath(_file_)), '..', '..'))

seq, cam_id, frame = 0, 0, 250
sequence = f'2013_05_28_drive_{seq:04d}_sync'

image_path = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}', 'data_rect', f'{frame:010d}.png')
lidar_path = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data', f'{frame:010d}.bin')
bbox_path = os.path.join(kitti360Path, 'bboxes_3d_cam0', f'BBoxes_{frame}.json')
output_dir = os.path.join("results", 'pointclouds', f"scene_{frame}")
os.makedirs(output_dir, exist_ok=True)

with open(bbox_path, 'r') as f:
    data = json.load(f)

# ==== CALIBRATION ====
P_rect = np.array([[552.554261, 0, 682.049453, 0], [0, 552.554261, 238.769549, 0], [0, 0, 1, 0]])
T_cam_velo = np.array([[0.04307, -0.08829, 0.99516, 0.80439], [-0.99900, 0.00778, 0.04392, 0.29934], [-0.01162, -0.99606, -0.08786, -0.17702], [0, 0, 0, 1]])
T_velo_cam = np.linalg.inv(T_cam_velo)

# ==== PROCESSING ====
xyz_hom = load_lidar_points(lidar_path)
xyz_cam = transform_lidar_to_camera(xyz_hom, T_velo_cam)
uv, depth = project_to_image_plane(xyz_cam, P_rect)
depth_colormap = normalize_depth(depth)
colors = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET).reshape(-1, 3)

image = cv2.imread(image_path)
blended = draw_colored_points(image, uv, colors)
cv2.imshow("Depth Colored LiDAR Projection", blended)
cv2.imwrite("projected_lidar_overlay.png", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== YOLO SEGMENTATION ====
model = YOLO("yolo11x-seg.pt")
results = model(image, conf=0.5)[0]

car_lidar_points = {}
uv_int = uv.astype(int)

# Define color dictionary: keys are car IDs, values are tuples (BGR color, color name)
seg_colors = {
    0: ((0, 0, 255), "Red"),          # Red
    1: ((255, 0, 0), "Blue"),         # Blue
    2: ((0, 255, 0), "Green"),        # Green
    3: ((0, 255, 255), "Yellow"),     # Yellow (BGR)
    4: ((0, 165, 255), "Orange"),     # Orange
    5: ((255, 0, 255), "Purple"),     # Purple
    6: ((203, 192, 255), "Pink"),     # Pink
    7: ((0, 0, 0), "Black"),          # Black
    8: ((128, 128, 128), "Gray"),     # Gray
    9: ((42, 42, 165), "Brown"),      # Brown
}

car_id = 0
for i, mask in enumerate(results.masks.data):
    if model.names[int(results.boxes.cls[i])] != "car":
        continue

    mask = mask.cpu().numpy()
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bin = cv2.erode((mask_resized > 0.7).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2).astype(bool)

    car_points = [xyz_cam[j] for j, (u, v) in enumerate(uv_int) if 0 <= u < image.shape[1] and 0 <= v < image.shape[0] and mask_bin[v, u]]
    car_lidar_points[car_id] = np.array(car_points)
    car_id += 1

# ==== VISUALIZE SEGMENTED CARS ====
overlay = image.copy()
for car_id, pts in car_lidar_points.items():
    color_bgr = seg_colors.get(car_id, ((255,255,255), "Unknown"))[0]
    for pt in pts:
        u, v, _ = (P_rect @ np.append(pt[:3], 1.0)) / pt[2]
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(overlay, (int(u), int(v)), 2, color_bgr, -1)
cv2.imshow("LiDAR Points per Car", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== CREATE POINT CLOUDS ====
pcd_list = []
for car_id, pts in car_lidar_points.items():
    if pts.size == 0:
        continue
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # Convert BGR to RGB and normalize to [0,1]
    bgr = seg_colors.get(car_id, ((1, 1, 1), "Unknown"))[0]
    rgb = [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255]
    pcd.paint_uniform_color(rgb)
    # o3d.io.write_point_cloud(os.path.join(output_dir, f"car_{car_id:02d}.ply"), pcd)
    pcd_list.append(pcd)

# ==== VISUALIZE 3D BBOXES ====
geometries = [create_3d_bbox(np.array(obj['corners_cam0']), color=[1, 0, 0]) for obj in data]
o3d.visualization.draw_geometries(pcd_list + [o3d.geometry.TriangleMesh.create_coordinate_frame()] + geometries)

# ==== EVALUATE CARS INSIDE BOXES ====
results_eval = {}
for car_id, pts in car_lidar_points.items():
    best_box, max_inliers, inside_mask = None, 0, None
    for box in data:
        in_box_mask = is_inside_box(pts[:, :3], np.array(box["corners_cam0"]))
        if np.sum(in_box_mask) > max_inliers:
            best_box, max_inliers, inside_mask = box, np.sum(in_box_mask), in_box_mask
    if best_box is not None:
        results_eval[car_id] = {
            "box_index": best_box["index"],
            "points_inside": int(np.sum(inside_mask)),
            "points_outside": int(np.sum(~inside_mask)),
        }

for car_id, res in results_eval.items():
    color_name = seg_colors.get(car_id, ((0, 0, 0), "Unknown"))[1]
    print(f"Car {car_id} ({color_name}):")
    print(f"  ↳ Best GT Box Index: {res['box_index']}")
    print(f"  ↳ Points Inside:     {res['points_inside']}")
    print(f"  ↳ Points Outside:    {res['points_outside']}")

# ==== FINAL VISUALIZATION ====

# Step 1: Add original (full) point cloud
original_pcd = o3d.geometry.PointCloud()
original_pcd.points = o3d.utility.Vector3dVector(xyz_cam[:, :3])
original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
vis_geometries = [original_pcd]  # Start with full point cloud

# Step 2: Add segmented cars and bounding boxes
for car_id, res in results_eval.items():
    bgr = seg_colors.get(car_id, ((1, 1, 1), "Unknown"))[0]
    rgb = [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(car_lidar_points[car_id][:, :3])
    pcd.paint_uniform_color(rgb)
    vis_geometries.append(pcd)

    corners = np.array(next(b for b in data if b["index"] == res["box_index"])["corners_cam0"])
    vis_geometries.append(create_3d_bbox(corners, color=rgb))

# Optional: Add coordinate frame
vis_geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame())

# Visualize all
o3d.visualization.draw_geometries(vis_geometries)