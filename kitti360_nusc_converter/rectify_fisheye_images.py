import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import hashlib

# --- Calibration and rectification parameters (reuse your values) ---
width, height = 1408, 376
HFOV = np.radians(103.75)  # Desired horizontal FOV
f_x = width / (2 * np.tan(HFOV / 2))
f_y = f_x
K_pinhole = np.array([[f_x, 0, width / 2],
                      [0, f_y, height / 2],
                      [0, 0, 1]])
K_pinhole[0,2] = 682.049453
K_pinhole[1,2] = 158.769549

K_right = np.array([[1.4854388981875156e+03, 0, 6.9888316784030962e+02],
            [0, 1.4849477411748708e+03, 6.9814541887723055e+02],
            [0, 0, 1]], dtype=np.float64)
D_right = np.array([4.9370396274089505e-02, 4.5068455478645308e+00, 1.3477698472982495e-03, -7.0340482615055284e-04], dtype=np.float64)
xi_right = np.array([2.5535139132482758], dtype=np.float64)
K_left = np.array([[1.3363220825849971e+03, 0, 7.1694323510126321e+02],
            [0, 1.3357883350012958e+03, 7.0576498308221585e+02],
            [0, 0, 1]], dtype=np.float64)
D_left = np.array([1.6798235660113681e-02, 1.6548773243373522e+00, 4.2223943394772046e-04,4.2462134260997584e-04], dtype=np.float64)
xi_left = np.array([2.2134047507854890], dtype=np.float64)

# --- Rotation matrices for each direction ---
def cal_rot_from_angle(theta_y, theta_x=4):
    theta_rad = np.radians(theta_y)
    R = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])
    angle = np.radians(theta_x)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return Rx @ R

R_front = cal_rot_from_angle(30)
R_back = cal_rot_from_angle(-46)
R_front_left = cal_rot_from_angle(-30)
R_back_left = cal_rot_from_angle(46)


# --- Precompute rectification maps for each camera/direction ---
rectify_maps = {}
for cam, K, D, xi, R_front_this, R_back_this in [
    ("left", K_left, D_left, xi_left, R_front_left, R_back_left),
    ("right", K_right, D_right, xi_right, R_front, R_back)
]:
    # Front
    key_front = f"{cam}_front"
    rectify_maps[key_front] = cv2.omnidir.initUndistortRectifyMap(
        K, D, xi, R_front_this, K_pinhole, (width, height), cv2.CV_32FC1, flags=cv2.omnidir.RECTIFY_PERSPECTIVE)
    # Back
    key_back = f"{cam}_back"
    rectify_maps[key_back] = cv2.omnidir.initUndistortRectifyMap(
        K, D, xi, R_back_this, K_pinhole, (width, height), cv2.CV_32FC1, flags=cv2.omnidir.RECTIFY_PERSPECTIVE)

def rectify_and_save(image, map1, map2, out_path):
    rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, rectified)

# --- Main processing loop ---
json_file_path = "./kitti360/v1.0-trainval/sample_data.json"  # JSON path
source_base_path = "./kitti360/"  # original data root
output_base_dir = "./kitti360/"  # new output root

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Only process annotated frames with image_02 or image_03 and is_key_frame true
filtered_entries = [entry for entry in data if ("image_02" in entry["filename"] or "image_03" in entry["filename"]) and entry.get("is_key_frame", False)]

for entry in tqdm(filtered_entries, desc="Rectifying Annotated Frames", unit="file"):
    filename = entry["filename"]
    source_file = os.path.join(source_base_path, filename)
    if not os.path.exists(source_file):
        print(f"Warning: Source file not found: {source_file}")
        continue

    # Determine left/right by filename
    if "image_02" in filename:
        cam = "left"
    elif "image_03" in filename:
        cam = "right"
    else:
        continue

    # Read image
    image = cv2.imread(source_file)
    if image is None:
        print(f"Warning: Could not read image: {source_file}")
        continue

    # Output dirs
    out_dirs = {
        f"{cam}_front": os.path.join(output_base_dir, os.path.dirname(filename).replace("data_rgb", "data_rect_front")),
        f"{cam}_back": os.path.join(output_base_dir, os.path.dirname(filename).replace("data_rgb", "data_rect_back")),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Output filenames
    file_basename = os.path.basename(filename)
    out_path_front = os.path.join(out_dirs[f"{cam}_front"], file_basename)
    out_path_back = os.path.join(out_dirs[f"{cam}_back"], file_basename)

    # Rectify and save using precomputed maps
    map1_front, map2_front = rectify_maps[f"{cam}_front"]
    map1_back, map2_back = rectify_maps[f"{cam}_back"]
    rectify_and_save(image, map1_front, map2_front, out_path_front)
    rectify_and_save(image, map1_back, map2_back, out_path_back)

print("\nAll annotated frames rectified and saved as left_front, left_back, right_front, right_back!")

def new_token(data):
    s = f"{data['filename']}_{data['calibrated_sensor_token']}_{data['timestamp']}"
    return hashlib.shake_256(s.encode()).hexdigest(16)

# rectified sensor tokens for left/right, front/back
rectified_sensor_tokens = {
    "image_02_front": "5c53332b6666659e2a8c6774c6f110a3",
    "image_02_back": "3956d6f127c626b2ba1f3b17ebe2b45e",
    "image_03_front": "d536d22df084625e0869b8dc65488028",
    "image_03_back": "9cb30607391282f1e7b632418470d473"
}
rectified_height, rectified_width = 376, 1408

# --- Generate new sample_data items for rectified images ---
sample_data_path = './kitti360/v1.0-trainval/sample_data.json'
with open(sample_data_path) as f:
    sample_data = json.load(f)

orig_items = [item for item in sample_data if item.get("is_key_frame") and (
    "image_02/data_rgb" in item["filename"] or "image_03/data_rgb" in item["filename"])]

new_items = []
for direction in ["front", "back"]:
    group = []
    for item in orig_items:
        new_item = item.copy()
        cam = "image_02" if "image_02" in item["filename"] else "image_03"
        new_item["calibrated_sensor_token"] = rectified_sensor_tokens[f"{cam}_{direction}"]
        new_item["height"] = rectified_height
        new_item["width"] = rectified_width
        new_item["filename"] = item["filename"].replace("data_rgb", f"data_rect_{direction}")
        group.append(new_item)
    # Sort by timestamp before relinking
    old_token_to_new_token = {item["token"]: new_token(item) for item in group}
    # Update token and prev/next links for each item
    for item in group:
        item["token"] = old_token_to_new_token[item["token"]]
        
        # Update prev link if exists
        if item.get("prev"):
            if item["prev"] in old_token_to_new_token:
                item["prev"] = old_token_to_new_token[item["prev"]]
            else:
                item["prev"] = ""
                
        # Update next link if exists
        if item.get("next"):
            if item["next"] in old_token_to_new_token:
                item["next"] = old_token_to_new_token[item["next"]]
            else:
                item["next"] = ""
    
    # Add the relinked items to new_items
    new_items.extend(group)
    

with open('./kitti360/v1.0-trainval/sample_data_with_rect.json', 'w') as f:
    json.dump(sample_data + new_items, f, indent=4)
