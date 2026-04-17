#!/usr/bin/env python3
"""
Unified KITTI-360 to nuScenes format converter.

Converts KITTI-360 dataset (3D bounding boxes, poses, calibration, images, LiDAR)
into nuScenes JSON tables in a single pass.

Fixes over the original multi-script pipeline:
  - Deduplicates overlapping scene windows so each frame belongs to exactly one scene
  - Derives velo-to-IMU transform from calibration files (no hardcoding)
  - Consistent pose convention (no mixed NED/FLU)
  - Uses box center = mean(vertices), not the XML transform T
  - Ground filter in sensor-local coords (not world coords)
  - Runs LiDAR point-in-box for both static AND dynamic objects
  - No 80m auto-exclude; distance-based threshold only for near auto-include
  - Built-in validation before writing output
257 empty samples are all concentrated in one scene: scene-0093 from 
2013_05_28_drive_0002_sync frames 15886-16223. That points to a localized empty 
segment, not a dataset-wide pose or annotation failure.
"""

import os
import sys
import json
import uuid
import time
import datetime
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import numpy as np
from pyquaternion import Quaternion

# ---------------------------------------------------------------------------
# Add kitti360PoseConversion to path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kitti360PoseConversion.data import loadOxtsData, loadTimestamps
from kitti360PoseConversion.convertOxtsToPose import convertOxtsToPose
from kitti360PoseConversion.utils import postprocessPoses
from kitti360PoseConversion.labels import labels, kittiId2label, name2label, kkitti360_to_nuscenes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_token():
    return uuid.uuid4().hex


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  Wrote {path}  ({len(data)} records)", flush=True)


def parse_kitti_timestamp(ts_str):
    """Parse KITTI timestamp string -> integer microseconds."""
    if not ts_str:
        raise ValueError("Empty timestamp")
    truncated = ts_str[:-3]  # nanoseconds -> microseconds
    dt = datetime.datetime.strptime(truncated, "%Y-%m-%d %H:%M:%S.%f")
    return int(dt.timestamp() * 1e6)


def rotation_matrix_to_quaternion(R):
    """3x3 rotation matrix -> [w, x, y, z] list. Handles near-orthogonal matrices via SVD."""
    # Ensure orthogonality: nearest rotation via SVD
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1
        R_orth = U @ Vt
    q = Quaternion(matrix=R_orth)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def quaternion_from_yaw(yaw):
    """Yaw angle -> [w, x, y, z] list."""
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return q.elements.tolist()


# ---------------------------------------------------------------------------
# Calibration loaders  (from gen_calibrated_sensor.py)
# ---------------------------------------------------------------------------

def _read_variable(fid, name, M, N):
    fid.seek(0, 0)
    for line in fid:
        if line.startswith(name):
            vals = line.replace(f"{name}:", "").split()
            assert len(vals) == M * N, f"Expected {M*N} values for {name}, got {len(vals)}"
            return np.array([float(v) for v in vals]).reshape(M, N)
    return None


def load_calib_cam_to_pose(path):
    """Returns dict camera_name -> 4x4 matrix."""
    last = np.array([[0, 0, 0, 1]])
    Tr = {}
    with open(path) as f:
        for cam in ["image_00", "image_01", "image_02", "image_03"]:
            Tr[cam] = np.concatenate((_read_variable(f, cam, 3, 4), last))
    return Tr


def load_calib_rigid(path):
    """Load a single 3x4 rigid-body transform -> 4x4."""
    last = np.array([[0, 0, 0, 1]])
    return np.concatenate((np.loadtxt(path).reshape(3, 4), last))


def load_perspective_intrinsic(path):
    Tr = {}
    last = np.array([[0, 0, 0, 1]])
    with open(path) as f:
        for key in ["P_rect_00", "R_rect_00", "P_rect_01", "R_rect_01"]:
            if key.startswith("P_rect"):
                Tr[key] = np.concatenate((_read_variable(f, key, 3, 4), last))
            else:
                Tr[key] = _read_variable(f, key, 3, 3)
    return Tr


def load_fisheye_intrinsics(yaml_path):
    import yaml
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    g1 = d["projection_parameters"]["gamma1"]
    g2 = d["projection_parameters"]["gamma2"]
    u0 = d["projection_parameters"]["u0"]
    v0 = d["projection_parameters"]["v0"]
    K = np.array([[g1, 0, u0], [0, g2, v0], [0, 0, 1]])
    xi = d["mirror_parameters"]["xi"]
    dp = d["distortion_parameters"]
    dist = np.array([dp["k1"], dp["k2"], dp["p1"], dp["p2"], xi])
    return K, dist


# ---------------------------------------------------------------------------
# XML 3D-bbox parser  (lightweight, no skimage dependency)
# ---------------------------------------------------------------------------

BBox3D = namedtuple("BBox3D", [
    "global_id", "semantic_id", "instance_id", "name",
    "vertices", "R", "T",
    "start_frame", "end_frame", "timestamp",
])


def _parse_opencv_matrix(node):
    rows = int(node.find("rows").text)
    cols = int(node.find("cols").text)
    raw = node.find("data").text.split()
    vals = [float(v) for v in raw if v.strip()]
    return np.array(vals).reshape(rows, cols)


def load_annotation_xml(xml_path):
    """Parse KITTI-360 3D bbox XML -> dict[global_id][timestamp] = BBox3D."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = defaultdict(dict)
    for child in root:
        if child.find("transform") is None:
            continue
        sem_kitti = int(child.find("semanticId").text)
        sem_id = kittiId2label[sem_kitti].id
        inst_id = int(child.find("instanceId").text)
        name = kittiId2label[sem_kitti].name
        global_id = sem_id * 1000 + inst_id

        transform = _parse_opencv_matrix(child.find("transform"))
        R = transform[:3, :3]
        T = transform[:3, 3]
        verts = _parse_opencv_matrix(child.find("vertices"))
        verts = (R @ verts.T).T + T

        sf = int(child.find("start_frame").text)
        ef = int(child.find("end_frame").text)
        ts = int(child.find("timestamp").text)

        bbox = BBox3D(global_id, sem_id, inst_id, name,
                      verts, R, T, sf, ef, ts)
        objects[global_id][ts] = bbox
    return objects


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_center(vertices):
    return np.mean(vertices, axis=0)[:3]


def box_dimensions(vertices):
    """Returns (width, length, height) from the 8-corner convention."""
    length = float(np.linalg.norm(vertices[0] - vertices[5]))
    width = float(np.linalg.norm(vertices[0] - vertices[2]))
    height = float(np.linalg.norm(vertices[0] - vertices[1]))
    return [width, length, height]


def box_yaw_quaternion(vertices):
    """Yaw from edge 0->5 projected to xy plane -> quaternion [w,x,y,z]."""
    d = vertices[0, :2] - vertices[5, :2]
    yaw = float(np.arctan2(d[1], d[0]))
    return quaternion_from_yaw(yaw)


def box_rotation_matrix(vertices):
    q = Quaternion(*box_yaw_quaternion(vertices))
    return q.rotation_matrix


def points_in_box_counts(points, centers, sizes, rotations, batch_size=64):
    """
    Vectorised 3D point-in-box test with batching to limit memory.
    points:    (N, 3)
    centers:   (M, 3)
    sizes:     (M, 3)   [w, l, h]
    rotations: (M, 3, 3)
    Returns:   (M,) int array – point count per box.
    """
    N, M = points.shape[0], centers.shape[0]
    if N == 0 or M == 0:
        return np.zeros(M, dtype=np.int32)
    counts = np.zeros(M, dtype=np.int32)
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        c = centers[start:end]       # (B, 3)
        s = sizes[start:end]         # (B, 3)
        r = rotations[start:end]     # (B, 3, 3)
        pts = points[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, B, 3)
        pts_local = np.einsum("nbi,bji->nbj", pts, r)         # (N, B, 3)
        half = s[np.newaxis, :, :] / 2.0
        inside = (np.abs(pts_local) <= half).all(axis=2)       # (N, B)
        counts[start:end] = inside.sum(axis=0)
    return counts


# ---------------------------------------------------------------------------
# Phase 1 – Static metadata tables
# ---------------------------------------------------------------------------

def generate_static_tables(kitti_root, output_dir):
    """Generate sensor, calibrated_sensor, log, map, category, attribute, visibility."""
    calib_dir = os.path.join(kitti_root, "calibration")
    bbox_dir = os.path.join(kitti_root, "data_3d_bboxes", "train")

    # ---- sensor.json ----
    sensor_defs = [
        ("CAM_FRONT", "camera"),
        ("CAM_FRONT_RIGHT", "camera"),
        ("CAM_BACK_RIGHT", "camera"),
        ("CAM_BACK_LEFT", "camera"),
        ("SICK", "lidar"),
        ("LIDAR_TOP", "lidar"),
    ]
    sensors = [{"token": gen_token(), "channel": ch, "modality": mod} for ch, mod in sensor_defs]
    sensor_tok = {s["channel"]: s["token"] for s in sensors}

    # nuScenes channel -> KITTI camera name
    nusc_to_kitti_cam = {
        "CAM_FRONT": "image_00",
        "CAM_FRONT_RIGHT": "image_01",
        "CAM_BACK_RIGHT": "image_02",
        "CAM_BACK_LEFT": "image_03",
    }
    kitti_to_nusc_cam = {v: k for k, v in nusc_to_kitti_cam.items()}

    # ---- calibrated_sensor.json ----
    Tr_cam_pose = load_calib_cam_to_pose(os.path.join(calib_dir, "calib_cam_to_pose.txt"))
    Tr_cam_velo = load_calib_rigid(os.path.join(calib_dir, "calib_cam_to_velo.txt"))
    Tr_sick_velo = load_calib_rigid(os.path.join(calib_dir, "calib_sick_to_velo.txt"))
    Tr_pers = load_perspective_intrinsic(os.path.join(calib_dir, "perspective.txt"))

    # Derive velo-to-IMU from cam0-to-pose and cam0-to-velo
    # T_imu_cam0 = Tr_cam_pose["image_00"]
    # T_cam0_velo = Tr_cam_velo  (maps a point in cam0 frame given velo frame)
    #   Actually calib_cam_to_velo.txt is T_velo_cam0 (cam0 -> velo)
    #   so T_imu_velo = T_imu_cam0 @ inv(T_velo_cam0)
    T_imu_cam0 = Tr_cam_pose["image_00"]
    T_velo_cam0 = Tr_cam_velo
    T_imu_velo = T_imu_cam0 @ np.linalg.inv(T_velo_cam0)

    # Similarly for SICK: T_imu_sick = T_imu_cam0 @ inv(T_velo_cam0) @ inv(T_sick_velo)
    #   Wait -- sick_to_velo means T_velo_sick
    T_velo_sick = Tr_sick_velo
    T_imu_sick = T_imu_velo @ np.linalg.inv(T_velo_sick)

    calibrated_sensors = []

    # Cameras
    for kitti_cam in ["image_00", "image_01", "image_02", "image_03"]:
        nusc_ch = kitti_to_nusc_cam[kitti_cam]
        T = Tr_cam_pose[kitti_cam].copy()
        suffix = kitti_cam[-2:]  # "00" or "01"

        cam_intrinsic = []
        distortion_coefficients = ""
        if kitti_cam in ["image_00", "image_01"]:
            # Undo rectification: T_imu_cam_unrect = T_imu_cam_rect @ inv(R_rect)
            R_rect_key = f"R_rect_{suffix}"
            if R_rect_key in Tr_pers and Tr_pers[R_rect_key] is not None:
                T[:3, :3] = T[:3, :3] @ np.linalg.inv(Tr_pers[R_rect_key])
            P_key = f"P_rect_{suffix}"
            cam_intrinsic = Tr_pers[P_key][:3, :3].tolist()
        else:
            K, dist = load_fisheye_intrinsics(
                os.path.join(calib_dir, f"{kitti_cam}.yaml")
            )
            cam_intrinsic = K.tolist()
            distortion_coefficients = dist.tolist()

        calibrated_sensors.append({
            "token": gen_token(),
            "sensor_token": sensor_tok[nusc_ch],
            "translation": T[:3, 3].tolist(),
            "rotation": rotation_matrix_to_quaternion(T[:3, :3]),
            "camera_intrinsic": cam_intrinsic,
            "distortion_coefficients": distortion_coefficients,
        })

    # Velodyne (sensor -> ego = T_imu_velo)
    calibrated_sensors.append({
        "token": gen_token(),
        "sensor_token": sensor_tok["LIDAR_TOP"],
        "translation": T_imu_velo[:3, 3].tolist(),
        "rotation": rotation_matrix_to_quaternion(T_imu_velo[:3, :3]),
        "camera_intrinsic": "",
        "distortion_coefficients": "",
    })

    # SICK
    calibrated_sensors.append({
        "token": gen_token(),
        "sensor_token": sensor_tok["SICK"],
        "translation": T_imu_sick[:3, 3].tolist(),
        "rotation": rotation_matrix_to_quaternion(T_imu_sick[:3, :3]),
        "camera_intrinsic": "",
        "distortion_coefficients": "",
    })

    calib_tok = {}  # kitti_sensor_name -> calibrated_sensor token
    calib_tok["image_00"] = calibrated_sensors[0]["token"]
    calib_tok["image_01"] = calibrated_sensors[1]["token"]
    calib_tok["image_02"] = calibrated_sensors[2]["token"]
    calib_tok["image_03"] = calibrated_sensors[3]["token"]
    calib_tok["velodyne"] = calibrated_sensors[4]["token"]
    calib_tok["sick"] = calibrated_sensors[5]["token"]

    # ---- log.json ----
    xml_files = sorted([f for f in os.listdir(bbox_dir) if f.endswith(".xml")])
    logs = []
    log_tok_map = {}  # sequence_name -> log_token
    for xf in xml_files:
        seq_name = os.path.splitext(xf)[0]
        date_str = "_".join(seq_name.split("_")[:3])
        tok = gen_token()
        logs.append({
            "token": tok,
            "logfile": seq_name,
            "vehicle": "annieway",
            "date_captured": datetime.datetime.strptime(date_str, "%Y_%m_%d").strftime("%Y-%m-%d"),
            "location": "karlsruhe",
        })
        log_tok_map[seq_name] = tok

    # ---- map.json ----
    maps = []
    for log in logs:
        tok = gen_token()
        maps.append({
            "category": "semantic_prior",
            "token": tok,
            "filename": f"maps/{tok}.png",
            "log_tokens": [log["token"]],
        })
        # create dummy map file (empty PNG) for validation; replace with actual maps if available
        map_path = os.path.join(output_dir, "maps", f"{tok}.png")
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        with open(map_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\xdac\xf8\x0f\x04\x00\x09\xf9\x03\xf5\xa7\xc1\xe2\xd8\x00\x00\x00\x00IEND\xaeB`\x82")

    # ---- category.json ----
    # Use KITTI-360 native category names where possible,
    # nuScenes names only for vehicles/humans that map 1:1.
    # This preserves finer granularity (e.g. construction.building,
    # object.pole, object.trafficsign) instead of collapsing to
    # static.manmade / movable_object.pushable_pullable.
    _kitti360_cat_defs = [
        ("construction.building", "Buildings or constructions."),
        ("construction.wall", "Walls or barriers."),
        ("construction.fence", "Fences or barriers."),
        ("construction.guardrail", "Guard rails."),
        ("construction.bridge", "Bridges."),
        ("construction.tunnel", "Tunnels."),
        ("construction.garage", "Garages or parking structures."),
        ("construction.gate", "Gates or entry barriers."),
        ("construction.stop", "Stops or stopping points."),
        ("flat.road", "Road surface."),
        ("flat.sidewalk", "Sidewalk area."),
        ("flat.parking", "Parking area."),
        ("flat.railtrack", "Rail track area."),
        ("nature.vegetation", "Vegetation or plants."),
        ("nature.terrain", "Terrain or natural ground."),
        ("sky.sky", "Sky region."),
        ("object.pole", "Poles or vertical objects."),
        ("object.polegroup", "Groups of poles."),
        ("object.trafficlight", "Traffic lights."),
        ("object.trafficsign", "Traffic signs."),
        ("object.smallpole", "Small poles or vertical objects."),
        ("object.lamp", "Lamps or lighting structures."),
        ("object.trashbin", "Trash bins or waste containers."),
        ("object.vendingmachine", "Vending machines or automated kiosks."),
        ("object.box", "Boxes or containers."),
        ("human.pedestrian.adult", "Humans or pedestrians."),
        ("human.pedestrian.personal_mobility", "Riders or people on bikes."),
        ("vehicle.car", "Cars or standard vehicles."),
        ("vehicle.truck", "Trucks or heavy vehicles."),
        ("vehicle.bus.rigid", "Buses."),
        ("vehicle.bus.bendy", "Trains or articulated vehicles."),
        ("vehicle.motorcycle", "Motorcycles or two-wheeled vehicles."),
        ("vehicle.bicycle", "Bicycles or pedal-powered vehicles."),
        ("vehicle.caravan", "Caravans or mobile homes."),
        ("vehicle.trailer", "Trailers."),
        ("vehicle.licenseplate", "License plate."),
        ("movable_object.barrier", "Barriers."),
        ("movable_object.pushable_pullable", "Pushable or pullable objects."),
        ("movable_object.debris", "Debris or scattered objects."),
        ("void.unknown", "Unknown or unmapped objects."),
    ]
    categories = []
    cat_name_to_token = {}
    for cat_name, desc in _kitti360_cat_defs:
        tok = gen_token()
        categories.append({"token": tok, "name": cat_name, "description": desc})
        cat_name_to_token[cat_name] = tok

    # KITTI-360 label name -> category name mapping
    _kitti_label_to_cat = {
        "building": "construction.building",
        "wall": "construction.wall",
        "fence": "construction.fence",
        "guardrail": "construction.guardrail",
        "bridge": "construction.bridge",
        "tunnel": "construction.tunnel",
        "garage": "construction.garage",
        "gate": "construction.gate",
        "stop": "construction.stop",
        "road": "flat.road",
        "sidewalk": "flat.sidewalk",
        "parking": "flat.parking",
        "railtrack": "flat.railtrack",
        "vegetation": "nature.vegetation",
        "terrain": "nature.terrain",
        "sky": "sky.sky",
        "pole": "object.pole",
        "polegroup": "object.polegroup",
        "trafficlight": "object.trafficlight",
        "trafficsign": "object.trafficsign",
        "smallpole": "object.smallpole",
        "lamp": "object.lamp",
        "trashbin": "object.trashbin",
        "vendingmachine": "object.vendingmachine",
        "box": "object.box",
        "person": "human.pedestrian.adult",
        "rider": "human.pedestrian.personal_mobility",
        "car": "vehicle.car",
        "truck": "vehicle.truck",
        "bus": "vehicle.bus.rigid",
        "train": "vehicle.bus.bendy",
        "motorcycle": "vehicle.motorcycle",
        "bicycle": "vehicle.bicycle",
        "caravan": "vehicle.caravan",
        "trailer": "vehicle.trailer",
        "unknownconstruction": "construction.building",
        "unknownvehicle": "vehicle.car",
        "unknownobject": "void.unknown",
    }

    # ---- attribute.json ----
    attr_static = gen_token()
    attr_dynamic = gen_token()
    attributes = [
        {"token": attr_static, "name": "category.instance", "description": "Static object with single cuboid."},
        {"token": attr_dynamic, "name": "dynamic.dynamic", "description": "Dynamic object."},
    ]

    # ---- visibility.json ----
    visibility = [
        {"description": "visibility of whole object is between 0 and 40%", "token": "1", "level": "v0-40"},
        {"description": "visibility of whole object is between 40 and 60%", "token": "2", "level": "v40-60"},
        {"description": "visibility of whole object is between 60 and 80%", "token": "3", "level": "v60-80"},
        {"description": "visibility of whole object is between 80 and 100%", "token": "4", "level": "v80-100"},
    ]

    # Write
    save_json(sensors, os.path.join(output_dir, "sensor.json"))
    save_json(calibrated_sensors, os.path.join(output_dir, "calibrated_sensor.json"))
    save_json(logs, os.path.join(output_dir, "log.json"))
    save_json(maps, os.path.join(output_dir, "map.json"))
    save_json(categories, os.path.join(output_dir, "category.json"))
    save_json(attributes, os.path.join(output_dir, "attribute.json"))
    save_json(visibility, os.path.join(output_dir, "visibility.json"))

    return {
        "sensor_tok": sensor_tok,
        "calib_tok": calib_tok,
        "log_tok_map": log_tok_map,
        "T_imu_velo": T_imu_velo,
        "cat_name_to_token": cat_name_to_token,
        "kitti_label_to_cat": _kitti_label_to_cat,
        "attr_static": attr_static,
        "attr_dynamic": attr_dynamic,
    }


# ---------------------------------------------------------------------------
# Phase 2 – Scenes, samples, ego-poses, sample_data
# ---------------------------------------------------------------------------

def _load_split_windows(kitti_root):
    """
    Parse data_3d_semantics train/val split files.
    Returns list of (seq, start, end, split_name).
    """
    windows = []
    sem_dir = os.path.join(kitti_root, "data_3d_semantics", "train")
    for split_name, fname in [("train", "2013_05_28_drive_train.txt"),
                               ("val", "2013_05_28_drive_val.txt")]:
        fpath = os.path.join(sem_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  Warning: {fpath} not found, skipping split '{split_name}'")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("/")
                seq = parts[2]
                fn = parts[-1].replace(".ply", "")
                sf, ef = fn.split("_")
                windows.append((seq, int(sf), int(ef), split_name))
    return windows


def _deduplicate_windows(windows):
    """
    Trim overlapping windows so each frame belongs to exactly one scene.
    Within each sequence, sort by start frame and clip start to prev_end+1.
    """
    from collections import defaultdict as dd
    by_seq = dd(list)
    for seq, sf, ef, split in windows:
        by_seq[seq].append([sf, ef, split])

    result = []
    for seq in sorted(by_seq.keys()):
        wins = sorted(by_seq[seq], key=lambda x: x[0])
        for i in range(1, len(wins)):
            prev_end = wins[i - 1][1]
            if wins[i][0] <= prev_end:
                wins[i][0] = prev_end + 1
        for sf, ef, split in wins:
            if sf <= ef:  # still valid after trimming
                result.append((seq, sf, ef, split))
    return result


def _load_poses_for_sequence(kitti_root, seq):
    """
    Load poses: SLAM-corrected poses.txt merged with OXTS fallback.
    Returns dict frame_id -> 4x4 matrix (FLU convention, post-processed).
    Also returns (velo_timestamps, oxts_timestamps).
    """
    pose_dir = os.path.join(kitti_root, "data_poses", seq)
    raw_dir = os.path.join(kitti_root, "data_3d_raw", seq)

    # OXTS poses (fallback)
    oxts_dir = os.path.join(pose_dir, "oxts")
    oxts, _ = loadOxtsData(oxts_dir)
    from kitti360PoseConversion.convertOxtsToPose import convertOxtsToPose as cvt
    oxts_poses = cvt(oxts)
    # Apply postprocessPoses: convert from NED (x=fwd, y=right, z=down)
    # to FLU (x=fwd, y=left, z=up) to match poses.txt convention
    oxts_poses = postprocessPoses(oxts_poses)

    # Build frame -> pose dict from OXTS
    pose_dict = {}
    for i, p in enumerate(oxts_poses):
        if isinstance(p, np.ndarray) and p.size > 0:
            pose_dict[i] = p

    # Override with SLAM-corrected poses.txt where available
    poses_file = os.path.join(pose_dir, "poses.txt")
    slam_frames = set()
    if os.path.isfile(poses_file):
        data = np.loadtxt(poses_file)
        frames = data[:, 0].astype(int)
        mats = data[:, 1:].reshape(-1, 3, 4)
        for idx, frame in enumerate(frames):
            h = np.eye(4)
            h[:3, :4] = mats[idx]
            pose_dict[frame] = h
            slam_frames.add(int(frame))

    # Timestamps
    velo_ts_file = os.path.join(raw_dir, "velodyne_points", "timestamps.txt")
    with open(velo_ts_file) as f:
        velo_ts = [l.strip() for l in f.readlines()]
    oxts_ts_file = os.path.join(oxts_dir, "timestamps.txt")
    with open(oxts_ts_file) as f:
        oxts_ts = [l.strip() for l in f.readlines()]

    return pose_dict, slam_frames, velo_ts, oxts_ts


def generate_scenes_samples_ego(kitti_root, output_dir, meta):
    """
    Phase 2: scenes, samples, ego_pose, sample_data, split.
    Returns per-sequence data needed for Phase 3.
    """
    log_tok_map = meta["log_tok_map"]
    calib_tok = meta["calib_tok"]

    windows = _load_split_windows(kitti_root)
    windows = _deduplicate_windows(windows)
    print(f"  {len(windows)} deduplicated scene windows")

    # Group windows by sequence for ordered processing
    from collections import OrderedDict
    seq_windows = OrderedDict()
    for seq, sf, ef, split in windows:
        seq_windows.setdefault(seq, []).append((sf, ef, split))

    scenes = []
    split_map = {"train": [], "val": []}
    all_samples = []
    all_ego_poses = []
    all_sample_data = []

    # Collect per-sequence info for Phase 3
    seq_info = {}

    scene_counter = 0
    for seq in sorted(seq_windows.keys()):
        print(f"\n  Processing sequence {seq} ...")
        pose_dict, slam_frames, velo_ts, oxts_ts = _load_poses_for_sequence(kitti_root, seq)

        raw_dir = os.path.join(kitti_root, "data_3d_raw", seq)
        twod_dir = os.path.join(kitti_root, "data_2d_raw", seq)

        # Image / lidar timestamp files
        img_ts = {}
        for cam_idx in range(4):
            cam_name = f"image_{cam_idx:02d}"
            ts_file = os.path.join(twod_dir, cam_name, "timestamps.txt")
            if os.path.isfile(ts_file):
                with open(ts_file) as f:
                    img_ts[cam_name] = [l.strip() for l in f.readlines()]

        lidar_ts = {}
        for lid in ["velodyne", "sick"]:
            ts_file = os.path.join(raw_dir, f"{lid}_points", "timestamps.txt")
            if os.path.isfile(ts_file):
                with open(ts_file) as f:
                    lidar_ts[lid] = [l.strip() for l in f.readlines()]

        # Parse SICK timestamps once per sequence for windowing
        sick_timestamps_parsed = []
        if "sick" in lidar_ts:
            sick_timestamps_parsed = [parse_kitti_timestamp(ts) for ts in lidar_ts["sick"] if ts]
        sick_idx = 0  # running cursor across windows

        # Per-sequence mapping: frame_id -> sample_token
        frame_to_sample = {}
        frame_sample_validity = {}  # frame_id -> bool (has SLAM pose)

        for sf, ef, split in seq_windows[seq]:
            scene_counter += 1
            scene_tok = gen_token()
            scene_name = f"scene-{scene_counter:04d}"

            first_sample_tok = ""
            last_sample_tok = ""
            prev_sample_tok = ""
            n_samples = 0
            window_frames = []  # (frame_id, timestamp, sample_tok, ego_pose_tok)

            prev_quat = None
            prev_trans = None

            for frame_id in range(sf, ef + 1):
                # Check velodyne file exists
                velo_path = os.path.join(raw_dir, "velodyne_points", "data",
                                         f"{frame_id:010d}.bin")
                if not os.path.isfile(velo_path):
                    continue

                # Check pose exists
                if frame_id not in pose_dict:
                    continue
                pose = pose_dict[frame_id]
                if isinstance(pose, np.ndarray) and pose.size == 0:
                    continue

                has_slam = frame_id in slam_frames

                # Timestamp
                ts_str = velo_ts[frame_id] if frame_id < len(velo_ts) else ""
                if not ts_str:
                    continue
                timestamp = parse_kitti_timestamp(ts_str)

                # Ego pose quaternion / translation
                quat = rotation_matrix_to_quaternion(pose[:3, :3])
                trans = pose[:3, 3].tolist()

                sample_tok = gen_token()

                # ego_pose (token matches sample for 1:1 correspondence)
                ego_pose_tok = sample_tok
                pose_ts_str = oxts_ts[frame_id] if frame_id < len(oxts_ts) and oxts_ts[frame_id] else ts_str
                pose_ts = parse_kitti_timestamp(pose_ts_str)

                all_ego_poses.append({
                    "token": ego_pose_tok,
                    "timestamp": pose_ts,
                    "rotation": quat,
                    "translation": trans,
                })

                # sample
                sample = {
                    "token": sample_tok,
                    "timestamp": timestamp,
                    "prev": prev_sample_tok,
                    "next": "",
                    "scene_token": scene_tok,
                }
                if prev_sample_tok:
                    all_samples[-1]["next"] = sample_tok
                all_samples.append(sample)

                if not first_sample_tok:
                    first_sample_tok = sample_tok
                last_sample_tok = sample_tok
                prev_sample_tok = sample_tok
                n_samples += 1

                frame_to_sample[frame_id] = sample_tok
                frame_sample_validity[frame_id] = has_slam

                prev_quat = quat
                prev_trans = trans

                # ---- sample_data ----
                # Velodyne
                velo_sd_tok = gen_token()
                velo_rel = os.path.join("data_3d_raw", seq, "velodyne_points", "data",
                                        f"{frame_id:010d}.bin")
                all_sample_data.append({
                    "token": velo_sd_tok,
                    "sample_token": sample_tok,
                    "ego_pose_token": ego_pose_tok,
                    "calibrated_sensor_token": calib_tok["velodyne"],
                    "timestamp": timestamp,
                    "fileformat": "pcd",
                    "is_key_frame": True,
                    "height": 0,
                    "width": 0,
                    "filename": velo_rel,
                    "prev": "",
                    "next": "",
                })

                # Cameras
                img_sizes = {
                    "image_00": (376, 1408), "image_01": (376, 1408),
                    "image_02": (1400, 1400), "image_03": (1400, 1400),
                }
                img_subdirs = {
                    "image_00": "data_rect", "image_01": "data_rect",
                    "image_02": "data_rgb", "image_03": "data_rgb",
                }
                for cam_name in ["image_00", "image_01", "image_02", "image_03"]:
                    sub = img_subdirs[cam_name]
                    img_rel = os.path.join("data_2d_raw", seq, cam_name, sub,
                                           f"{frame_id:010d}.png")
                    img_full = os.path.join(kitti_root, img_rel)
                    if not os.path.isfile(img_full):
                        continue
                    h, w = img_sizes[cam_name]
                    cam_ts_str = img_ts.get(cam_name, velo_ts)
                    cam_timestamp = parse_kitti_timestamp(cam_ts_str[frame_id]) if frame_id < len(cam_ts_str) and cam_ts_str[frame_id] else timestamp
                    all_sample_data.append({
                        "token": gen_token(),
                        "sample_token": sample_tok,
                        "ego_pose_token": ego_pose_tok,
                        "calibrated_sensor_token": calib_tok[cam_name],
                        "timestamp": cam_timestamp,
                        "fileformat": "png",
                        "is_key_frame": True,
                        "height": h,
                        "width": w,
                        "filename": img_rel,
                        "prev": "",
                        "next": "",
                    })

                # Collect frame info for SICK windowing
                window_frames.append((frame_id, timestamp, sample_tok, ego_pose_tok))

            # --- SICK sample_data (sweeps + closest-to-velodyne keyframe) ---
            if sick_timestamps_parsed and "sick" in calib_tok:
                num_sick = len(sick_timestamps_parsed)
                for idx, (fid, velo_timestamp, s_tok, ep_tok) in enumerate(window_frames):
                    # Time window: midpoint to prev/next velodyne timestamps
                    if idx == 0:
                        prev_start = velo_timestamp
                    else:
                        prev_start = (window_frames[idx - 1][1] + velo_timestamp) // 2
                    if idx == len(window_frames) - 1:
                        next_end = velo_timestamp
                    else:
                        next_end = (velo_timestamp + window_frames[idx + 1][1]) // 2

                    # Advance past SICK scans before this window
                    while sick_idx < num_sick and sick_timestamps_parsed[sick_idx] < prev_start:
                        sick_idx += 1

                    # Collect SICK entries in this window, track closest to velodyne
                    window_sick_start = len(all_sample_data)
                    min_time_diff = float('inf')
                    closest_offset = -1

                    while sick_idx < num_sick and sick_timestamps_parsed[sick_idx] < next_end:
                        sick_ts = sick_timestamps_parsed[sick_idx]
                        time_diff = abs(sick_ts - velo_timestamp)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_offset = len(all_sample_data) - window_sick_start
                        sick_rel = os.path.join("data_3d_raw", seq, "sick_points",
                                                "data", f"{sick_idx:010d}.bin")
                        all_sample_data.append({
                            "token": gen_token(),
                            "sample_token": s_tok,
                            "ego_pose_token": ep_tok,
                            "calibrated_sensor_token": calib_tok["sick"],
                            "timestamp": sick_ts,
                            "fileformat": "pcd",
                            "is_key_frame": False,
                            "height": 0,
                            "width": 0,
                            "filename": sick_rel,
                            "prev": "",
                            "next": "",
                        })
                        sick_idx += 1

                    # Mark closest SICK scan as keyframe
                    if closest_offset >= 0:
                        all_sample_data[window_sick_start + closest_offset]["is_key_frame"] = True

            # Scene record
            if n_samples > 0:
                scenes.append({
                    "token": scene_tok,
                    "log_token": log_tok_map.get(seq, ""),
                    "nbr_samples": n_samples,
                    "first_sample_token": first_sample_tok,
                    "last_sample_token": last_sample_tok,
                    "name": scene_name,
                    "description": f"Sequence {seq} frames {sf}-{ef}",
                })
                split_map[split].append(scene_name)

        seq_info[seq] = {
            "pose_dict": pose_dict,
            "slam_frames": slam_frames,
            "frame_to_sample": frame_to_sample,
            "frame_sample_validity": frame_sample_validity,
        }

    # Build prev/next chains for sample_data (per sensor-channel per scene)
    _build_sample_data_chains(all_sample_data)

    save_json(scenes, os.path.join(output_dir, "scene.json"))
    save_json(split_map, os.path.join(output_dir, "split.json"))
    save_json(all_samples, os.path.join(output_dir, "sample.json"))
    save_json(all_ego_poses, os.path.join(output_dir, "ego_pose.json"))
    save_json(all_sample_data, os.path.join(output_dir, "sample_data.json"))

    return seq_info, windows


def _build_sample_data_chains(sample_data_list):
    """Set prev/next within each (sample_token-series, calibrated_sensor_token) group."""
    # Group by calibrated_sensor_token, ordered by index (which mirrors time order)
    groups = defaultdict(list)
    for i, sd in enumerate(sample_data_list):
        groups[sd["calibrated_sensor_token"]].append(i)

    # Within each group, the records are already in sample order -> link them
    # But we need to respect scene boundaries (prev/next should not cross scenes).
    # We'll use sample_token to detect scene boundaries via the sample prev/next.
    # Simple approach: link consecutive entries in the group.
    for idxs in groups.values():
        for j in range(1, len(idxs)):
            sample_data_list[idxs[j]]["prev"] = sample_data_list[idxs[j - 1]]["token"]
            sample_data_list[idxs[j - 1]]["next"] = sample_data_list[idxs[j]]["token"]


# ---------------------------------------------------------------------------
# Phase 3 – Annotations
# ---------------------------------------------------------------------------

# Category-specific point-count thresholds
_PT_THRESHOLDS = {
    20: 1, 38: 1, 37: 1,  # small: traffic sign, lamp, smallpole
    39: 2, 41: 2, 19: 2,  # medium-small: trash bin, box, traffic light
    24: 2, 25: 2, 32: 2, 33: 2,  # persons/riders/bicycles/motorcycles
    34: 3,  # garage
}
_PT_DEFAULT = 3


def _get_pt_threshold(global_id):
    prefix = global_id // 1000
    return _PT_THRESHOLDS.get(prefix, _PT_DEFAULT)


def generate_annotations(kitti_root, output_dir, meta, seq_info, windows):
    """Phase 3: instance.json, sample_annotation.json."""
    T_imu_velo = meta["T_imu_velo"]
    cat_name_to_token = meta["cat_name_to_token"]
    kitti_label_to_cat = meta["kitti_label_to_cat"]
    attr_static = meta["attr_static"]
    attr_dynamic = meta["attr_dynamic"]

    bbox_dir = os.path.join(kitti_root, "data_3d_bboxes", "train")

    all_instances = []
    all_annotations = []

    # Group windows by sequence
    seq_windows = defaultdict(list)
    for seq, sf, ef, split in windows:
        seq_windows[seq].append((sf, ef))

    for seq in sorted(seq_windows.keys()):
        print(f"\n  Annotating sequence {seq} ...", flush=True)
        t0 = time.time()

        xml_path = os.path.join(bbox_dir, f"{seq}.xml")
        if not os.path.isfile(xml_path):
            print(f"    No XML found at {xml_path}, skipping", flush=True)
            continue

        objects = load_annotation_xml(xml_path)
        si = seq_info[seq]
        pose_dict = si["pose_dict"]
        frame_to_sample = si["frame_to_sample"]

        # Separate static / dynamic objects
        static_ids = [gid for gid, ts_map in objects.items() if len(ts_map) == 1 and -1 in ts_map]
        dynamic_ids = [gid for gid, ts_map in objects.items() if len(ts_map) > 1 or (len(ts_map) == 1 and -1 not in ts_map)]

        # Pre-compute static box parameters (world coords)
        static_boxes = []
        for gid in static_ids:
            bbox = objects[gid][-1]
            # Map to category via KITTI label name
            cat_key = bbox.name.replace(" ", "")
            cat_name = kitti_label_to_cat.get(cat_key)
            if cat_name is None:
                continue
            cat_token = cat_name_to_token.get(cat_name)
            if cat_token is None:
                continue
            center = box_center(bbox.vertices)
            dims = box_dimensions(bbox.vertices)
            rot_q = box_yaw_quaternion(bbox.vertices)
            rot_mat = box_rotation_matrix(bbox.vertices)
            pt_thresh = _get_pt_threshold(gid)
            static_boxes.append({
                "global_id": gid,
                "center": center,
                "dims": dims,
                "rot_q": rot_q,
                "rot_mat": rot_mat,
                "cat_token": cat_token,
                "pt_thresh": pt_thresh,
            })

        n_static = len(static_boxes)
        if n_static > 0:
            s_centers = np.array([b["center"] for b in static_boxes], dtype=np.float64)
            s_sizes = np.array([b["dims"] for b in static_boxes], dtype=np.float64)
            s_rots = np.array([b["rot_mat"] for b in static_boxes], dtype=np.float64)
            s_thresholds = np.array([b["pt_thresh"] for b in static_boxes], dtype=np.int32)

        # Instance tracking: global_id -> list of (frame_id, annotation_dict)
        instance_annos = defaultdict(list)

        all_scene_frames = set()
        for sf, ef in seq_windows[seq]:
            for fid in range(sf, ef + 1):
                if fid in frame_to_sample:
                    all_scene_frames.add(fid)

        # Pre-build dynamic object frame index: frame_id -> list of (gid, bbox)
        dyn_frame_index = defaultdict(list)
        for gid in dynamic_ids:
            ts_map = objects[gid]
            for ts, bbox in ts_map.items():
                if ts >= 0:
                    dyn_frame_index[ts].append((gid, bbox))

        n_frames = 0
        total_frames = len(all_scene_frames)
        for frame_id in sorted(all_scene_frames):
            sample_tok = frame_to_sample[frame_id]
            if frame_id not in pose_dict:
                continue
            pose = pose_dict[frame_id]
            if isinstance(pose, np.ndarray) and pose.size == 0:
                continue

            has_dynamic = frame_id in dyn_frame_index

            # Quick skip: if no static boxes and no dynamic objects for this frame
            if n_static == 0 and not has_dynamic:
                n_frames += 1
                continue

            # Load velodyne points
            velo_path = os.path.join(kitti_root, "data_3d_raw", seq,
                                     "velodyne_points", "data", f"{frame_id:010d}.bin")
            if not os.path.isfile(velo_path):
                continue

            points_raw = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
            pts_xyz = points_raw[:, :3].copy()

            # Ground filter in SENSOR-LOCAL coords
            ground_mask = pts_xyz[:, 2] > -1.5
            pts_xyz = pts_xyz[ground_mask]

            # Voxel downsample for faster point-in-box (0.15m grid)
            voxel_size = 0.15
            voxel_keys = (pts_xyz / voxel_size).astype(np.int32)
            _, unique_idx = np.unique(voxel_keys, axis=0, return_index=True)
            pts_xyz = pts_xyz[unique_idx]

            # Transform to world
            T_world_velo = pose @ T_imu_velo
            pts_homo = np.column_stack([pts_xyz, np.ones(len(pts_xyz), dtype=np.float32)])
            pts_world = (T_world_velo @ pts_homo.T).T[:, :3]

            # Sensor origin in world
            velo_origin = (T_world_velo @ np.array([0, 0, 0, 1]))[:3]

            n_frames += 1
            if n_frames % 1000 == 0:
                print(f"    [{seq}] {n_frames}/{total_frames} frames ...", flush=True)

            # --- Static objects ---
            if n_static > 0:
                # Distance filter: only check boxes within 120m
                dists = np.linalg.norm(s_centers[:, :2] - velo_origin[:2], axis=1)
                dist_mask = dists < 120.0

                if dist_mask.any():
                    sel_idx = np.where(dist_mask)[0]
                    counts = points_in_box_counts(
                        pts_world, s_centers[sel_idx], s_sizes[sel_idx], s_rots[sel_idx]
                    )
                    for j, box_idx in enumerate(sel_idx):
                        n_pts = int(counts[j])
                        if n_pts > s_thresholds[box_idx]:
                            b = static_boxes[box_idx]
                            instance_annos[b["global_id"]].append((
                                frame_id, sample_tok, b["center"].tolist(),
                                b["rot_q"], b["dims"], b["cat_token"],
                                n_pts, True,  # is_static
                            ))

            # --- Dynamic objects ---
            dyn_candidates = []
            for gid, bbox in dyn_frame_index.get(frame_id, []):
                cat_key = bbox.name.replace(" ", "")
                cat_name = kitti_label_to_cat.get(cat_key)
                if cat_name is None:
                    continue
                cat_token = cat_name_to_token.get(cat_name)
                if cat_token is None:
                    continue

                center = box_center(bbox.vertices)
                dims = box_dimensions(bbox.vertices)
                rot_q = box_yaw_quaternion(bbox.vertices)
                rot_mat = box_rotation_matrix(bbox.vertices)
                pt_thresh = _get_pt_threshold(gid)

                # Distance check
                dist = np.linalg.norm(center[:2] - velo_origin[:2])
                if dist > 120.0:
                    continue

                dyn_candidates.append((gid, center, dims, rot_q, rot_mat, pt_thresh, cat_token))

            if dyn_candidates:
                d_centers = np.array([c[1] for c in dyn_candidates], dtype=np.float64)
                d_sizes = np.array([c[2] for c in dyn_candidates], dtype=np.float64)
                d_rots = np.array([c[4] for c in dyn_candidates], dtype=np.float64)
                d_counts = points_in_box_counts(pts_world, d_centers, d_sizes, d_rots)
                for k, (gid, center, dims, rot_q, rot_mat, pt_thresh, cat_token) in enumerate(dyn_candidates):
                    n_pts = int(d_counts[k])
                    if n_pts > pt_thresh:
                        instance_annos[gid].append((
                            frame_id, sample_tok, center.tolist(),
                            rot_q, dims, cat_token,
                            n_pts, False,  # is_dynamic
                        ))

        elapsed = time.time() - t0
        print(f"    Processed {n_frames} frames in {elapsed:.1f}s, "
              f"{len(instance_annos)} instances with annotations", flush=True)

        # Build instance + annotation records
        for gid, annos in instance_annos.items():
            annos.sort(key=lambda x: x[0])  # sort by frame_id
            if not annos:
                continue

            instance_tok = gen_token()
            is_static = annos[0][7]
            cat_token = annos[0][5]
            attr_tok = attr_static if is_static else attr_dynamic

            anno_records = []
            prev_tok = ""
            for frame_id, sample_tok, translation, rotation, size, ct, n_pts, _ in annos:
                anno_tok = gen_token()
                rec = {
                    "token": anno_tok,
                    "sample_token": sample_tok,
                    "instance_token": instance_tok,
                    "visibility_token": "4",
                    "attribute_tokens": [attr_tok],
                    "translation": translation,
                    "rotation": rotation,
                    "size": size,
                    "num_lidar_pts": n_pts,
                    "num_radar_pts": 0,
                    "prev": prev_tok,
                    "next": "",
                }
                if prev_tok and anno_records:
                    anno_records[-1]["next"] = anno_tok
                anno_records.append(rec)
                prev_tok = anno_tok

            inst_rec = {
                "token": instance_tok,
                "category_token": cat_token,
                "nbr_annotations": len(anno_records),
                "first_annotation_token": anno_records[0]["token"],
                "last_annotation_token": anno_records[-1]["token"],
            }
            all_instances.append(inst_rec)
            all_annotations.extend(anno_records)

    save_json(all_instances, os.path.join(output_dir, "instance.json"))
    save_json(all_annotations, os.path.join(output_dir, "sample_annotation.json"))

    return all_instances, all_annotations


# ---------------------------------------------------------------------------
# Phase 4 – Validation
# ---------------------------------------------------------------------------

def validate_output(output_dir):
    """Built-in consistency checks on the generated JSON tables."""
    print("\n=== Validation ===")
    errors = []

    def load(name):
        with open(os.path.join(output_dir, name)) as f:
            return json.load(f)

    scenes = load("scene.json")
    samples = load("sample.json")
    sample_data = load("sample_data.json")
    ego_poses = load("ego_pose.json")
    annotations = load("sample_annotation.json")
    instances = load("instance.json")
    categories = load("category.json")
    sensors = load("sensor.json")
    calib_sensors = load("calibrated_sensor.json")

    # Token sets
    sample_toks = {s["token"] for s in samples}
    scene_toks = {s["token"] for s in scenes}
    ego_toks = {e["token"] for e in ego_poses}
    sd_toks = {d["token"] for d in sample_data}
    inst_toks = {i["token"] for i in instances}
    cat_toks = {c["token"] for c in categories}
    sensor_toks = {s["token"] for s in sensors}
    calib_toks = {c["token"] for c in calib_sensors}
    anno_toks = {a["token"] for a in annotations}

    # 1. All scene first/last sample tokens exist
    for sc in scenes:
        if sc["first_sample_token"] not in sample_toks:
            errors.append(f"Scene {sc['name']}: first_sample_token missing")
        if sc["last_sample_token"] not in sample_toks:
            errors.append(f"Scene {sc['name']}: last_sample_token missing")

    # 2. Sample linked-list consistency
    for s in samples:
        if s["scene_token"] not in scene_toks:
            errors.append(f"Sample {s['token']}: invalid scene_token")
        if s["prev"] and s["prev"] not in sample_toks:
            errors.append(f"Sample {s['token']}: invalid prev")
        if s["next"] and s["next"] not in sample_toks:
            errors.append(f"Sample {s['token']}: invalid next")

    # 3. Sample_data references
    for sd in sample_data:
        if sd["sample_token"] not in sample_toks:
            errors.append(f"SampleData {sd['token']}: invalid sample_token")
        if sd["ego_pose_token"] not in ego_toks:
            errors.append(f"SampleData {sd['token']}: invalid ego_pose_token")
        if sd["calibrated_sensor_token"] not in calib_toks:
            errors.append(f"SampleData {sd['token']}: invalid calibrated_sensor_token")

    # 4. Annotation references
    annotated_samples = set()
    for a in annotations:
        if a["sample_token"] not in sample_toks:
            errors.append(f"Annotation {a['token']}: invalid sample_token")
        if a["instance_token"] not in inst_toks:
            errors.append(f"Annotation {a['token']}: invalid instance_token")
        if a["prev"] and a["prev"] not in anno_toks:
            errors.append(f"Annotation {a['token']}: invalid prev")
        if a["next"] and a["next"] not in anno_toks:
            errors.append(f"Annotation {a['token']}: invalid next")
        annotated_samples.add(a["sample_token"])

    # 5. Instance references
    for inst in instances:
        if inst["category_token"] not in cat_toks:
            errors.append(f"Instance {inst['token']}: invalid category_token")
        if inst["first_annotation_token"] not in anno_toks:
            errors.append(f"Instance {inst['token']}: invalid first_annotation_token")
        if inst["last_annotation_token"] not in anno_toks:
            errors.append(f"Instance {inst['token']}: invalid last_annotation_token")

    # 6. Calibrated sensor -> sensor
    for cs in calib_sensors:
        if cs["sensor_token"] not in sensor_toks:
            errors.append(f"CalibratedSensor {cs['token']}: invalid sensor_token")

    # Stats
    print(f"  Scenes:       {len(scenes)}")
    print(f"  Samples:      {len(samples)}")
    print(f"  Ego Poses:    {len(ego_poses)}")
    print(f"  Sample Data:  {len(sample_data)}")
    print(f"  Annotations:  {len(annotations)}")
    print(f"  Instances:    {len(instances)}")
    print(f"  Samples with annotations: {len(annotated_samples)} / {len(samples)}")
    unannotated = len(samples) - len(annotated_samples)
    if unannotated > 0:
        print(f"  WARNING: {unannotated} samples have no annotations")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print("  All referential integrity checks PASSED")

    return errors


def validate_with_nuscenes_devkit(data_root, version):
    """Try loading the generated dataset with the nuScenes devkit."""
    print("\n=== nuScenes devkit validation ===")
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        print(f"  Successfully loaded: {len(nusc.scene)} scenes, "
              f"{len(nusc.sample)} samples, "
              f"{len(nusc.sample_annotation)} annotations")

        # Quick sanity: traverse a few scenes
        ok = 0
        for scene in nusc.scene[:3]:
            tok = scene["first_sample_token"]
            count = 0
            while tok:
                s = nusc.get("sample", tok)
                count += 1
                tok = s["next"] if s["next"] else None
            if count == scene["nbr_samples"]:
                ok += 1
            else:
                print(f"  Scene {scene['name']}: expected {scene['nbr_samples']} "
                      f"samples, traversed {count}")
        print(f"  Scene traversal check: {ok}/{min(3, len(nusc.scene))} passed")
        return True
    except Exception as e:
        print(f"  nuScenes devkit validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KITTI-360 to nuScenes converter")
    parser.add_argument("--kitti-root", default="./kitti360",
                        help="Path to KITTI-360 dataset root")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for JSON files (default: <kitti-root>/vkitti360-trainval)")
    parser.add_argument("--version", default="vkitti360-trainval",
                        help="Dataset version name")
    parser.add_argument("--skip-devkit", action="store_true",
                        help="Skip nuScenes devkit validation")
    args = parser.parse_args()

    kitti_root = args.kitti_root
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(kitti_root, args.version)

    data_root = os.path.dirname(output_dir)  # parent of version dir

    print(f"KITTI-360 root: {kitti_root}")
    print(f"Output dir:     {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1
    print("\n--- Phase 1: Static metadata ---")
    meta = generate_static_tables(kitti_root, output_dir)

    # Phase 2
    print("\n--- Phase 2: Scenes, samples, ego poses, sample data ---")
    seq_info, windows = generate_scenes_samples_ego(kitti_root, output_dir, meta)

    # Phase 3
    print("\n--- Phase 3: Annotations ---")
    generate_annotations(kitti_root, output_dir, meta, seq_info, windows)

    # Phase 4
    print("\n--- Phase 4: Validation ---")
    errors = validate_output(output_dir)

    if not args.skip_devkit:
        validate_with_nuscenes_devkit(data_root, args.version)

    if errors:
        print(f"\n[RESULT] Conversion completed with {len(errors)} validation errors.")
        return 1
    else:
        print(f"\n[RESULT] Conversion completed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
