#!/usr/bin/env python3
"""Create a mini nuScenes-style subset from vkitti360-trainval."""

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_SCENES = [
    "scene-0014",
    "scene-0203",
    "scene-0256",
    "scene-0117",
    "scene-0292",
    "scene-0287",
]

GLOBAL_TABLES = [
    "sensor.json",
    "calibrated_sensor.json",
    "attribute.json",
    "visibility.json",
]


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4)
    print(f"Wrote {path} ({len(data)} records)")


def prune_links(records, token_key="token"):
    valid_tokens = {record[token_key] for record in records}
    for record in records:
        if record.get("prev") and record["prev"] not in valid_tokens:
            record["prev"] = ""
        if record.get("next") and record["next"] not in valid_tokens:
            record["next"] = ""


def copy_table_if_present(source_dir, target_dir, table_name):
    source = source_dir / table_name
    if source.exists():
        target = target_dir / table_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Copied {source} -> {target}")


def ensure_map_masks(source_dir, dataroot, map_records):
    dataroot_maps = dataroot / "maps"
    dataroot_maps.mkdir(parents=True, exist_ok=True)
    source_maps = source_dir / "maps"

    for record in map_records:
        rel_path = Path(record["filename"])
        destination = dataroot / rel_path
        if destination.exists():
            continue

        fallback = source_maps / rel_path.name
        if fallback.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, destination)
            print(f"Copied missing map mask {fallback} -> {destination}")
            continue

        raise FileNotFoundError(
            f"Missing map mask {destination}. Expected either {destination} "
            f"or fallback {fallback}."
        )


def validate_references(target_dir):
    scenes = load_json(target_dir / "scene.json")
    samples = load_json(target_dir / "sample.json")
    sample_data = load_json(target_dir / "sample_data.json")
    ego_poses = load_json(target_dir / "ego_pose.json")
    annotations = load_json(target_dir / "sample_annotation.json")
    instances = load_json(target_dir / "instance.json")
    categories = load_json(target_dir / "category.json")
    sensors = load_json(target_dir / "sensor.json")
    calibrated_sensors = load_json(target_dir / "calibrated_sensor.json")

    sample_tokens = {item["token"] for item in samples}
    scene_tokens = {item["token"] for item in scenes}
    sample_data_tokens = {item["token"] for item in sample_data}
    ego_pose_tokens = {item["token"] for item in ego_poses}
    annotation_tokens = {item["token"] for item in annotations}
    instance_tokens = {item["token"] for item in instances}
    category_tokens = {item["token"] for item in categories}
    sensor_tokens = {item["token"] for item in sensors}
    calibrated_sensor_tokens = {item["token"] for item in calibrated_sensors}

    errors = []

    for scene in scenes:
        if scene["token"] not in scene_tokens:
            errors.append(f"invalid scene token {scene['token']}")
        if scene["first_sample_token"] not in sample_tokens:
            errors.append(f"scene {scene['name']} has invalid first_sample_token")
        if scene["last_sample_token"] not in sample_tokens:
            errors.append(f"scene {scene['name']} has invalid last_sample_token")

    for sample in samples:
        if sample["scene_token"] not in scene_tokens:
            errors.append(f"sample {sample['token']} has invalid scene_token")
        if sample.get("prev") and sample["prev"] not in sample_tokens:
            errors.append(f"sample {sample['token']} has invalid prev")
        if sample.get("next") and sample["next"] not in sample_tokens:
            errors.append(f"sample {sample['token']} has invalid next")

    for record in sample_data:
        if record["sample_token"] not in sample_tokens:
            errors.append(f"sample_data {record['token']} has invalid sample_token")
        if record["ego_pose_token"] not in ego_pose_tokens:
            errors.append(f"sample_data {record['token']} has invalid ego_pose_token")
        if record["calibrated_sensor_token"] not in calibrated_sensor_tokens:
            errors.append(f"sample_data {record['token']} has invalid calibrated_sensor_token")
        if record.get("prev") and record["prev"] not in sample_data_tokens:
            errors.append(f"sample_data {record['token']} has invalid prev")
        if record.get("next") and record["next"] not in sample_data_tokens:
            errors.append(f"sample_data {record['token']} has invalid next")

    for record in annotations:
        if record["sample_token"] not in sample_tokens:
            errors.append(f"annotation {record['token']} has invalid sample_token")
        if record["instance_token"] not in instance_tokens:
            errors.append(f"annotation {record['token']} has invalid instance_token")
        if record.get("prev") and record["prev"] not in annotation_tokens:
            errors.append(f"annotation {record['token']} has invalid prev")
        if record.get("next") and record["next"] not in annotation_tokens:
            errors.append(f"annotation {record['token']} has invalid next")

    for record in instances:
        if record["category_token"] not in category_tokens:
            errors.append(f"instance {record['token']} has invalid category_token")
        if record["first_annotation_token"] not in annotation_tokens:
            errors.append(f"instance {record['token']} has invalid first_annotation_token")
        if record["last_annotation_token"] not in annotation_tokens:
            errors.append(f"instance {record['token']} has invalid last_annotation_token")

    for record in calibrated_sensors:
        if record["sensor_token"] not in sensor_tokens:
            errors.append(f"calibrated_sensor {record['token']} has invalid sensor_token")

    return errors


def validate_with_devkit(dataroot, version):
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
    print(
        f"Loaded {version}: {len(nusc.scene)} scenes, {len(nusc.sample)} samples, "
        f"{len(nusc.sample_annotation)} annotations"
    )


def build_mini_dataset(source_dir, target_dir, scene_names):
    scenes = load_json(source_dir / "scene.json")
    selected_scenes = [scene for scene in scenes if scene["name"] in scene_names]
    found_scene_names = {scene["name"] for scene in selected_scenes}
    missing_scene_names = [name for name in scene_names if name not in found_scene_names]
    if missing_scene_names:
        raise ValueError(f"Requested scenes not found: {missing_scene_names}")

    save_json(selected_scenes, target_dir / "scene.json")

    selected_scene_tokens = {scene["token"] for scene in selected_scenes}
    selected_log_tokens = {scene["log_token"] for scene in selected_scenes}

    logs = load_json(source_dir / "log.json")
    selected_logs = [record for record in logs if record["token"] in selected_log_tokens]
    save_json(selected_logs, target_dir / "log.json")

    maps = load_json(source_dir / "map.json")
    selected_maps = [
        record for record in maps if any(token in selected_log_tokens for token in record["log_tokens"])
    ]
    save_json(selected_maps, target_dir / "map.json")

    samples = load_json(source_dir / "sample.json")
    selected_samples = [record for record in samples if record["scene_token"] in selected_scene_tokens]
    prune_links(selected_samples)
    save_json(selected_samples, target_dir / "sample.json")
    selected_sample_tokens = {record["token"] for record in selected_samples}

    ego_poses = load_json(source_dir / "ego_pose.json")
    selected_ego_poses = [record for record in ego_poses if record["token"] in selected_sample_tokens]
    save_json(selected_ego_poses, target_dir / "ego_pose.json")

    sample_data = load_json(source_dir / "sample_data.json")
    selected_sample_data = [record for record in sample_data if record["sample_token"] in selected_sample_tokens]
    prune_links(selected_sample_data)
    save_json(selected_sample_data, target_dir / "sample_data.json")

    annotations = load_json(source_dir / "sample_annotation.json")
    selected_annotations = [record for record in annotations if record["sample_token"] in selected_sample_tokens]
    prune_links(selected_annotations)
    save_json(selected_annotations, target_dir / "sample_annotation.json")
    selected_instance_tokens = {record["instance_token"] for record in selected_annotations}

    instances = load_json(source_dir / "instance.json")
    annotations_by_instance = {}
    for record in selected_annotations:
        annotations_by_instance.setdefault(record["instance_token"], []).append(record)

    selected_instances = []
    for record in instances:
        if record["token"] not in selected_instance_tokens:
            continue
        instance_annotations = annotations_by_instance.get(record["token"], [])
        if not instance_annotations:
            continue
        updated_record = dict(record)
        updated_record["nbr_annotations"] = len(instance_annotations)
        updated_record["first_annotation_token"] = instance_annotations[0]["token"]
        updated_record["last_annotation_token"] = instance_annotations[-1]["token"]
        selected_instances.append(updated_record)
    save_json(selected_instances, target_dir / "instance.json")
    selected_category_tokens = {record["category_token"] for record in selected_instances}

    categories = load_json(source_dir / "category.json")
    selected_categories = [record for record in categories if record["token"] in selected_category_tokens]
    save_json(selected_categories, target_dir / "category.json")

    split_path = source_dir / "split.json"
    if split_path.exists():
        split_map = load_json(split_path)
        filtered_split = {
            split_name: [scene_name for scene_name in split_scenes if scene_name in found_scene_names]
            for split_name, split_scenes in split_map.items()
        }
        save_json(filtered_split, target_dir / "split.json")

    for table_name in GLOBAL_TABLES:
        copy_table_if_present(source_dir, target_dir, table_name)

    ensure_map_masks(source_dir, source_dir.parent, selected_maps)


def parse_args():
    parser = argparse.ArgumentParser(description="Create vkitti360-mini from vkitti360-trainval")
    parser.add_argument(
        "--source-dir",
        default="./vkitti360-trainval",
        help="Source trainval directory",
    )
    parser.add_argument(
        "--target-dir",
        default="./vkitti360-mini",
        help="Target mini directory",
    )
    parser.add_argument(
        "--scene",
        dest="scenes",
        action="append",
        help="Scene name to include. Repeat to override defaults.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the target directory before generating the mini dataset.",
    )
    parser.add_argument(
        "--skip-devkit",
        action="store_true",
        help="Skip nuScenes devkit validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scene_names = args.scenes if args.scenes else DEFAULT_SCENES

    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    dataroot = target_dir.parent

    if args.overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    build_mini_dataset(source_dir, target_dir, scene_names)

    reference_errors = validate_references(target_dir)
    if reference_errors:
        raise RuntimeError(
            "Mini dataset validation failed:\n" + "\n".join(reference_errors[:20])
        )

    print("Referential integrity checks passed")

    if not args.skip_devkit:
        validate_with_devkit(dataroot, target_dir.name)

    print("Mini dataset creation complete")


if __name__ == "__main__":
    main()