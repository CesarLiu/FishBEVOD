# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

import json
import os
from typing import Dict, List

from nuscenes import NuScenes

train_detect = \
    ["scene-0001",
        "scene-0002",
        "scene-0003",
        "scene-0004",
        "scene-0005",
        "scene-0007",
        "scene-0008",
        "scene-0009",
        "scene-0010",
        "scene-0011",
        "scene-0012",
        "scene-0013",
        "scene-0014",
        "scene-0015",
        "scene-0016",
        "scene-0017",
        "scene-0018",
        "scene-0019",
        "scene-0020",
        "scene-0021",
        "scene-0022",
        "scene-0023",
        "scene-0024",
        "scene-0025",
        "scene-0026",
        "scene-0027",
        "scene-0028",
        "scene-0029",
        "scene-0030",
        "scene-0031",
        "scene-0032",
        "scene-0033",
        "scene-0034",
        "scene-0035",
        "scene-0036",
        "scene-0037",
        "scene-0038",
        "scene-0039",
        "scene-0041",
        "scene-0042",
        "scene-0043",
        "scene-0044",
        "scene-0045",
        "scene-0046",
        "scene-0047",
        "scene-0048",
        "scene-0049",
        "scene-0050",
        "scene-0052",
        "scene-0053",
        "scene-0054",
        "scene-0055",
        "scene-0056",
        "scene-0057",
        "scene-0058",
        "scene-0059",
        "scene-0060",
        "scene-0061",
        "scene-0062",
        "scene-0063",
        "scene-0064",
        "scene-0065",
        "scene-0066",
        "scene-0067",
        "scene-0068",
        "scene-0069",
        "scene-0070",
        "scene-0073",
        "scene-0075",
        "scene-0079",
        "scene-0081",
        "scene-0082",
        "scene-0084",
        "scene-0085",
        "scene-0086",
        "scene-0087",
        "scene-0088",
        "scene-0090",
        "scene-0091",
        "scene-0092",
        "scene-0093",
        "scene-0094",
        "scene-0095",
        "scene-0096",
        "scene-0097",
        "scene-0098",
        "scene-0099",
        "scene-0100",
        "scene-0101",
        "scene-0102",
        "scene-0103",
        "scene-0104",
        "scene-0105",
        "scene-0106",
        "scene-0107",
        "scene-0108",
        "scene-0109",
        "scene-0110",
        "scene-0111",
        "scene-0112",
        "scene-0113",
        "scene-0114",
        "scene-0115",
        "scene-0116",
        "scene-0117",
        "scene-0118",
        "scene-0120",
        "scene-0121",
        "scene-0122",
        "scene-0123",
        "scene-0124",
        "scene-0126",
        "scene-0127",
        "scene-0128",
        "scene-0129",
        "scene-0131",
        "scene-0132",
        "scene-0134",
        "scene-0135",
        "scene-0136",
        "scene-0137",
        "scene-0138",
        "scene-0139",
        "scene-0140",
        "scene-0141",
        "scene-0142",
        "scene-0143",
        "scene-0144",
        "scene-0145",
        "scene-0146",
        "scene-0148",
        "scene-0149",
        "scene-0150",
        "scene-0151",
        "scene-0152",
        "scene-0155",
        "scene-0156",
        "scene-0157",
        "scene-0158",
        "scene-0159",
        "scene-0160",
        "scene-0161",
        "scene-0162",
        "scene-0163",
        "scene-0179",
        "scene-0180",
        "scene-0183",
        "scene-0184",
        "scene-0185",
        "scene-0186",
        "scene-0187",
        "scene-0188",
        "scene-0189",
        "scene-0190",
        "scene-0191",
        "scene-0192",
        "scene-0193",
        "scene-0196",
        "scene-0197",
        "scene-0198",
        "scene-0199",
        "scene-0200",
        "scene-0201",
        "scene-0202",
        "scene-0203",
        "scene-0204",
        "scene-0205",
        "scene-0206",
        "scene-0207",
        "scene-0208",
        "scene-0209",
        "scene-0210",
        "scene-0211",
        "scene-0212",
        "scene-0213",
        "scene-0214",
        "scene-0215",
        "scene-0216",
        "scene-0217",
        "scene-0218",
        "scene-0219",
        "scene-0220",
        "scene-0221",
        "scene-0222",
        "scene-0223",
        "scene-0224",
        "scene-0225",
        "scene-0226",
        "scene-0227",
        "scene-0228",
        "scene-0229",
        "scene-0230",
        "scene-0231",
        "scene-0232",
        "scene-0233",
        "scene-0234",
        "scene-0235",
        "scene-0236",
        "scene-0237",
        "scene-0238",
        "scene-0239",
        "scene-0240",
        "scene-0241",
        "scene-0242",
        "scene-0243",
        "scene-0244",
        "scene-0245",
        "scene-0246",
        "scene-0247",
        "scene-0248",
        "scene-0251",
        "scene-0252",
        "scene-0253",
        "scene-0254",
        "scene-0255",
        "scene-0256",
        "scene-0257",
        "scene-0258"
        ]

train_track = \
    ["scene-0001",
        "scene-0002",
        "scene-0003",
        "scene-0004",
        "scene-0005",
        "scene-0007",
        "scene-0008",
        "scene-0009",
        "scene-0010",
        "scene-0011",
        "scene-0012",
        "scene-0013",
        "scene-0014",
        "scene-0015",
        "scene-0016",
        "scene-0017",
        "scene-0018",
        "scene-0019",
        "scene-0020",
        "scene-0021",
        "scene-0022",
        "scene-0023",
        "scene-0024",
        "scene-0025",
        "scene-0026",
        "scene-0027",
        "scene-0028",
        "scene-0029",
        "scene-0030",
        "scene-0031",
        "scene-0032",
        "scene-0033",
        "scene-0034",
        "scene-0035",
        "scene-0036",
        "scene-0037",
        "scene-0038",
        "scene-0039",
        "scene-0041",
        "scene-0042",
        "scene-0043",
        "scene-0044",
        "scene-0045",
        "scene-0046",
        "scene-0047",
        "scene-0048",
        "scene-0049",
        "scene-0050",
        "scene-0052",
        "scene-0053",
        "scene-0054",
        "scene-0055",
        "scene-0056",
        "scene-0057",
        "scene-0058",
        "scene-0059",
        "scene-0060",
        "scene-0061",
        "scene-0062",
        "scene-0063",
        "scene-0064",
        "scene-0065",
        "scene-0066",
        "scene-0067",
        "scene-0068",
        "scene-0069",
        "scene-0070",
        "scene-0073",
        "scene-0075",
        "scene-0079",
        "scene-0081",
        "scene-0082",
        "scene-0084",
        "scene-0085",
        "scene-0086",
        "scene-0087",
        "scene-0088",
        "scene-0090",
        "scene-0091",
        "scene-0092",
        "scene-0093",
        "scene-0094",
        "scene-0095",
        "scene-0096",
        "scene-0097",
        "scene-0098",
        "scene-0099",
        "scene-0100",
        "scene-0101",
        "scene-0102",
        "scene-0103",
        "scene-0104",
        "scene-0105",
        "scene-0106",
        "scene-0107",
        "scene-0108",
        "scene-0109",
        "scene-0110",
        "scene-0111",
        "scene-0112",
        "scene-0113",
        "scene-0114",
        "scene-0115",
        "scene-0116",
        "scene-0117",
        "scene-0118",
        "scene-0120",
        "scene-0121",
        "scene-0122",
        "scene-0123",
        "scene-0124",
        "scene-0126",
        "scene-0127",
        "scene-0128",
        "scene-0129",
        "scene-0131",
        "scene-0132",
        "scene-0134",
        "scene-0135",
        "scene-0136",
        "scene-0137",
        "scene-0138",
        "scene-0139",
        "scene-0140",
        "scene-0141",
        "scene-0142",
        "scene-0143",
        "scene-0144",
        "scene-0145",
        "scene-0146",
        "scene-0148",
        "scene-0149",
        "scene-0150",
        "scene-0151",
        "scene-0152",
        "scene-0155",
        "scene-0156",
        "scene-0157",
        "scene-0158",
        "scene-0159",
        "scene-0160",
        "scene-0161",
        "scene-0162",
        "scene-0163",
        "scene-0179",
        "scene-0180",
        "scene-0183",
        "scene-0184",
        "scene-0185",
        "scene-0186",
        "scene-0187",
        "scene-0188",
        "scene-0189",
        "scene-0190",
        "scene-0191",
        "scene-0192",
        "scene-0193",
        "scene-0196",
        "scene-0197",
        "scene-0198",
        "scene-0199",
        "scene-0200",
        "scene-0201",
        "scene-0202",
        "scene-0203",
        "scene-0204",
        "scene-0205",
        "scene-0206",
        "scene-0207",
        "scene-0208",
        "scene-0209",
        "scene-0210",
        "scene-0211",
        "scene-0212",
        "scene-0213",
        "scene-0214",
        "scene-0215",
        "scene-0216",
        "scene-0217",
        "scene-0218",
        "scene-0219",
        "scene-0220",
        "scene-0221",
        "scene-0222",
        "scene-0223",
        "scene-0224",
        "scene-0225",
        "scene-0226",
        "scene-0227",
        "scene-0228",
        "scene-0229",
        "scene-0230",
        "scene-0231",
        "scene-0232",
        "scene-0233",
        "scene-0234",
        "scene-0235",
        "scene-0236",
        "scene-0237",
        "scene-0238",
        "scene-0239",
        "scene-0240",
        "scene-0241",
        "scene-0242",
        "scene-0243",
        "scene-0244",
        "scene-0245",
        "scene-0246",
        "scene-0247",
        "scene-0248",
        "scene-0251",
        "scene-0252",
        "scene-0253",
        "scene-0254",
        "scene-0255",
        "scene-0256",
        "scene-0257",
        "scene-0258"
        ]

train = list(sorted(set(train_detect + train_track)))

val = \
    [   "scene-0259",
        "scene-0260",
        "scene-0261",
        "scene-0262",
        "scene-0263",
        "scene-0264",
        "scene-0265",
        "scene-0266",
        "scene-0269",
        "scene-0270",
        "scene-0272",
        "scene-0273",
        "scene-0274",
        "scene-0275",
        "scene-0276",
        "scene-0278",
        "scene-0279",
        "scene-0286",
        "scene-0287",
        "scene-0288",
        "scene-0289",
        "scene-0290",
        "scene-0291",
        "scene-0292",
        "scene-0293",
        "scene-0294",
        "scene-0295",
        "scene-0297",
        "scene-0298",
        "scene-0299",
        "scene-0300"]

test = \
    []

mini_train = \
    ['scene-0001', 'scene-0160', 'scene-0240']

mini_val = \
    ['scene-0292'] # 


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    # assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val,
                    'train_detect': train_detect, 'train_track': train_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


def get_scenes_of_split(split_name: str, nusc : NuScenes, verbose: bool = False) -> List[str]:
    """
    Returns the scenes in a given split.
    :param split_name: The name of the split.
    :param nusc: The NuScenes instance to know where to look up potential custom splits.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A list of scenes in that split.
    """

    if is_predefined_split(split_name=split_name):
        return create_splits_scenes(verbose=verbose)[split_name]
    else:
        return get_scenes_of_custom_split(split_name=split_name, nusc=nusc)

def is_predefined_split(split_name: str) -> bool:
    """
    Returns whether the split name is one of the predefined splits in the nuScenes dataset.
    :param split_name: The name of the split.
    :return: Whether the split is predefined.
    """
    return split_name in create_splits_scenes().keys()


def get_scenes_of_custom_split(split_name: str, nusc : NuScenes) -> List[str]:
    """Returns the scene names from a custom `splits.json` file."""

    splits_file_path: str = _get_custom_splits_file_path(nusc)

    splits_data: dict = {}
    with open(splits_file_path, 'r') as file:
        splits_data = json.load(file)

    if split_name not in splits_data.keys():
        raise ValueError(f"Custom split {split_name} requested, but not found in {splits_file_path}.")

    scene_names_of_split : List[str] = splits_data[split_name]
    assert isinstance(scene_names_of_split, list), \
        f'Custom split {split_name} must be a list of scene names in {splits_file_path}.'
    return scene_names_of_split


def _get_custom_splits_file_path(nusc : NuScenes) -> str:
    """Use a separate function for this so we can mock it well in unit tests."""

    splits_file_path: str = os.path.join(nusc.dataroot, nusc.version, "splits.json")
    if (not os.path.exists(splits_file_path)) or (not os.path.isfile(splits_file_path)):
        raise ValueError(f"Custom split requested, but no valid file found at {splits_file_path}.")

    return splits_file_path


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
