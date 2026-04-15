# Optional config for better memory efficiency
import os

import cv2

# Required imports
import numpy as np
import open3d as o3d
import torch
import torchvision
import yaml
from libtiff import TIFF
from PIL import Image

from models import *


def absolute_file_paths(directory):
    files = os.listdir(directory)
    # Filtering only the files.
    files = [
        directory + "/" + f
        for f in files
        if os.path.isfile(directory + "/" + f)
        if f.lower().endswith((".tif"))
    ]
    return sorted(files)


def file_names(directory):
    files = os.listdir(directory)
    # Filtering only the files.
    files = [
        f.rsplit(".", 1)[0]
        for f in files
        if os.path.isfile(directory + "/" + f)
        if f.lower().endswith((".tif"))
    ]
    return sorted(files)


def read_yaml_pose(yaml_pose_file):
    with open(yaml_pose_file, "r") as file:
        camera_extrinsics_yaml = yaml.safe_load(file)

        # Parse 3x4 extrinsic matrix
        transform_matrix = np.array(
            [
                [
                    float(camera_extrinsics_yaml[0][0]),
                    float(camera_extrinsics_yaml[0][1]),
                    float(camera_extrinsics_yaml[0][2]),
                    float(camera_extrinsics_yaml[0][3]),
                ],
                [
                    float(camera_extrinsics_yaml[1][0]),
                    float(camera_extrinsics_yaml[1][1]),
                    float(camera_extrinsics_yaml[1][2]),
                    float(camera_extrinsics_yaml[1][3]),
                ],
                [
                    float(camera_extrinsics_yaml[2][0]),
                    float(camera_extrinsics_yaml[2][1]),
                    float(camera_extrinsics_yaml[2][2]),
                    float(camera_extrinsics_yaml[2][3]),
                ],
            ]
        )

    return transform_matrix


def read_yaml_calib(yaml_calib_file):
    with open(yaml_calib_file, "r") as file:
        camera_intrisics_yaml = yaml.safe_load(file)

        # Parse 3x3 calib matrix
        calib_matrix = [
            [
                float(camera_intrisics_yaml[0][0]),
                float(camera_intrisics_yaml[0][1]),
                float(camera_intrisics_yaml[0][2]),
            ],
            [
                float(camera_intrisics_yaml[1][0]),
                float(camera_intrisics_yaml[1][1]),
                float(camera_intrisics_yaml[1][2]),
            ],
            [
                float(camera_intrisics_yaml[2][0]),
                float(camera_intrisics_yaml[2][1]),
                float(camera_intrisics_yaml[2][2]),
            ],
        ]
        calib_matrix = np.array(calib_matrix, dtype=np.float32)

    return calib_matrix


def write_instrinsics_yaml(file_path, intrinsics):
    intrinsics_yaml = [
        [
            "{:.6f}".format(float(intrinsics[0][0])),
            "{:.6f}".format(float(intrinsics[0][1])),
            "{:.6f}".format(float(intrinsics[0][2])),
        ],
        [
            "{:.6f}".format(float(intrinsics[1][0])),
            "{:.6f}".format(float(intrinsics[1][1])),
            "{:.6f}".format(float(intrinsics[1][2])),
        ],
        [
            "{:.6f}".format(float(intrinsics[2][0])),
            "{:.6f}".format(float(intrinsics[2][1])),
            "{:.6f}".format(float(intrinsics[2][2])),
        ],
    ]
    with open(
        file_path,
        "w",
    ) as file:
        yaml.dump(intrinsics_yaml, file, default_flow_style=None)


def write_extrinsics_yaml(file_path, extrinsics):
    extrinsics_yaml = [
        [
            "{:.6f}".format(float(extrinsics[0][0])),
            "{:.6f}".format(float(extrinsics[0][1])),
            "{:.6f}".format(float(extrinsics[0][2])),
            "{:.6f}".format(float(extrinsics[0][3])),
        ],
        [
            "{:.6f}".format(float(extrinsics[1][0])),
            "{:.6f}".format(float(extrinsics[1][1])),
            "{:.6f}".format(float(extrinsics[1][2])),
            "{:.6f}".format(float(extrinsics[1][3])),
        ],
        [
            "{:.6f}".format(float(extrinsics[2][0])),
            "{:.6f}".format(float(extrinsics[2][1])),
            "{:.6f}".format(float(extrinsics[2][2])),
            "{:.6f}".format(float(extrinsics[2][3])),
        ],
    ]

    with open(file_path, "w") as file:
        yaml.dump(extrinsics_yaml, file, default_flow_style=None)


def read_yaml_parameter_sets(yaml_parameter_sets_file):
    with open(yaml_parameter_sets_file, "r") as file:
        yaml_parameter_sets_yaml = yaml.safe_load(file)

    return yaml_parameter_sets_yaml


from scipy.spatial.transform import RigidTransform as T
from scipy.spatial.transform import Rotation as R


def extrinsics_micmac_to_colmap(extrinsics_micmac):
    print(extrinsics_micmac)

    rotation_micmac = R.from_matrix(extrinsics_micmac[0:3, 0:3])

    rotation_colmap = rotation_micmac.inv()

    translation_micmac = extrinsics_micmac[:, 3]

    translation_colmap = rotation_colmap.apply(translation_micmac)

    extrinsics_colmap = T.from_components(
        translation_colmap, rotation_colmap
    ).as_matrix()[0:3, 0:4]

    print(extrinsics_colmap)
    return extrinsics_colmap


def run(parameter_set_file="", dataset_dir="", results_dir=""):
    # Read parameter set
    yaml_parameter_sets = read_yaml_parameter_sets(
        os.path.dirname(os.path.abspath(__file__)) + "/parameter_sets.yaml"
    )
    selected_yaml_parameter_set = None
    for yaml_parameter_set in yaml_parameter_sets:
        if yaml_parameter_set["name"] == parameter_set_file:
            selected_yaml_parameter_set = yaml_parameter_set["parameters"]

    # Get inference device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    torch.set_num_threads(10)

    # Load and preprocess example images (replace with your own image paths)
    image_paths = absolute_file_paths(dataset_dir + "/images")

    # Load and preprocess example images (replace with your own image paths)
    image_names = file_names(dataset_dir + "/images")

    reference_intrinsics_list = []
    reference_extrinsics_list = []
    image_idx = 0
    origin_extrinsics = None
    images = []
    torch_projection_tensor_array = []
    torch_depth_tensor_array = []
    depth_array = []
    for image_path in image_paths:
        # Read reference intrinsics
        reference_intrinsics = read_yaml_calib(
            dataset_dir + f"/camera_intrinsics/{image_names[image_idx]}.yaml"
        )
        reference_intrinsics_list.append(reference_intrinsics.astype(np.float64))

        # Read reference extrinsics
        reference_extrinsics_micmac = read_yaml_pose(
            f"{dataset_dir}/camera_extrinsics/{image_names[image_idx]}.yaml"
        )

        reference_extrinsics = reference_extrinsics_micmac

        reference_extrinsics = np.append(reference_extrinsics, [[0, 0, 0, 1]], axis=0)
        if image_idx == 0:
            origin_extrinsics = reference_extrinsics.copy()

        reference_extrinsics[0][3] = (
            reference_extrinsics[0][3] - origin_extrinsics[0][3]
        )
        reference_extrinsics[1][3] = (
            reference_extrinsics[1][3] - origin_extrinsics[1][3]
        )
        reference_extrinsics[2][3] = (
            reference_extrinsics[2][3] - origin_extrinsics[2][3]
        )

        # Change extrinsics to world_to_cam
        reference_extrinsics = np.linalg.inv(reference_extrinsics)
        reference_extrinsics_list.append(reference_extrinsics.astype(np.float64))

        image_idx = image_idx + 1

        # Add image to array
        #
        image = Image.open(image_path)
        image = torchvision.transforms.ToTensor()(image)
        images.append(image)

        torch_extrinsics = torch.tensor(reference_extrinsics.astype(np.float64))

        torch_intrinsics = torch.tensor(
            np.append(
                np.append(
                    reference_intrinsics.astype(np.float64), [[0, 0, 0]], axis=0
                ).transpose(),
                [[0, 0, 0, 1]],
                axis=0,
            ).transpose()
        )

        torch_projection = torch.stack((torch_extrinsics, torch_intrinsics))
        torch_projection_tensor_array.append(torch_projection)

        torch_depth_tensor = torch.tensor([np.linspace(1350, 1450, num=2000)])
        torch_depth_tensor_array.append(torch_depth_tensor)

        print(torch_projection.size())

    print("torch_projection_tensor_array")
    print(torch_projection_tensor_array)

    torch_projections_tensor = (
        torch.stack(torch_projection_tensor_array)
        .reshape([1, 4, 2, 4, 4])
        .type(torch.float64)
    )

    print("torch_projections_tensor")
    print(torch_projections_tensor)

    stage2_pjmats = (
        torch.stack(torch_projection_tensor_array.copy())
        .reshape([1, 4, 2, 4, 4])
        .type(torch.float64)
    )
    stage2_pjmats[:, :, 1, :2, :] = torch_projections_tensor[:, :, 1, :2, :] / 2.0

    stage3_pjmats = (
        torch.stack(torch_projection_tensor_array.copy())
        .reshape([1, 4, 2, 4, 4])
        .type(torch.float64)
    )
    stage3_pjmats[:, :, 1, :2, :] = torch_projections_tensor[:, :, 1, :2, :] / 4.0

    proj_matrices_ms = {
        "stage1": stage3_pjmats,
        "stage2": stage2_pjmats,
        "stage3": torch_projections_tensor,
    }

    print(proj_matrices_ms)

    torch_depths_tensor = torch.stack(torch_depth_tensor_array)

    images = np.array(images).reshape([1, 4, 3, 832, 1312])
    print(images.shape)
    images = torch.tensor(images)
    print(images.size())

    # model
    model = TransMVSNet(
        refine=False,
        ndepths=[48, 32, 8],
        depth_interals_ratio=[4.0, 2.0, 1.0],
        cr_base_chs=[8, 8, 8],
    )

    # load checkpoint file specified by args.loadckpt
    state_dict = torch.load(
        "/home/AGonzalez-Admin/work/IGN/local/4_estimation/methods/transmvsnet/checkpoints/model_dtu.ckpt",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state_dict["model"], strict=False)
    # model = torch.nn.DataParallel(model)

    model.eval()

    print(len(images))
    print(torch_projections_tensor.size())

    with torch.no_grad():
        predictions = model(images, proj_matrices_ms, torch_depth_tensor)

    torch.save(
        predictions,
        f"{results_dir}/predictions.pt",
    )

    predictions = torch.load(
        f"{results_dir}/predictions.pt",
        weights_only=False,
    )

    print(predictions.keys())
    print(predictions["stage1"].keys())
    print(predictions["depth"].shape)

    for camera_idx in range(predictions["depth"].shape[0]):
        # Save depth to a TIFF file
        depth_image = predictions["depth"][camera_idx].cpu()

        tif = TIFF.open(
            f"{results_dir}/depth_images/data/{image_names[(camera_idx)]}.tif",
            mode="w",
        )
        tif.write_image(depth_image)

        # Save depth confidence to a TIFF file
        depth_confidence_image = predictions["photometric_confidence"][camera_idx].cpu()
        tif = TIFF.open(
            f"{results_dir}/depth_images/confidence/{image_names[(camera_idx)]}.tif",
            mode="w",
        )
        tif.write_image(depth_confidence_image)


import sys
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--parameter_set_file", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--results_dir", type=str)

    args = parser.parse_args(sys.argv[1:])

    print(args.parameter_set_file)
    print(args.dataset_dir)
    print(args.results_dir)

    run(args.parameter_set_file, args.dataset_dir, args.results_dir)
