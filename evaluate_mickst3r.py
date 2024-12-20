"""
Built to be similar to https://github.com/nianticlabs/mickey/blob/main/submission.py
Can be run as -> python evaluate_mickst3r.py --config config/prob_pose.yaml --checkpoint checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --split val --o results/<experiment_name>
"""

import argparse
from natsort import natsorted
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import torch
import numpy as np
from tqdm import tqdm

from config.default import cfg
from mast3r_src.model import AsymmetricMASt3R
from src.datasets.utils import read_intrinsics
from dust3r_src.dust3r.utils.image import load_images
from dust3r_src.dust3r.inference import inference
from mickey_src.lib.utils.data import data_to_model_device

from src.diff_downsample_maps import downsample_maps_w_kpts
from src.diff_probabilistic_procrustes import e2eProbabilisticProcrustesSolver
from src.diff_compute_correspondences import ComputeCorrespondences

from transforms3d.quaternions import mat2quat


@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {"float": lambda v: f"{v:.6f}"}
        max_line_width = 1000
        q_str = np.array2string(
            self.q, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        t_str = np.array2string(
            self.t, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        return f"{self.image_name} {q_str} {t_str} {self.inliers}"


def predict(loader, model):
    results_dict = defaultdict(list)

    max_batch_size = loader["BATCH_SIZE"]
    print(f"Running inference with batch size: {max_batch_size}")

    device = next(model.parameters()).device

    scenes = natsorted(loader["SCENES"].keys())
    for scene in scenes:
        print(f"Processing scene: {scene}")
        intrinsics_path = loader["SCENES"][scene]["intrinsics_path"]
        K = read_intrinsics(
            intrinsics_path, resize=(384, 512)
        )  # resize instrinsics to model input size

        query_image_paths = loader["SCENES"][scene]["query_image_paths"]
        ref_image_path = loader["SCENES"][scene]["ref_image_path"]

        ref_image = load_images([ref_image_path], size=512, verbose=False)[0]
        # split query images into batches
        query_image_batches = [
            query_image_paths[i : i + max_batch_size]
            for i in range(0, len(query_image_paths), max_batch_size)
        ]

        for query_image_batch in tqdm(query_image_batches):
            curr_batch_size = len(query_image_batch)

            query_images = load_images(query_image_batch, size=512, verbose=False)

            pairs = [(ref_image, query_image) for query_image in query_images]
            pairs_rev = [(query_image, ref_image) for query_image in query_images]

            output_1 = inference(pairs, model, device, curr_batch_size, verbose=False)
            output_2 = inference(
                pairs_rev, model, device, curr_batch_size, verbose=False
            )

            _, pred1 = output_1["view1"], output_1["pred1"]
            _, pred2 = output_2["view1"], output_2["pred1"]

            data_batch = downsample_maps_w_kpts(
                pred1, pred2, target_size=(51, 38), conf_type="desc_conf", device=device
            )

            # for K_color_0 read intrinsics of reference image and create a 8, 3, 3 tensor
            K_color0 = (
                torch.from_numpy(K[f"seq0/{Path(ref_image_path).stem}.jpg"])
                .unsqueeze(0)
                .to(device)
            )
            K_color0_batch = K_color0.repeat(curr_batch_size, 1, 1)

            # for K_color_1 read intrinsics of all 8 query image and create a 8, 3, 3 tensor
            K_color1_batch = torch.stack(
                [
                    torch.from_numpy(K[f"seq1/{Path(image_q_path).stem}.jpg"])
                    for image_q_path in query_image_batch
                ]
            ).to(device)

            data_batch["K_color0"] = K_color0_batch
            data_batch["K_color1"] = K_color1_batch

            compute_Correspondences = ComputeCorrespondences(cfg, device)
            e2e_Procrustes = e2eProbabilisticProcrustesSolver(cfg)

            data_batch = compute_Correspondences.prepare_data(data_batch)
            R_batched, t_batched, inliers_batched, _ = e2e_Procrustes.estimate_pose(
                data_batch, return_inliers=True
            )

            for i_batch in range(curr_batch_size):
                R = R_batched[i_batch].unsqueeze(0).detach().cpu().numpy()
                t = t_batched[i_batch].reshape(-1).detach().cpu().numpy()
                inliers = inliers_batched[i_batch].item()

                query_img = query_image_batch[i_batch]

                # ignore frames without poses (e.g. not enough feature matches)
                if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                    continue

                # populate results_dict
                estimated_pose = Pose(
                    image_name=query_img,
                    q=mat2quat(R).reshape(-1),
                    t=t.reshape(-1),
                    inliers=inliers,
                )
                results_dict[scene].append(estimated_pose)

    return results_dict


def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, "w") as zip:
        for scene, poses in results_dict.items():
            poses_str = "\n".join((str(pose) for pose in poses))
            zip.writestr(f"pose_{scene}.txt", poses_str.encode("utf-8"))


def eval(args):
    # Load configs
    cfg.merge_from_file("config/mapfree.yaml")
    cfg.merge_from_file(args.config)

    # HACK: Currently a dummy in place of dataloader, taking advantage of collate_fn
    if args.split == "test":
        BATCH_SIZE = 16
        DOWNSAMPLE = 5
    elif args.split == "val":
        BATCH_SIZE = 16
        DOWNSAMPLE = 5
    else:
        raise ValueError(f"Invalid split: {args.split}")

    scene_path_root = Path(cfg.DATASET.DATA_ROOT) / args.split
    scene_paths = natsorted(list(scene_path_root.glob("*")))
    print(
        f"Using {args.split} split |Downsample: {DOWNSAMPLE} |Number of scenes: {len(scene_paths)}"
    )

    # create a dataloader thats a dict containing 2 things essentially
    dataloader = {"BATCH_SIZE": BATCH_SIZE, "SCENES": {}}

    for scene_path in scene_paths:
        intrinsics_path = str(scene_path / "intrinsics.txt")

        ref_image_folder = scene_path / "seq0"
        ref_image_path = str(ref_image_folder / "frame_00000.jpg")

        query_image_folder = scene_path / "seq1"
        query_image_paths = natsorted(
            list(str(p) for p in query_image_folder.glob("*.jpg"))
        )[::DOWNSAMPLE]

        # write the dataloader dict
        dataloader["SCENES"][scene_path.name] = {
            "intrinsics_path": intrinsics_path,
            "ref_image_path": ref_image_path,
            "query_image_paths": query_image_paths,
        }

    # Create model and place on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(device)

    # Get predictions from model
    results_dict = predict(dataloader, model)

    # # Save predictions to txt per scene within zip
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_submission(results_dict, args.output_root / "submission.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    parser.add_argument(
        "--checkpoint",
        help="path to model checkpoint (models with learned parameters)",
        default="",
    )
    parser.add_argument("--output_root", "-o", type=Path, default=Path("results/"))
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to use for evaluation. Choose from test or val. Default: test",
    )
    args = parser.parse_args()
    eval(args)
