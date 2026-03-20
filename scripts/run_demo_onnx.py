"""ONNX Runtime inference demo for Fast-FoundationStereo.

Supports two modes (auto-detected):
  1. End-to-end: single foundation_stereo.onnx  (from make_onnx_end2end.py)
  2. Two-stage:  feature_runner.onnx + post_runner.onnx  (from make_onnx.py)

Usage:
    uv run python scripts/run_demo_onnx.py \
        --onnx_dir output/onnx_e2e \
        --left_file demo_data/left.png \
        --right_file demo_data/right.png \
        --intrinsic_file demo_data/K.txt \
        --out_dir stereo_output \
        --remove_invisible 1 \
        --denoise_cloud 1 \
        --get_pc 1 \
        --zfar 100 \
        --show 1
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import time

import cv2
import imageio.v3 as iio
import numpy as np
import onnxruntime as ort
import yaml

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ---------------------------------------------------------------------------
# Utility helpers (replicated from project Utils.py to stay self-contained)
# ---------------------------------------------------------------------------


def set_logging_format(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s", datefmt="%m-%d|%H:%M:%S")


def vis_disparity(
    disp: np.ndarray,
    min_val: float | None = None,
    max_val: float | None = None,
    invalid_thres: float = np.inf,
    color_map: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    disp = disp.copy()
    H, W = disp.shape[:2]
    invalid_mask = disp >= invalid_thres
    if (invalid_mask == 0).sum() == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)
    if min_val is None:
        min_val = disp[~invalid_mask].min()
    if max_val is None:
        max_val = disp[~invalid_mask].max()
    vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[..., ::-1]
    if invalid_mask.any():
        vis[invalid_mask] = 0
    return vis.astype(np.uint8)


def depth2xyzmap(depth: np.ndarray, K: np.ndarray, zmin: float = 0.1) -> np.ndarray:
    invalid_mask = depth < zmin
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(H), np.arange(W), sparse=False, indexing="ij")
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs, ys, zs), axis=1)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a PLY point cloud without open3d."""
    N = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if colors is not None:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            f.write(struct.pack("<fff", *points[i].astype(np.float32)))
            if colors is not None:
                r, g, b = colors[i].astype(np.uint8)
                f.write(struct.pack("BBB", r, g, b))


def to_open3d_cloud(
    points: np.ndarray, colors: np.ndarray | None = None
) -> "o3d.geometry.PointCloud":
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


# ---------------------------------------------------------------------------
# GWC volume construction (pure numpy, mirrors the PyTorch implementation)
# ---------------------------------------------------------------------------


def build_gwc_volume_numpy(
    ref_feat: np.ndarray,
    tar_feat: np.ndarray,
    max_disp: int,
    num_groups: int,
    normalize: bool = True,
) -> np.ndarray:
    """Build group-wise correlation volume.

    Args:
        ref_feat:  (B, C, H, W) float32
        tar_feat:  (B, C, H, W) float32
        max_disp:  maximum disparity (at 1/4 resolution)
        num_groups: number of correlation groups
        normalize: L2-normalize each group before dot product

    Returns:
        volume: (B, num_groups, max_disp, H, W) float32
    """
    B, C, H, W = ref_feat.shape
    K = C // num_groups

    ref = ref_feat.reshape(B, num_groups, K, H, W)
    tar = tar_feat.reshape(B, num_groups, K, H, W)

    if normalize:
        ref_norm = np.linalg.norm(ref, axis=2, keepdims=True).clip(min=1e-8)
        tar_norm = np.linalg.norm(tar, axis=2, keepdims=True).clip(min=1e-8)
        ref = ref / ref_norm
        tar = tar / tar_norm

    volume = np.zeros((B, num_groups, max_disp, H, W), dtype=np.float32)
    for d in range(max_disp):
        if d == 0:
            volume[:, :, d, :, :] = (ref * tar).sum(axis=2)
        else:
            volume[:, :, d, :, d:] = (ref[:, :, :, :, d:] * tar[:, :, :, :, :-d]).sum(
                axis=2
            )
    return volume


# ---------------------------------------------------------------------------
# ONNX Runtime session helpers
# ---------------------------------------------------------------------------


def create_session(onnx_path: str) -> ort.InferenceSession:
    providers = []
    if "TensorrtExecutionProvider" in ort.get_available_providers():
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_max_workspace_size": 5 * 1024 * 1024 * 1024,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "trt_engine_cache",
                    "trt_engine_cache_prefix": "ffs",
                    "trt_detailed_build_log": False,
                },
            )
        )
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    logging.info(f"Loading ONNX model: {onnx_path}  providers={providers}")
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def run_session(
    sess: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, inputs)
    return dict(zip(output_names, results))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description="ONNX Runtime inference for Fast-FoundationStereo"
    )
    parser.add_argument(
        "--onnx_dir",
        default=f"{code_dir}/../output/onnx_e2e",
        type=str,
        help="Directory containing ONNX model(s) and onnx.yaml",
    )
    parser.add_argument(
        "--left_file", default=f"{code_dir}/../demo_data/left.png", type=str
    )
    parser.add_argument(
        "--right_file", default=f"{code_dir}/../demo_data/right.png", type=str
    )
    parser.add_argument(
        "--intrinsic_file",
        default=f"{code_dir}/../demo_data/K.txt",
        type=str,
        help="Camera intrinsic matrix and baseline file",
    )
    parser.add_argument("--out_dir", default="stereo_output", type=str)
    parser.add_argument("--remove_invisible", default=1, type=int)
    parser.add_argument("--denoise_cloud", default=0, type=int)
    parser.add_argument("--denoise_nb_points", type=int, default=30)
    parser.add_argument("--denoise_radius", type=float, default=0.03)
    parser.add_argument("--get_pc", type=int, default=1, help="Save point cloud output")
    parser.add_argument(
        "--zfar", type=float, default=100, help="Max depth to include in point cloud"
    )
    parser.add_argument(
        "--show",
        type=int,
        default=1,
        help="Show disparity and point cloud visualization",
    )
    args = parser.parse_args()

    set_logging_format()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load config from onnx.yaml ----
    yaml_path = os.path.join(args.onnx_dir, "onnx.yaml")
    if not os.path.isfile(yaml_path):
        yaml_path = os.path.join(os.path.dirname(args.onnx_dir), "onnx.yaml")
    with open(yaml_path, "r") as f:
        cfg: dict = yaml.safe_load(f)
    logging.info(f"Config: {cfg}")

    end2end = cfg.get("end2end", False)
    e2e_path = os.path.join(args.onnx_dir, "foundation_stereo.onnx")
    if not end2end and os.path.isfile(e2e_path):
        end2end = True

    # ---- Detect mode & create ONNX session(s) ----
    if end2end:
        logging.info("=== End-to-end mode (single ONNX) ===")
        sess = create_session(e2e_path)
        input_shape = sess.get_inputs()[0].shape  # [1, 3, H, W]
        image_size = [input_shape[2], input_shape[3]]
    else:
        logging.info("=== Two-stage mode (feature_runner + post_runner) ===")
        feature_sess = create_session(
            os.path.join(args.onnx_dir, "feature_runner.onnx")
        )
        post_sess = create_session(os.path.join(args.onnx_dir, "post_runner.onnx"))
        input_shape = feature_sess.get_inputs()[0].shape
        image_size = [input_shape[2], input_shape[3]]

    max_disp: int = cfg["max_disp"]
    cv_group: int = cfg.get("cv_group", 8)
    normalize: bool = cfg.get("normalize", True)

    # ---- Load & preprocess images ----
    img0 = iio.imread(args.left_file)
    img1 = iio.imread(args.right_file)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    fx = image_size[1] / img0.shape[1]
    fy = image_size[0] / img0.shape[0]
    if fx != 1 or fy != 1:
        logging.info(
            f"WARNING: resizing image to {image_size}, fx={fx:.4f}, fy={fy:.4f}"
        )
    img0 = cv2.resize(img0, None, fx=fx, fy=fy)
    img1 = cv2.resize(img1, None, fx=fx, fy=fy)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()
    logging.info(f"Image size after resize: {H}x{W}")
    iio.imwrite(os.path.join(args.out_dir, "left.png"), img0)
    iio.imwrite(os.path.join(args.out_dir, "right.png"), img1)

    left_input = img0.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
    right_input = img1.astype(np.float32).transpose(2, 0, 1)[np.newaxis]

    # ---- Inference ----
    t0 = time.perf_counter()

    if end2end:
        logging.info("Running end-to-end inference ...")
        out = run_session(sess, {"left": left_input, "right": right_input})
        disp = out["disp"]
    else:
        logging.info("Running feature_runner ...")
        feat_out = run_session(feature_sess, {"left": left_input, "right": right_input})
        logging.info(f"features_left_04 shape: {feat_out['features_left_04'].shape}")

        logging.info("Building GWC volume ...")
        gwc_volume = build_gwc_volume_numpy(
            feat_out["features_left_04"].astype(np.float32),
            feat_out["features_right_04"].astype(np.float32),
            max_disp // 4,
            cv_group,
            normalize=normalize,
        )
        logging.info(f"GWC volume shape: {gwc_volume.shape}")

        logging.info("Running post_runner ...")
        post_inputs = {
            "features_left_04": feat_out["features_left_04"],
            "features_left_08": feat_out["features_left_08"],
            "features_left_16": feat_out["features_left_16"],
            "features_left_32": feat_out["features_left_32"],
            "features_right_04": feat_out["features_right_04"],
            "stem_2x": feat_out["stem_2x"],
            "gwc_volume": gwc_volume,
        }
        post_out = run_session(post_sess, post_inputs)
        disp = post_out["disp"]

    elapsed = time.perf_counter() - t0
    logging.info(f"Inference done in {elapsed:.3f}s")

    disp = disp.reshape(H, W).clip(0, None) * (1.0 / fx)

    # ---- Visualize disparity ----
    vis = vis_disparity(disp)
    vis_concat = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    iio.imwrite(os.path.join(args.out_dir, "disp_vis.png"), vis_concat)
    logging.info(f"Disparity visualization saved to {args.out_dir}/disp_vis.png")

    if args.show:
        s = 1280 / vis_concat.shape[1]
        resized = cv2.resize(
            vis_concat, (int(vis_concat.shape[1] * s), int(vis_concat.shape[0] * s))
        )
        cv2.imshow("disp", resized[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---- Remove invisible points ----
    if args.remove_invisible:
        _, xx = np.meshgrid(
            np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
        )
        us_right = xx - disp
        disp[us_right < 0] = np.inf

    # ---- Point cloud ----
    if args.get_pc:
        with open(args.intrinsic_file, "r") as f:
            lines = f.readlines()
            K = np.array(
                list(map(float, lines[0].strip().split())), dtype=np.float32
            ).reshape(3, 3)
            baseline = float(lines[1])
        K[0] *= fx
        K[1] *= fy
        depth = K[0, 0] * baseline / disp
        np.save(os.path.join(args.out_dir, "depth_meter.npy"), depth)

        xyz_map = depth2xyzmap(depth, K)
        pts = xyz_map.reshape(-1, 3)
        colors = img0_ori.reshape(-1, 3)
        keep = (pts[:, 2] > 0) & (pts[:, 2] <= args.zfar)
        pts = pts[keep]
        colors = colors[keep]

        if o3d is not None:
            pcd = to_open3d_cloud(pts, colors)
            o3d.io.write_point_cloud(os.path.join(args.out_dir, "cloud.ply"), pcd)
            logging.info(
                f"Point cloud saved to {args.out_dir}/cloud.ply  ({len(pts)} points)"
            )

            if args.denoise_cloud:
                logging.info("Denoising point cloud ...")
                pcd = pcd.voxel_down_sample(voxel_size=0.001)
                _, ind = pcd.remove_radius_outlier(
                    nb_points=args.denoise_nb_points, radius=args.denoise_radius
                )
                pcd = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(
                    os.path.join(args.out_dir, "cloud_denoise.ply"), pcd
                )

            if args.show:
                logging.info("Visualizing point cloud. Press ESC to exit.")
                viewer = o3d.visualization.Visualizer()
                viewer.create_window()
                viewer.add_geometry(pcd)
                viewer.get_render_option().point_size = 1.0
                viewer.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
                ctr = viewer.get_view_control()
                ctr.set_front([0, 0, -1])
                closest_id = np.asarray(pcd.points)[:, 2].argmin()
                ctr.set_lookat(np.asarray(pcd.points)[closest_id])
                ctr.set_up([0, -1, 0])
                viewer.run()
                viewer.destroy_window()
        else:
            save_ply(os.path.join(args.out_dir, "cloud.ply"), pts, colors)
            logging.info(
                f"Point cloud saved to {args.out_dir}/cloud.ply  ({len(pts)} points, open3d not available, using built-in PLY writer)"
            )
            if args.denoise_cloud:
                logging.warning("Denoise requires open3d, skipping.")
            if args.show:
                logging.warning("3D visualization requires open3d, skipping.")


if __name__ == "__main__":
    main()
