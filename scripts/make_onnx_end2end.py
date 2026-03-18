"""Export a single end-to-end ONNX model for Fast-FoundationStereo.

Unlike make_onnx.py which exports two separate ONNX files (feature_runner +
post_runner) with a triton/numpy GWC volume in between, this script exports
ONE ONNX that takes (left, right) images and directly outputs the disparity.

Usage:
    uv run python scripts/make_onnx_end2end.py \
        --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
        --save_path output/onnx_e2e \
        --height 448 --width 640 \
        --valid_iters 8 \
        --max_disp 192 \
        --opset_version 11
"""

import os
import sys
import argparse
import logging

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{code_dir}/../")

import yaml
import onnx
import onnxslim
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from core.foundation_stereo import FastFoundationStereo, normalize_image
from core.submodule import (
    disparity_regression,
)
from core.geometry import Combined_Geo_Encoding_Volume
import Utils as U

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# ONNX-friendly ops (no Tensor.unfold / F.unfold — unsupported by ONNX)
# ---------------------------------------------------------------------------


def _build_shifted_volume(fea: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """Build a disparity-shifted volume without Tensor.unfold.

    Replaces the pattern:
        padded.unfold(3, W, 1) -> flip -> permute
    with explicit per-disparity slicing + stack, which ONNX fully supports.

    Args:
        fea: (B, C, H, W)
        maxdisp: number of disparity levels

    Returns:
        volume: (B, C, maxdisp, H, W)  where volume[:,:,d,:,d:] = fea[:,:,:,:-d]
    """
    B, C, H, W = fea.shape
    slices = []
    for d in range(maxdisp):
        if d == 0:
            slices.append(fea)
        else:
            pad = fea.new_zeros(B, C, H, d)
            slices.append(torch.cat([pad, fea[:, :, :, :-d]], dim=3))
    return torch.stack(slices, dim=2)


def build_gwc_volume_onnx(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
    num_groups: int,
    normalize: bool = True,
) -> torch.Tensor:
    """GWC volume using only ONNX-exportable PyTorch ops."""
    dtype = refimg_fea.dtype
    B, C, H, W = refimg_fea.shape
    channels_per_group = C // num_groups

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    target_volume = _build_shifted_volume(targetimg_fea, maxdisp)

    ref_volume = ref_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
    target_volume = target_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)

    if normalize:
        ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
        target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)

    cost_volume = (ref_volume * target_volume).sum(dim=2)
    return cost_volume.contiguous()


def build_concat_volume_onnx(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
) -> torch.Tensor:
    """Concat volume using only ONNX-exportable PyTorch ops."""
    B, C, H, W = refimg_fea.shape
    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    target_volume = _build_shifted_volume(targetimg_fea, maxdisp)
    volume = torch.cat((ref_volume, target_volume), dim=1)
    return volume.contiguous()


def context_upsample_onnx(
    disp_low: torch.Tensor, up_weights: torch.Tensor
) -> torch.Tensor:
    """ONNX-friendly replacement for context_upsample (avoids F.unfold).

    F.unfold(x, 3, 1, 1) on a (b,1,h,w) tensor extracts 3x3 patches →
    equivalent to 9 shifted copies. We use F.pad + slicing instead.
    """
    b, c, h, w = disp_low.shape
    padded = F.pad(disp_low, (1, 1, 1, 1), mode="constant", value=0)
    patches = []
    for dy in range(3):
        for dx in range(3):
            patches.append(padded[:, :, dy : dy + h, dx : dx + w])
    disp_unfold = torch.cat(patches, dim=1)  # (b, 9, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode="nearest")
    return (disp_unfold * up_weights).sum(1)


# ---------------------------------------------------------------------------
# End-to-end wrapper
# ---------------------------------------------------------------------------


class End2EndStereo(nn.Module):
    """Wraps FastFoundationStereo into a single ONNX-exportable forward pass.

    Input:  left (1,3,H,W), right (1,3,H,W)  float32 in [0,255]
    Output: disp (1,1,H,W) float32
    """

    def __init__(self, model: FastFoundationStereo):
        super().__init__()
        self.args = model.args
        self.dtype = model.dtype
        self.cv_group = model.cv_group

        self.feature = model.feature
        self.stem_2 = model.stem_2
        self.proj_cmb = model.proj_cmb
        self.corr_stem = model.corr_stem
        self.corr_feature_att = model.corr_feature_att
        self.cost_agg = model.cost_agg
        self.classifier = model.classifier
        self.cnet = model.cnet
        self.update_block = model.update_block
        self.sam = model.sam
        self.cam = model.cam
        self.spx_2_gru = model.spx_2_gru
        self.spx_gru = model.spx_gru
        self.register_buffer("dx", model.dx)

    def upsample_disp(
        self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor
    ) -> torch.Tensor:
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample_onnx(disp * 4.0, spx_pred).unsqueeze(1)
        return up_disp.to(self.dtype)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        B = left.shape[0]
        left_n = normalize_image(left)
        right_n = normalize_image(right)

        out = self.feature(torch.cat([left_n, right_n], dim=0))
        features_left = [o[:B] for o in out]
        features_right = [o[B:] for o in out]
        stem_2x = self.stem_2(left_n)

        maxdisp_quarter = self.args.max_disp // 4
        gwc_volume = build_gwc_volume_onnx(
            features_left[0],
            features_right[0],
            maxdisp_quarter,
            self.cv_group,
            normalize=self.args.get("normalize", True),
        )

        left_tmp = self.proj_cmb(features_left[0])
        right_tmp = self.proj_cmb(features_right[0])
        concat_volume = build_concat_volume_onnx(
            left_tmp, right_tmp, maxdisp=maxdisp_quarter
        )

        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        comb_volume = self.corr_stem(comb_volume)
        comb_volume = self.corr_feature_att(comb_volume, features_left[0])
        comb_volume = self.cost_agg(comb_volume, features_left)

        logits = self.classifier(comb_volume).squeeze(1)
        prob = F.softmax(logits, dim=1)
        init_disp = disparity_regression(prob, maxdisp_quarter)

        cnet_list = list(
            self.cnet(features_left[0], features_left[1], features_left[2])
        )
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]

        geo_fn = Combined_Geo_Encoding_Volume(
            features_left[0].to(self.dtype),
            features_right[0].to(self.dtype),
            comb_volume.to(self.dtype),
            num_levels=self.args.corr_levels,
        )
        b, c, h, w = features_left[0].shape
        coords = (
            torch.arange(w, dtype=torch.float, device=left.device)
            .reshape(1, 1, w, 1)
            .repeat(b, h, 1, 1)
        )
        disp = init_disp.to(self.dtype)

        for itr in range(self.args.valid_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, dx=self.dx, low_memory=True)
            net_list, mask_feat_4, delta_disp = self.update_block(
                net_list, inp_list, geo_feat.to(self.dtype), disp, att
            )
            disp = disp + delta_disp.to(self.dtype)

        disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
        return disp_up


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export end-to-end ONNX for Fast-FoundationStereo"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=f"{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"{code_dir}/../output/onnx_e2e",
        help="Directory to save the ONNX model and config",
    )
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--opset_version", type=int, default=17)
    args = parser.parse_args()

    assert args.height % 32 == 0 and args.width % 32 == 0, (
        "height and width must be divisible by 32"
    )
    os.makedirs(args.save_path, exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    logging.info(f"Loading model from {args.model_dir} ...")
    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()

    e2e = End2EndStereo(model).cuda().eval()

    left_img = torch.randn(1, 3, args.height, args.width, device="cuda").float() * 255
    right_img = torch.randn(1, 3, args.height, args.width, device="cuda").float() * 255

    logging.info("Running a test forward pass ...")
    with torch.amp.autocast("cuda", enabled=True, dtype=U.AMP_DTYPE):
        test_out = e2e(left_img, right_img)
    logging.info(f"Test output shape: {test_out.shape}")

    onnx_path = os.path.join(args.save_path, "foundation_stereo.onnx")
    logging.info(f"Exporting ONNX to {onnx_path} (opset {args.opset_version}) ...")

    with torch.amp.autocast("cuda", enabled=False, dtype=U.AMP_DTYPE):
        torch.onnx.export(
            e2e,
            (left_img, right_img),
            onnx_path,
            opset_version=args.opset_version,
            input_names=["left", "right"],
            output_names=["disp"],
            do_constant_folding=True,
            dynamo=False,
        )
    model_onnx = onnx.load(onnx_path)  # load onnx model
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, onnx_path)

    logging.info(f"ONNX model saved to {onnx_path}")

    cfg_out = OmegaConf.to_container(model.args)
    cfg_out["image_size"] = [args.height, args.width]
    cfg_out["end2end"] = True
    yaml_path = os.path.join(args.save_path, "onnx.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg_out, f)
    logging.info(f"Config saved to {yaml_path}")
    logging.info("Done!")
