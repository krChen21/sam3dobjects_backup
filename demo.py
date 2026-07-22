# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"

# debug: model loading context (no functional change)
import torch

print("[debug] __file__:", __file__)
print("[debug] cwd:", os.getcwd())
print("[debug] config path absolute:", str(Path(config_path).resolve()))
print("[debug] sys.path[:5]:", sys.path[:5])
print("[debug] Inference class module:", Inference.__module__)
print("[debug] DINO_LOCAL_REPO env:", os.environ.get("DINO_LOCAL_REPO"))
print("[debug] HF_HOME env:", os.environ.get("HF_HOME"))
print("[debug] TORCH_HOME env:", os.environ.get("TORCH_HOME"))
print("[debug] TORCH_HUB_OFFLINE env:", os.environ.get("TORCH_HUB_OFFLINE"))
print("[debug] torch hub dir:", torch.hub.get_dir())

inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
