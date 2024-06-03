import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace
import io
import glob
import yaml
import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import uvicorn
from torch.cuda.amp import autocast
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis
from src.dataset.postprocessing import rm_dust_fh

app = FastAPI()

models_list = []
normalisations_list = []

def load_models():
    # Define the directory path where the models are stored
    runs_dir = "D:/project/project/runs"

    # Find all subdirectories (each represents a model run)
    subdirs = [subdir for subdir in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, subdir))]

    for subdir in subdirs:
        subdir_path = os.path.join(runs_dir, subdir)
        # Find YAML configuration file
        config_file = glob.glob(os.path.join(subdir_path, "*.yaml"))
        if not config_file:
            print(f"No YAML configuration file found in {subdir_path}")
            continue
        config_file = config_file[0]

        # Load YAML configuration
        with open(config_file, "r") as file:
            model_args = yaml.safe_load(file)
            model_args = SimpleNamespace(**model_args, ckpt=os.path.join(subdir_path, "model_best.pth.tar"))

        # Print the loaded configuration for debugging
        print(f"Loaded configuration for {subdir_path}: {model_args}")

        # Load model
        model_maker = getattr(models, model_args.arch)
        model = model_maker(4, 3, width=model_args.width, deep_supervision=model_args.deep_sup,
                            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        reload_ckpt_bis(str(model_args.ckpt), model)

        # Append loaded model and normalization type to the global lists
        models_list.append(model)
        if hasattr(model_args, 'normalisation'):
            normalisations_list.append(model_args.normalisation)
        else:
            print(f"Warning: 'normalisation' attribute missing in {subdir_path}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/predict")
async def predict(t1n: UploadFile = File(...), t1c: UploadFile = File(...), t2w: UploadFile = File(...), t2f: UploadFile = File(...)):
    # Load the uploaded image files
    t1n_image = sitk.ReadImage(io.BytesIO(await t1n.read()), sitk.sitkFloat32)
    t1c_image = sitk.ReadImage(io.BytesIO(await t1c.read()), sitk.sitkFloat32)
    t2w_image = sitk.ReadImage(io.BytesIO(await t2w.read()), sitk.sitkFloat32)
    t2f_image = sitk.ReadImage(io.BytesIO(await t2f.read()), sitk.sitkFloat32)

    # Combine the images into a single input tensor
    images = [t1n_image, t1c_image, t2w_image, t2f_image]
    inputs_minmax = torch.tensor([sitk.GetArrayFromImage(img) for img in images]).unsqueeze(0).float()
    inputs_zscore = inputs_minmax.clone()  # Modify this based on actual preprocessing steps
    inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
    inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)

    crops_idx_minmax = [(0, inputs_minmax.size(2)), (0, inputs_minmax.size(3)), (0, inputs_minmax.size(4))]
    crops_idx_zscore = [(0, inputs_zscore.size(2)), (0, inputs_zscore.size(3)), (0, inputs_zscore.size(4))]

    model_preds = []
    for model, normalisation in zip(models_list, normalisations_list):
        if normalisation == "minmax":
            inputs = inputs_minmax.cuda()
            pads = pads_minmax
            crops_idx = crops_idx_minmax
        elif normalisation == "zscore":
            inputs = inputs_zscore.cuda()
            pads = pads_zscore
            crops_idx = crops_idx_zscore

        model.cuda()
        with autocast():
            with torch.no_grad():
                pre_segs = model(inputs).sigmoid_().cpu()
                maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
                pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                segs = torch.zeros((1, 3, 155, 240, 240))
                segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                model_preds.append(segs)

        model.cpu()

    if not model_preds:
        return JSONResponse(content={"error": "No predictions"}, status_code=500)

    pre_segs = torch.stack(model_preds).mean(dim=0)
    segs = pre_segs[0].numpy() > 0.5
    et = segs[0]
    net = np.logical_and(segs[1], np.logical_not(et))
    ed = np.logical_and(segs[2], np.logical_not(segs[1]))
    labelmap = np.zeros(segs[0].shape)
    labelmap[et] = 3
    labelmap[net] = 1
    labelmap[ed] = 2

    for label in range(1, 4):
        mask = labelmap == label
        processed_mask = rm_dust_fh(mask)
        labelmap[mask] = 0
        labelmap[processed_mask] = label

    labelmap = sitk.GetImageFromArray(labelmap.astype(np.uint8))
    labelmap.CopyInformation(t1n_image)

    # Save the labelmap to a temporary file
    temp_filename = "segmented_result.nii"
    sitk.WriteImage(labelmap, temp_filename)

    return FileResponse(temp_filename, media_type='application/octet-stream', filename=temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)
