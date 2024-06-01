import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace

import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
from torch.cuda.amp import autocast

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis
from src.dataset.postprocessing import rm_dust_fh

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["val","train","test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")
    print(args.config)

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
    save_folder.mkdir(parents=True, exist_ok=True)

    with (save_folder / 'args.txt').open('w') as f:
        print(vars(args), file=f)

    args_list = []
    for config in args.config:
        config_file = pathlib.Path(config).resolve()
        print(config_file)
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)
            # set default normalisation
            if not hasattr(old_args, "normalisation"):
                old_args.normalisation = "minmax"
        print(old_args)
        args_list.append(old_args)

    if args.on == "test":
        args.pred_folder = save_folder / f"test_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    elif args.on == "val":
        args.pred_folder = save_folder / f"validation_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    else:
        args.pred_folder = save_folder / f"training_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)

    # Create model

    models_list = []
    normalisations_list = []
    for model_args in args_list:
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)

        model = model_maker(
            4, 3,
            width=model_args.width, deep_supervision=model_args.deep_sup,
            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        print(f"Creating {model_args.arch}")

        reload_ckpt_bis(str(model_args.ckpt), model)
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")
        print(model)

    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch:", len(loader_minmax))
    generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list, args)


# def generate_segmentations(data_loaders, models, normalisations, args):
#     # TODO: try reuse the function used for train...
#     for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
#         patient_id = batch_minmax["patient_id"][0]
#         ref_img_path = batch_minmax["seg_path"][0]
#         crops_idx_minmax = batch_minmax["crop_indexes"]
#         crops_idx_zscore = batch_zscore["crop_indexes"]
#         inputs_minmax = batch_minmax["image"]
#         inputs_zscore = batch_zscore["image"]
#         inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
#         inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
#         model_preds = []
#         last_norm = None
#         for model, normalisation in zip(models, normalisations):
#             if normalisation == last_norm:
#                 pass
#             elif normalisation == "minmax":
#                 inputs = inputs_minmax.cuda()
#                 pads = pads_minmax
#                 crops_idx = crops_idx_minmax
#             elif normalisation == "zscore":
#                 inputs = inputs_zscore.cuda()
#                 pads = pads_zscore
#                 crops_idx = crops_idx_zscore
#             model.cuda()  # go to gpu
#             with autocast():
#                 with torch.no_grad():
#                     if args.tta:
#                         pre_segs = apply_simple_tta(model, inputs, True)
#                         model_preds.append(pre_segs)
#                     else:
#                         if model.deep_supervision:
#                             pre_segs, _ = model(inputs)
#                         else:
#                             pre_segs = model(inputs)
#                         pre_segs = pre_segs.sigmoid_().cpu()
#                     # remove pads
#                     maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - \
#                                        pads[2]
#                     pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
#                     print("pre_segs size", pre_segs.shape)
#                     segs = torch.zeros((1, 3, 155, 240, 240))
#                     segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
#                     print("segs size", segs.shape)

#                     model_preds.append(segs)
#             model.cpu()  # free for the next one

            
#         pre_segs = torch.stack(model_preds).mean(dim=0)
# # ----------------------------------
#         segs = pre_segs[0].numpy() > 0.5

#         et = segs[0]
#         net = np.logical_and(segs[1], np.logical_not(et))
#         ed = np.logical_and(segs[2], np.logical_not(segs[1]))
#         labelmap = np.zeros(segs[0].shape)
#         labelmap[et] = 3
#         labelmap[net] = 1
#         labelmap[ed] = 2
#         labelmap = sitk.GetImageFromArray(labelmap)
#         ref_img = sitk.ReadImage(ref_img_path)
#         labelmap.CopyInformation(ref_img)
#         pre_segs = rm_dust_fh(labelmap)
#         print(f"Writing {str(args.pred_folder)}/{patient_id}.nii.gz")
#         sitk.WriteImage(labelmap, f"{str(args.pred_folder)}/{patient_id}.nii.gz")

def reload_ckpt_bis(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    # Print the keys to debug
    print("Checkpoint keys:", checkpoint.keys())

    # Print the type and keys of state_dict to ensure it contains the model's state
    if 'state_dict' in checkpoint:
        print("Loading state_dict")
        state_dict = checkpoint['state_dict']
    else:
        raise KeyError("No valid state dict found in checkpoint")

    print("State dict keys:", state_dict.keys())
    model.load_state_dict(state_dict)

def generate_segmentations(data_loaders, models, normalisations, args):
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]
        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]
        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
        model_preds = []
        last_norm = None
        for model, normalisation in zip(models, normalisations):
            if normalisation == last_norm:
                pass
            elif normalisation == "minmax":
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
                    if args.tta:
                        pre_segs = apply_simple_tta(model, inputs, True)
                    else:
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)
                        pre_segs = pre_segs.sigmoid_().cpu()
                        if pre_segs is not None:
                            maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
                            pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                            segs = torch.zeros((1, 3, 155, 240, 240))
                            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                            model_preds.append(segs)
            model.cpu()

        # Before stacking, check if model_preds is not empty
        if not model_preds:
            print(f"No predictions for patient {patient_id}. Model predictions are empty.")
            continue

        pre_segs = torch.stack(model_preds).mean(dim=0)
        segs = pre_segs[0].numpy() > 0.5
        # Create label map
        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 3
        labelmap[net] = 1
        labelmap[ed] = 2

        # Postprocess the segmentation to remove dust and fill holes (Assuming `rm_dust_fh` is defined)
        for label in range(1, 4):  # Assuming labels 1, 2, 3
            mask = labelmap == label
            processed_mask = rm_dust_fh(mask)
            labelmap[mask] = 0
            labelmap[processed_mask] = label

        # Convert to SimpleITK image and copy metadata from reference image
        labelmap = sitk.GetImageFromArray(labelmap.astype(np.uint8))
        try:
            ref_img = sitk.ReadImage(ref_img_path)
            labelmap.CopyInformation(ref_img)
        except RuntimeError as e:
            print(f"Error: {e}")
            print(f"Shape of labelmap: {labelmap.GetSize()}")
            print(f"Shape of reference segmentation image: {ref_img.GetSize()}")

        try:
            print(f"Writing {str(args.pred_folder)}/{patient_id}.nii.gz")
            sitk.WriteImage(labelmap, f"{str(args.pred_folder)}/{patient_id}.nii.gz")
        except RuntimeError as e:
            print(f"Error writing image: {e}")



if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
