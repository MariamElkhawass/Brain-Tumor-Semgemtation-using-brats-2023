import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace
import io
import glob
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import yaml
import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast
from werkzeug.utils import secure_filename

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.tta import apply_simple_tta
from src.models.layers import get_norm_layer
from src.utils import reload_ckpt_bis
from src.dataset.postprocessing import rm_dust_fh

app = Flask(__name__)


models_list = []
normalisations_list = []

app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'nii'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def load_models():
    print("Loading models...")

    runs_dir = "D:/project/project/runs"

    subdirs = [subdir for subdir in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, subdir))]
    print(f"Found subdirectories: {subdirs}")

    for subdir in subdirs:
        subdir_path = os.path.join(runs_dir, subdir)
        print(f"Processing subdirectory: {subdir_path}")

        config_file = glob.glob(os.path.join(subdir_path, "*.yaml"))
        if not config_file:
            print(f"No YAML configuration file found in {subdir_path}")
            continue
        config_file = config_file[0]

        with open(config_file, "r") as file:
            model_args = yaml.safe_load(file)
            model_args = SimpleNamespace(**model_args, ckpt=os.path.join(subdir_path, "model_best.pth.tar"))

        print(f"Loaded configuration for {subdir_path}: {model_args}")

        model_maker = getattr(models, model_args.arch)
        model = model_maker(4, 3, width=model_args.width, deep_supervision=model_args.deep_sup,
                            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        
        try:
            checkpoint = torch.load(model_args.ckpt)
            state_dict = checkpoint['state_dict']

            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict
            
            model.load_state_dict(new_state_dict)
            print(f"Model loaded successfully from {subdir_path}")

        except Exception as e:
            print(f"Error loading model from {subdir_path}: {e}")
            continue

        model = torch.nn.DataParallel(model)
        models_list.append(model)
        normalisations_list.append(model_args.norm_layer)
    
    print("Models loading completed.")

# @app.before_first_request
# def startup_event():
#     load_models()

with app.app_context():
    load_models()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/test-models", methods=['GET'])
def test_models():
    if models_list:
        return {"message": f"Models are loaded successfully: {len(models_list)} models loaded."}
    else:
        return {"message": "No models are loaded"}

@app.route('/predict', methods=['POST'])
def predict():
    if 't1n' not in request.files or 't1c' not in request.files or 't2w' not in request.files or 't2f' not in request.files:
        return "Missing file(s)", 400
    
    print("API accessed")
    
    t1n = request.files['t1n']
    t1c = request.files['t1c']
    t2w = request.files['t2w']
    t2f = request.files['t2f']

    if t1n.filename == '' or t1c.filename == '' or t2w.filename == '' or t2f.filename == '':
        return "Empty filename", 400
    
    if not (allowed_file(t1n.filename) and allowed_file(t1c.filename) and allowed_file(t2w.filename) and allowed_file(t2f.filename)):
        return "Invalid file type", 400

    t1n_filename = secure_filename(t1n.filename)
    t1c_filename = secure_filename(t1c.filename)
    t2w_filename = secure_filename(t2w.filename)
    t2f_filename = secure_filename(t2f.filename)

    t1n_path = os.path.join(app.config['UPLOAD_FOLDER'], t1n_filename)
    t1c_path = os.path.join(app.config['UPLOAD_FOLDER'], t1c_filename)
    t2w_path = os.path.join(app.config['UPLOAD_FOLDER'], t2w_filename)
    t2f_path = os.path.join(app.config['UPLOAD_FOLDER'], t2f_filename)

    t1n.save(t1n_path)
    t1c.save(t1c_path)
    t2w.save(t2w_path)
    t2f.save(t2f_path)

    print("API accessed here 1")

    try:
        t1n_image = sitk.ReadImage(t1n_path, sitk.sitkFloat32)
        t1c_image = sitk.ReadImage(t1c_path, sitk.sitkFloat32)
        t2w_image = sitk.ReadImage(t2w_path, sitk.sitkFloat32)
        t2f_image = sitk.ReadImage(t2f_path, sitk.sitkFloat32)
    except Exception as e:
        return jsonify({"error": f"Error reading images: {e}"}), 500
    
    print("API accessed here 2")

    images = [t1n_image, t1c_image, t2w_image, t2f_image]
    inputs_minmax = torch.tensor(np.array([sitk.GetArrayFromImage(img) for img in images])).unsqueeze(0).float()
    inputs_zscore = inputs_minmax.clone()  # Modify this based on actual preprocessing steps
    inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
    inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
    
    crops_idx_minmax = [(0, inputs_minmax.size(2)), (0, inputs_minmax.size(3)), (0, inputs_minmax.size(4))]
    crops_idx_zscore = [(0, inputs_zscore.size(2)), (0, inputs_zscore.size(3)), (0, inputs_zscore.size(4))]
    print("API accessed here 3")
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
        else:
            return jsonify({"error": f"Unknown normalization method: {normalisation}"}), 500

        print("API accessed here 4")
        
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
    print("API accessed here 44")
    if not model_preds:
        return jsonify({"error": "No predictions"}), 500

    pre_segs = torch.stack(model_preds).mean(dim=0)
    segs = pre_segs[0].numpy() > 0.5
    et = segs[0]
    net = np.logical_and(segs[1], np.logical_not(et))
    ed = np.logical_and(segs[2], np.logical_not(segs[1]))
    labelmap = np.zeros(segs[0].shape)
    labelmap[et] = 3
    labelmap[net] = 1
    labelmap[ed] = 2

    print("API accessed here 5")

    for label in range(1, 4):
        mask = labelmap == label
        processed_mask = rm_dust_fh(mask, thresh=50)
        labelmap[mask] = 0
        labelmap[processed_mask] = label

    print("API accessed here 6")

    output_image = sitk.GetImageFromArray(labelmap.astype(np.uint8))
    output_image.CopyInformation(t1n_image)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_output.nii')
    sitk.WriteImage(output_image, output_path)

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Flask app")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the app on")
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)

