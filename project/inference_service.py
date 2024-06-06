# import argparse
# import os
# import pathlib
# import random
# from datetime import datetime
# from types import SimpleNamespace
# import io
# import glob
# from flask import render_template
# import yaml
# import SimpleITK as sitk
# import numpy as np
# import torch
# import torch.optim
# import torch.utils.data
# import uvicorn
# from torch.cuda.amp import autocast
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse

# from src import models
# from src.dataset import get_datasets
# from src.dataset.batch_utils import pad_batch1_to_compatible_size
# from src.tta import apply_simple_tta
# from src.models.layers import get_norm_layer
# from src.utils import reload_ckpt_bis
# from src.dataset.postprocessing import rm_dust_fh

# app = FastAPI()

# models_list = []
# normalisations_list = []


# def load_models():
#     print("Loading models...")  # Debugging statement

#     # Define the directory path where the models are stored
#     runs_dir = "D:/project/project/runs"

#     # Find all subdirectories (each represents a model run)
#     subdirs = [subdir for subdir in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, subdir))]
#     print(f"Found subdirectories: {subdirs}")  # Debugging statement

#     for subdir in subdirs:
#         subdir_path = os.path.join(runs_dir, subdir)
#         print(f"Processing subdirectory: {subdir_path}")  # Debugging statement

#         # Find YAML configuration file
#         config_file = glob.glob(os.path.join(subdir_path, "*.yaml"))
#         if not config_file:
#             print(f"No YAML configuration file found in {subdir_path}")
#             continue
#         config_file = config_file[0]

#         # Load YAML configuration
#         with open(config_file, "r") as file:
#             model_args = yaml.safe_load(file)
#             model_args = SimpleNamespace(**model_args, ckpt=os.path.join(subdir_path, "model_best.pth.tar"))

#         # Print the loaded configuration for debugging
#         print(f"Loaded configuration for {subdir_path}: {model_args}")

#         # Load model
#         model_maker = getattr(models, model_args.arch)
#         model = model_maker(4, 3, width=model_args.width, deep_supervision=model_args.deep_sup,
#                             norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        
#         # Load the checkpoint
#         checkpoint = torch.load(model_args.ckpt)
#         state_dict = checkpoint['state_dict']
        
#         # Check if the model is wrapped in DataParallel
#         if any(key.startswith('module.') for key in state_dict.keys()):
#             # Remove 'module.' prefix from the state dict keys
#             new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#         else:
#             new_state_dict = state_dict
        
#         try:
#             model.load_state_dict(new_state_dict)
#             print(f"Model loaded successfully from {subdir_path}")
#         except Exception as e:
#             print(f"Error loading model from {subdir_path}: {e}")
#             continue

#         # Wrap the model with DataParallel
#         model = torch.nn.DataParallel(model)
        
#         # Append the model to the global models_list
#         models_list.append(model)
#         normalisations_list.append(model_args.norm_layer)
    
#     print("Models loading completed.")  # Debugging statement

# @app.on_event("startup")
# async def startup_event():
#     load_models()

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')

# @app.get("/test-models")
# def test_models():
#     if models_list:
#         return {"message": f"Models are loaded successfully: {len(models_list)} models loaded."}
#     else:
#         return {"message": "No models are loaded"}
    
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     files = request.files.getlist("file[]")
    
# #     if len(files) != 4:
# #         return render_template('index.html', prediction_text="Please upload exactly 4 files.")

# #     int_features = []
# #     for file in files:
# #         file_path = os.path.join('/tmp', file.filename)  # Temporary path to save the file
# #         file.save(file_path)
# #         nii_data = nib.load(file_path)
# #         data = nii_data.get_fdata()
# #         int_features.append(np.mean(data))  # Example feature extraction
    
# #     final_features = [np.array(int_features)]
# #     prediction = model.predict(final_features)
# #     output = round(prediction[0])
    
# #     return render_template('index.html', prediction_text="Number of Weekly Rides Should be {}".format(output))


# @app.route('/predict', methods=['GET', 'POST'])
# async def predict(t1n: UploadFile = File(...), t1c: UploadFile = File(...), t2w: UploadFile = File(...), t2f: UploadFile = File(...)):
#     #Load the uploaded image files
#     t1n_image = sitk.ReadImage(io.BytesIO(await t1n.read()), sitk.sitkFloat32)
#     t1c_image = sitk.ReadImage(io.BytesIO(await t1c.read()), sitk.sitkFloat32)
#     t2w_image = sitk.ReadImage(io.BytesIO(await t2w.read()), sitk.sitkFloat32)
#     t2f_image = sitk.ReadImage(io.BytesIO(await t2f.read()), sitk.sitkFloat32)

#     # Combine the images into a single input tensor
#     images = [t1n_image, t1c_image, t2w_image, t2f_image]
#     inputs_minmax = torch.tensor([sitk.GetArrayFromImage(img) for img in images]).unsqueeze(0).float()
#     inputs_zscore = inputs_minmax.clone()  # Modify this based on actual preprocessing steps
#     inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
#     inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
#     print(images[0].shape)
#     crops_idx_minmax = [(0, inputs_minmax.size(2)), (0, inputs_minmax.size(3)), (0, inputs_minmax.size(4))]
#     crops_idx_zscore = [(0, inputs_zscore.size(2)), (0, inputs_zscore.size(3)), (0, inputs_zscore.size(4))]

#     model_preds = []
#     for model, normalisation in zip(models_list, normalisations_list):
#         if normalisation == "minmax":
#             inputs = inputs_minmax.cuda()
#             pads = pads_minmax
#             crops_idx = crops_idx_minmax
#         elif normalisation == "zscore":
#             inputs = inputs_zscore.cuda()
#             pads = pads_zscore
#             crops_idx = crops_idx_zscore

#         model.cuda()
#         with autocast():
#             with torch.no_grad():
#                 pre_segs = model(inputs).sigmoid_().cpu()
#                 maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
#                 pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
#                 segs = torch.zeros((1, 3, 155, 240, 240))
#                 segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
#                 model_preds.append(segs)

#         model.cpu()

#     if not model_preds:
#         return JSONResponse(content={"error": "No predictions"}, status_code=500)

#     pre_segs = torch.stack(model_preds).mean(dim=0)
#     segs = pre_segs[0].numpy() > 0.5
#     et = segs[0]
#     net = np.logical_and(segs[1], np.logical_not(et))
#     ed = np.logical_and(segs[2], np.logical_not(segs[1]))
#     labelmap = np.zeros(segs[0].shape)
#     labelmap[et] = 3
#     labelmap[net] = 1
#     labelmap[ed] = 2

#     for label in range(1, 4):
#         mask = labelmap == label
#         processed_mask = rm_dust_fh(mask)
#         labelmap[mask] = 0
#         labelmap[processed_mask] = label

#     labelmap = sitk.GetImageFromArray(labelmap.astype(np.uint8))
#     labelmap.CopyInformation(t1n_image)

#     # Save the labelmap to a temporary file
#     temp_filename = "segmented_result.nii"
#     sitk.WriteImage(labelmap, temp_filename)
#     output= (labelmap)

#     return FileResponse(temp_filename, media_type='application/octet-stream', filename=temp_filename)

# if __name__ == '__main__':
#     app.run(debug=True, port=3000)









import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace
import io
import glob
from flask import Flask, render_template, render_template_string, request, jsonify, send_file, send_from_directory
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

# @app.route('/')
# def home():
#     return "Hello, Flask!"

app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'nii'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

UPLOAD_FOLDER = 'saved_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def load_models():
    print("Loading models...")  # Debugging statement

    # Define the directory path where the models are stored
    runs_dir = "D:/project/project/runs"

    # Find all subdirectories (each represents a model run)
    subdirs = [subdir for subdir in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, subdir))]
    print(f"Found subdirectories: {subdirs}")  # Debugging statement

    for subdir in subdirs:
        subdir_path = os.path.join(runs_dir, subdir)
        print(f"Processing subdirectory: {subdir_path}")  # Debugging statement

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
        
        # Load the checkpoint
        checkpoint = torch.load(model_args.ckpt)
        state_dict = checkpoint['state_dict']
        
        # Check if the model is wrapped in DataParallel
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix from the state dict keys
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
        
        try:
            model.load_state_dict(new_state_dict)
            print(f"Model loaded successfully from {subdir_path}")
        except Exception as e:
            print(f"Error loading model from {subdir_path}: {e}")
            continue

        # Wrap the model with DataParallel
        model = torch.nn.DataParallel(model)
        
        # Append the model to the global models_list
        models_list.append(model)
        normalisations_list.append(model_args.norm_layer)
    
    print("Models loading completed.")  # Debugging statement

# @app.before_first_request
# def startup_event():
#     load_models()

@app.route('/', methods=['GET'])
def index():
    # Main page
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

    # Load the uploaded image files
    t1n_image = sitk.ReadImage(t1n_path, sitk.sitkFloat32)
    t1c_image = sitk.ReadImage(t1c_path, sitk.sitkFloat32)
    t2w_image = sitk.ReadImage(t2w_path, sitk.sitkFloat32)
    t2f_image = sitk.ReadImage(t2f_path, sitk.sitkFloat32)

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

    for label in range(1, 4):
        mask = labelmap == label
        processed_mask = rm_dust_fh(mask)
        labelmap[mask] = 0
        labelmap[processed_mask] = label

    labelmap = labelmap.astype(np.uint8)
    labelmap_sitk = sitk.GetImageFromArray(labelmap)
    labelmap_sitk.CopyInformation(t1n_image)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "predicted_segmentation.nii")
    sitk.WriteImage(labelmap_sitk, output_path)

    return send_file(output_path, as_attachment=True)


@app.route("/")
def main():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Segmentation</title>
    </head>
    <body>
        <h1>Brain Tumor Segmentation</h1>
        <form action="/predict" method="post" id="uploadForm" enctype="multipart/form-data">
            <label for="t1n">T1N:</label>
            <input type="file" id="t1n" name="t1n" accept=".nii"><br><br>
            <label for="t1c">T1C:</label>
            <input type="file" id="t1c" name="t1c" accept=".nii"><br><br>
            <label for="t2w">T2W:</label>
            <input type="file" id="t2w" name="t2w" accept=".nii"><br><br>
            <label for="t2f">T2F:</label>
            <input type="file" id="t2f" name="t2f" accept=".nii"><br><br>
            <button type="submit">Submit</button>
        </form>
        <h2>Uploaded MRI Files</h2>
        <div id="uploadedFiles">
            <p id="fileT1N">T1N: Not uploaded</p>
            <p id="fileT1C">T1C: Not uploaded</p>
            <p id="fileT2W">T2W: Not uploaded</p>
            <p id="fileT2F">T2F: Not uploaded</p>
        </div>
        <h2>Segmented MRI Image</h2>
        <div id="segmentationResult">
            <p id="segmentationMessage">Segmentation result will be displayed here.</p>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const formData = new FormData();
                formData.append('t1n', document.getElementById('t1n').files[0]);
                formData.append('t1c', document.getElementById('t1c').files[0]);
                formData.append('t2w', document.getElementById('t2w').files[0]);
                formData.append('t2f', document.getElementById('t2f').files[0]);

                // Display uploaded file names
                document.getElementById('fileT1N').textContent = `T1N: ${document.getElementById('t1n').files[0].name}`;
                document.getElementById('fileT1C').textContent = `T1C: ${document.getElementById('t1c').files[0].name}`;
                document.getElementById('fileT2W').textContent = `T2W: ${document.getElementById('t2w').files[0].name}`;
                document.getElementById('fileT2F').textContent = `T2F: ${document.getElementById('t2f').files[0].name}`;

                // Send request to backend
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const jsonResponse = await response.json();
                    const segmentationHex = jsonResponse.output;

                    // Assuming the backend returns the segmentation as a hexadecimal string
                    const byteArray = new Uint8Array(segmentationHex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
                    const blob = new Blob([byteArray], { type: 'application/octet-stream' });
                    const url = URL.createObjectURL(blob);

                    document.getElementById('segmentationMessage').textContent = `Segmentation done. Download result: `;
                    const downloadLink = document.createElement('a');
                    downloadLink.href = url;
                    downloadLink.download = 'segmented.nii';
                    downloadLink.textContent = 'Download Segmentation';
                    document.getElementById('segmentationMessage').appendChild(downloadLink);
                } else {
                    document.getElementById('segmentationMessage').textContent = 'Error during segmentation.';
                }
            });
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)
