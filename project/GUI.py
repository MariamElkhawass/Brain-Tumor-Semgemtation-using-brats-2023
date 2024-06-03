import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
from tkinter import filedialog
from tkinter.font import Font
import torch
import torchvision.transforms as transforms
from src.models.unet import Unet  # Make sure to import your model class

class BrainTumorSegmentationModel:
    def __init__(self, model_path):
        # Assuming `model_path` is the path to the saved model
        self.model_path = model_path
        # Provide the required arguments for initializing the Unet model
        inplanes = 3  # Assuming input has 3 channels (e.g., RGB image)
        num_classes = 2  # Assuming binary classification for tumor segmentation
        width = 64  # You need to specify the width parameter according to your requirements
        self.model = Unet(inplanes, num_classes, width)

        # Load the checkpoint
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        # Extract the state_dict
        state_dict = checkpoint['state_dict']
        # Load the state_dict into the model with strict=False
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()  # Set model to evaluation mode

        # Define the image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)
            output = torch.sigmoid(output).cpu().numpy()
            output = output > 0.5
            output = output.squeeze()

        return output

model_path = "D:/project/project/runs/20240530_191833__fold0_Unet_48_batch1_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupFalse_fp16_warm0__normgroup_dropout0.0_warm_restartFalse/model_best.pth.tar"
segmentation_model = BrainTumorSegmentationModel(model_path)

class Program:
    def __init__(self, app):
        self.text_ = 'Welcome to our Brain Tumor Segmentation Tool. We are dedicated to transforming brain tumor ' \
                     'diagnosis. Our cutting-edge technology and deep learning algorithms provide accurate and rapid ' \
                     'segmentation of brain tumors from MRI scans. Join us in the quest for early detection and ' \
                     'effective treatment. SCAN YOUR MRI REPORTS NOW.'

        self.filepath = None
        self.last_captured_frame = np.array([])

        self.app = app
        self.app.geometry("800x500")
        self.app.title("BRAIN TUMOR SEGMENTATION")
        self.app.config(bg="#FFFFFF")
        self.app.resizable(False, False)

        self.custom_font = Font(family="Helvetica", size=34, weight="bold")
        self.custom_font_2 = Font(family="Helvetica", size=22, weight="bold")
        self.custom_font3 = Font(family="Helvetica", size=8, weight="bold")

        # FRAMES -------------------------------------------------------
        self.window = Frame(self.app)
        self.window.grid(row=0, column=0)

        self.window_upload = Frame(self.app)

        self.window_scan = Frame(self.app)

        # PAGE 1
        self.label = Label(self.window, bg='#141414', width=115, height=15)
        self.label.grid(row=0, column=0, columnspan=2)
        self.label_title = Label(self.label, text='Brain Tumor Segmentation Tool', fg='white', width=100,
                                 height=5, bg='#141414', anchor='w', font=self.custom_font, justify='left')
        self.label_title.place(x=10, y=10)

        self.content_label = Label(self.window, bg='#FFFFFF', width=115, height=18, anchor='center', justify='center')
        self.content_label_1 = Label(self.content_label, text='BRAIN TUMOR', font=self.custom_font_2, fg='#141414',
                                     bg='#FFFFFF')
        self.content_label.grid(row=1, column=0)
        self.content_label_1.place(x=280, y=0)

        self.about_label = Label(self.content_label, text=self.text_, wraplength=600, width=70, font=('Helvetica', 12),
                                 height=5, justify="left", anchor="center", bg='#FFFFFF')
        self.about_label.place(x=85, y=40)

        self.buttons_label = Label(self.content_label, width=75, height=3, bg='#FFFFFF')
        self.buttons_label.place(x=160, y=155)

        self.UPLOAD_BUTTON = Button(self.buttons_label, text="UPLOAD MRI/ X-RAYS", width=30, height=2, fg='white',
                                    bg='#4ca3d9', borderwidth=0,
                                    highlightbackground=self.buttons_label.cget('background'), font=self.custom_font3,
                                    command=self.show_upload_page)
        self.SCAN_BUTTON = Button(self.buttons_label, text="SCAN MRI/ X-RAYS", width=30, height=2, fg='white',
                                  bg='#4ca3d9', borderwidth=0,
                                  highlightbackground=self.buttons_label.cget('background'), font=self.custom_font3,
                                  command=self.show_scan_page)

        self.UPLOAD_BUTTON.grid(row=0, column=0, padx=(0, 10))
        self.SCAN_BUTTON.grid(row=0, column=1, padx=(10, 0))

        # PAGE upload
        self.up_txt = "Please Upload Your MRI/CT scan (x-ray image) copy. The copy will be passed to the ML model for"\
                      " prediction. Early Treatment and Diagnosis is necessary for Tumors"

        self.go_back_upload = Label(self.window_upload, width=115, height=2, bg='#141414')
        self.go_Back_upload_button = Button(self.go_back_upload, text='Go back', fg='white', bg='#141414',
                                            borderwidth=0, highlightbackground=self.buttons_label.cget('background'),
                                            justify='left', width=10, height=2, font=("Helvetica", 8),
                                            command=self.go_back_upload_)
        self.upload_label = Label(self.window_upload, width=115, height=31)
        self.upload_title = Label(self.upload_label, text='UPLOAD YOUR MRI SCAN/ X-RAY IMAGE',
                                  font=Font(family="Helvetica", size=22, weight="bold"))
        self.upload_msg = Label(self.upload_label, text=self.up_txt, wraplength=500, width=120, font=('Helvetica', 8),
                                height=5, justify="left", anchor="center")
        self.upload_select = Button(self.upload_label, text='Upload', fg='white', bg='#4ca3d9', borderwidth=0,
                                    highlightbackground=self.buttons_label.cget('background'), justify='left', width=15,
                                    height=1, font=("Helvetica", 8), command=self.upload_image)
        self.upload_canvas = Canvas(self.upload_label, bg='#141414', width=640, height=480)

        self.upload_label.grid(row=0, column=0)
        self.go_back_upload.grid(row=0, column=0)
        self.go_Back_upload_button.grid(row=0, column=0)
        self.upload_title.place(x=160, y=0)
        self.upload_msg.place(x=160, y=50)
        self.upload_select.place(x=330, y=140)
        self.upload_canvas.place(x=160, y=170)

        # PAGE scan
        self.scan_txt = "Please use your camera to take pictures of your MRI/CT scan (x-ray image). The copy will be " \
                        "passed to the ML model for prediction. Early Treatment and Diagnosis is necessary for Tumors"

        self.go_back_scan = Label(self.window_scan, width=115, height=2, bg='#141414')
        self.go_Back_scan_button = Button(self.go_back_scan, text='Go back', fg='white', bg='#141414',
                                          borderwidth=0, highlightbackground=self.buttons_label.cget('background'),
                                          justify='left', width=10, height=2, font=("Helvetica", 8),
                                          command=self.go_back_scan_)
        self.scan_label = Label(self.window_scan, width=115, height=31)
        self.scan_title = Label(self.scan_label, text='SCAN YOUR MRI SCAN/ X-RAY IMAGE',
                                font=Font(family="Helvetica", size=22, weight="bold"))
        self.scan_msg = Label(self.scan_label, text=self.scan_txt, wraplength=500, width=120, font=('Helvetica', 8),
                              height=5, justify="left", anchor="center")
        self.scan_select = Button(self.scan_label, text='Capture', fg='white', bg='#4ca3d9', borderwidth=0,
                                  highlightbackground=self.buttons_label.cget('background'), justify='left', width=15,
                                  height=1, font=("Helvetica", 8), command=self.scan_image)
        self.scan_canvas = Canvas(self.scan_label, bg='#141414', width=640, height=480)

        self.scan_label.grid(row=0, column=0)
        self.go_back_scan.grid(row=0, column=0)
        self.go_Back_scan_button.grid(row=0, column=0)
        self.scan_title.place(x=160, y=0)
        self.scan_msg.place(x=160, y=50)
        self.scan_select.place(x=330, y=140)
        self.scan_canvas.place(x=160, y=170)

        # Display the first page
        self.window.grid(row=0, column=0)

    def show_upload_page(self):
        self.window.grid_forget()
        self.window_scan.grid_forget()
        self.window_upload.grid(row=0, column=0)

    def show_scan_page(self):
        self.window.grid_forget()
        self.window_upload.grid_forget()
        self.window_scan.grid(row=0, column=0)

    def go_back_upload_(self):
        self.window_upload.grid_forget()
        self.window.grid(row=0, column=0)

    def go_back_scan_(self):
        self.window_scan.grid_forget()
        self.window.grid(row=0, column=0)

    def upload_image(self):
        # Allow user to select an image file
        self.filepath = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Image",
            filetypes=(("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"))
        )
        if self.filepath:
            image = Image.open(self.filepath)
            image = image.resize((640, 480), Image.ANTIALIAS)
            self.display_image(image)
            # Perform prediction
            self.perform_prediction(image)

    def display_image(self, image):
        photo = ImageTk.PhotoImage(image)
        self.upload_canvas.create_image(0, 0, image=photo, anchor=NW)
        self.upload_canvas.image = photo

    def scan_image(self):
        # Use OpenCV to capture image from camera
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Capture Image", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    # Capture the frame and break the loop
                    self.last_captured_frame = frame
                    break
        cap.release()
        cv2.destroyAllWindows()
        if self.last_captured_frame.size != 0:
            image = cv2.cvtColor(self.last_captured_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((640, 480), Image.ANTIALIAS)
            self.display_scan_image(image)
            # Perform prediction
            self.perform_prediction(image)

    def display_scan_image(self, image):
        photo = ImageTk.PhotoImage(image)
        self.scan_canvas.create_image(0, 0, image=photo, anchor=NW)
        self.scan_canvas.image = photo

    def perform_prediction(self, image):
        image = image.convert("RGB")  # Ensure image is in RGB format
        prediction = segmentation_model.predict(image)

        # Display the prediction result on the same canvas
        result_image = Image.fromarray((prediction * 255).astype(np.uint8))
        result_image = result_image.resize((640, 480), Image.ANTIALIAS)
        self.display_image(result_image)

app = tk.Tk()
Program(app)
app.mainloop()
