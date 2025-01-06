import torch
import cv2
from pyzbar import pyzbar
from PIL import Image, ImageEnhance
import numpy as np
import subprocess
import os
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.evaluation.predictor import load_checkpoint, Predictor

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# LaMa Model Setup
def setup_lama():
    lama_config_path = "path/to/config.yaml"  # Replace with actual path to LaMa config
    lama_checkpoint_path = "path/to/checkpoint.pth"  # Replace with actual checkpoint path
    lama_model = load_checkpoint(lama_config_path, lama_checkpoint_path, map_location=device)
    predictor = Predictor(lama_model, device=device)
    return predictor

predictor = setup_lama()

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

# Set webcam resolution
width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform YOLOv5 inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for i, detection in detections.iterrows():
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
        x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]

        # Crop and preprocess barcode image
        cropped_img = frame[y1:y2, x1:x2]

        # Create a mask for inpainting
        mask = cv2.inRange(cropped_img, (0, 0, 0), (30, 30, 30))  # Example mask for dark artifacts

        # Prepare inputs for LaMa
        inpaint_input = {
            "image": cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
            "mask": mask
        }
        inpaint_input = {k: move_to_device(v, device) for k, v in inpaint_input.items()}

        # Perform inpainting with LaMa
        with torch.no_grad():
            inpainted_result = predictor(inpaint_input)

        inpainted_img = inpainted_result["predicted_image"][0].cpu().numpy()
        inpainted_img = (inpainted_img * 255).astype(np.uint8)
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)

        # Combine all transformations into a single display
        combined = np.hstack((cropped_img, inpainted_img))
        combined_resized = cv2.resize(combined, (1280, 360))
        cv2.imshow('Processing Stages (Original | Inpainted)', combined_resized)

        # Decode barcodes using Pyzbar
        barcodes = pyzbar.decode(inpainted_img)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            print(f"Detected barcode data: {data}")

    # Display the frame
    cv2.imshow('Result', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Resources released. Program terminated.")
