from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request
import insightface
from insightface.app import FaceAnalysis
import onnxruntime  # Ensure this package is installed

app = FastAPI()

# Initialize the face analysis model
app_model = FaceAnalysis(name='buffalo_l')
app_model.prepare(ctx_id=-1, det_size=(640, 640), providers=['AzureExecutionProvider'])  # Use AzureExecutionProvider

# Function to download the model if not available
def download_model_if_needed(model_path):
    if not os.path.exists(model_path):
        model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        print(f"Downloading model from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully.")

# Initialize the face swapper model
model_path = "inswapper_128.onnx"
download_model_if_needed(model_path)
swapper = insightface.app.FaceSwapper(app_model, name='inswapper_128', model_path=model_path, providers=['AzureExecutionProvider'])

@app.post("/swap")
async def swap_faces(source_file: UploadFile = File(...), target_file: UploadFile = File(...)):
    # Read the source and target images
    source_img = np.array(Image.open(BytesIO(await source_file.read())))
    target_img = np.array(Image.open(BytesIO(await target_file.read())))

    # Convert images to BGR format for OpenCV compatibility
    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    # Detect faces in the source image
    source_faces = app_model.get(source_img)
    if not source_faces:
        raise HTTPException(status_code=400, detail="No faces detected in the source image.")

    # Detect faces in the target image
    target_faces = app_model.get(target_img)
    if not target_faces:
        raise HTTPException(status_code=400, detail="No faces detected in the target image.")

    # Get the first detected face from the source image
    source_face = source_faces[0]

    # Perform the face swap
    res = target_img.copy()
    for face in target_faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    # Convert the result back to RGB for proper image display
    res_image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res_image_pil = Image.fromarray(res_image)

    # Create an in-memory buffer
    buffer = BytesIO()
    res_image_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")
