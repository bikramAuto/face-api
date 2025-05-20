import dlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps
import face_recognition
import numpy as np
import pickle
import base64
import io
import os
# from typing import Optional

app = FastAPI()

FACES_FILE = "faces.pkl"

# Load existing encodings
if os.path.exists(FACES_FILE) and os.path.getsize(FACES_FILE) > 0:
    with open(FACES_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}  # {id: [encoding1, encoding2, ...]}

class RegisterFace(BaseModel):
    id: str
    image_base64: str

@app.post("/register")
def register_face(data: RegisterFace):
    try:
        # Strip prefix if present
        base64_str = data.image_base64.split(",")[-1]

        # Decode base64
        image_data = base64.b64decode(base64_str)

        # Load image with PIL and ensure it's RGB
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = image.copy()

        # Convert to NumPy array with strict dtype
        image_np = np.array(image, dtype=np.uint8).copy()
        image_np = np.ascontiguousarray(image_np)
        image_np.setflags(write=True)

        # Sanity checks
        if image_np.ndim != 3 or image_np.shape[2] != 3 or image_np.dtype != np.uint8:
            raise HTTPException(status_code=400, detail="Image must be RGB and 8-bit")

        print("Face recognition version:", face_recognition.__version__)
        print("dlib version:", dlib.__version__)
        print("Image mode:", image.mode)
        print("Image shape:", image_np.shape)
        print("Image dtype:", image_np.dtype)
        print("Flags:", image_np.flags)
        print("numpy: ", np.__version__)

        # Attempt encoding
        encodings = face_recognition.face_encodings(image_np)
        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        encoding = encodings[0]

        # Load or initialize known_faces
        if os.path.exists(FACES_FILE) and os.path.getsize(FACES_FILE) > 0:
            with open(FACES_FILE, "rb") as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}

        known_faces.setdefault(data.id, []).append(encoding)

        # Save back to faces.pkl
        with open(FACES_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        return {"message": f"Face added for ID {data.id}"}

    except Exception as e:
        print(f"Error in /register: {e}")
        raise HTTPException(status_code=500, detail=str(e))




class RecognitionRequest(BaseModel):
    image_base64: str

@app.post("/recognize")
def recognize_face(data: RecognitionRequest):
    try:
        if not os.path.exists(FACES_FILE) or os.path.getsize(FACES_FILE) == 0:
            raise HTTPException(status_code=404, detail="No registered faces found")

        # Decode incoming image
        image_data = base64.b64decode(data.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        # Extract encoding
        encodings = face_recognition.face_encodings(image_np)
        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        unknown_encoding = encodings[0]

        # Load known faces
        with open(FACES_FILE, "rb") as f:
            known_faces = pickle.load(f)

        # Compare with each stored encoding
        for person_id, enc_list in known_faces.items():
            results = face_recognition.compare_faces(enc_list, unknown_encoding, tolerance=0.5)
            if any(results):
                return {"matched_id": person_id}

        return {"matched_id": None, "message": "No match found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9010, reload=True)