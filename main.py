from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image, ImageOps
import face_recognition
import numpy as np
from auth import verify_jwt
import pickle
import base64
import io
import os

app = FastAPI()

FACES_FILE = "faces.pkl"

if os.path.exists(FACES_FILE) and os.path.getsize(FACES_FILE) > 0:
    with open(FACES_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

class RegisterFace(BaseModel):
    id: str
    image_base64: str

def save_faces():
    with open(FACES_FILE, "wb") as f:
        pickle.dump(known_faces, f)

@app.post("/register")
def register_face(data: RegisterFace, user=Depends(verify_jwt)):
    try:
        print("Raw image_base64 length:", len(data.image_base64))

        base64_str = data.image_base64.split(",")[-1]
        image_data = base64.b64decode(base64_str)

        # Load and standardise image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print("Loaded format:", image.format)
        print("Original mode:", image.mode)

        image = ImageOps.exif_transpose(image)
        image.thumbnail((640, 640))
        image = image.copy()

        # Convert to NumPy array
        image_np = np.array(image)
        print("Image np dtype before:", image_np.dtype, "Shape:", image_np.shape)

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)

        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Image must be 3-channel RGB")

        image_np = np.ascontiguousarray(image_np)
        image_np.setflags(write=True)
        print("Final image dtype:", image_np.dtype)
        print("Is contiguous:", image_np.flags['C_CONTIGUOUS'])

        # Face encoding
        face_locations = face_recognition.face_locations(image_np)

        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Then extract encoding
        encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)

        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        encoding = encodings[0]

        # Cap at 20 encodings per ID
        known_faces.setdefault(data.id, [])
        if len(known_faces[data.id]) >= 20:
            known_faces[data.id].pop(0)
        known_faces[data.id].append(encoding)

        save_faces()
        return {"message": f"Face added for ID {data.id}"}

    except Exception as e:
        print(f"Error in /register: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RecognitionRequest(BaseModel):
    image_base64: str

@app.post("/recognize")
def recognize_face(data: RecognitionRequest, user=Depends(verify_jwt)):
    try:
        if not known_faces:
            raise HTTPException(status_code=404, detail="No registered faces found")

        image_data = base64.b64decode(data.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        encodings = face_recognition.face_encodings(image_np)
        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        unknown_encoding = encodings[0]

        for person_id, enc_list in known_faces.items():
            results = face_recognition.compare_faces(enc_list, unknown_encoding, tolerance=0.5)
            if any(results):
                return {"matched_id": person_id}

        return {"matched_id": None, "message": "No match found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/faces")
def delete_all_faces(user=Depends(verify_jwt)):
    global known_faces
    known_faces = {}
    if os.path.exists(FACES_FILE):
        os.remove(FACES_FILE)
    return {"message": "All face encodings deleted successfully."}

@app.delete("/faces/{id}")
def delete_face_by_id(id: str, user=Depends(verify_jwt)):
    if id not in known_faces:
        raise HTTPException(status_code=404, detail=f"No face data found for ID: {id}")
    del known_faces[id]
    save_faces()
    return {"message": f"Face encodings for ID '{id}' deleted successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9010, reload=True)
