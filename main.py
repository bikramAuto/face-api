from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.post("/encode")
async def encode_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"error": "No face detected"}

    return {"encoding": encodings[0].tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9010, reload=True)
