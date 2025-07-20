from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from utils.frame_extractor import extract_frames

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_videos"
FRAMES_DIR = "extracted_frames"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
from ml_model.detector import detect_crime_in_frames

@app.get("/analyze_frames/{video_name}")
def analyze_frames(video_name: str):
    frame_folder = os.path.join(FRAMES_DIR, video_name)

    if not os.path.exists(frame_folder):
        return {"error": f"No frames found for video '{video_name}'"}

    results = detect_crime_in_frames(frame_folder)
    return {"video": video_name, "analysis": results}

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract frames
    video_name = os.path.splitext(file.filename)[0]
    output_folder = os.path.join(FRAMES_DIR, video_name)
    num_frames = extract_frames(file_path, output_folder)

    return {
        "message": f"Video '{file.filename}' uploaded successfully.",
        "frames_extracted": num_frames,
        "frames_path": output_folder,
    }
from fastapi.staticfiles import StaticFiles

# Mount the static folder for serving extracted frame images
app.mount("/static/frames", StaticFiles(directory="extracted_frames"), name="static_frames")
