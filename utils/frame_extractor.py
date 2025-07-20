import cv2
import os

def extract_frames(video_path: str, output_folder: str, interval_seconds: int = 1):
    os.makedirs(output_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    saved_count = 0

    success, frame = video.read()

    while success:
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
        success, frame = video.read()
        frame_count += 1

    video.release()
    return saved_count
