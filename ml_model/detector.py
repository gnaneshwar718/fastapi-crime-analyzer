import os
from crime_model.predict import predict_single_frame, load_model_and_transform

# Load model and transform once
model, transform = load_model_and_transform("slowfast_crime.pth")

def detect_crime_in_frames(frame_folder: str):
    frame_results = []

    for filename in sorted(os.listdir(frame_folder)):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(frame_folder, filename)
            try:
                label = predict_single_frame(model, transform, frame_path)
                frame_results.append({"frame": filename, "label": label})
            except Exception as e:
                frame_results.append({"frame": filename, "error": str(e)})

    return frame_results
