import torch
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import resize
import sys
import os
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50
VIDEO_CLASSES = ['Assault', 'Robbery', 'Vandalism', 'Fighting', 'Normal']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VIDEO_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
IDX_TO_CLASS = {i: c for i, c in enumerate(VIDEO_CLASSES)}
def load_model_and_transform(model_path):
    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, len(VIDEO_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return model, transform

def predict_single_frame(model, transform, image_path):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Repeat frame for fast pathway (32 frames)
    fast = tensor.unsqueeze(2).repeat(1, 1, 32, 1, 1)  # [1, 3, 32, 224, 224]

    # Subsample for slow pathway (every 4th frame → 8 frames)
    index = torch.linspace(0, 31, 8).long()
    slow = torch.index_select(fast, 2, index)

    with torch.no_grad():
        output = model([slow, fast])
        pred = torch.argmax(output, dim=1).item()
        return IDX_TO_CLASS[pred]


def load_model(weights_path):
    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, len(VIDEO_CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_video(video_path, clip_duration=2.0):
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=0, end_sec=clip_duration)["video"]

    if clip.ndim == 5:
        clip = clip.permute(0, 4, 1, 2, 3).squeeze(0)
    elif clip.ndim != 4:
        raise ValueError(f"Unexpected shape: {clip.shape}")

    clip = clip.float() / 255.0
    c, t, h, w = clip.shape

    target_frames = 32
    if t < target_frames:
        pad = target_frames - t
        pad_tensor = torch.zeros((c, pad, h, w))
        clip = torch.cat([clip, pad_tensor], dim=1)
    elif t > target_frames:
        clip = clip[:, :target_frames, :, :]

    clip = resize(clip, [224, 224])

    alpha = 4
    fast_pathway = clip
    index = torch.linspace(0, target_frames - 1, target_frames // alpha).long()
    slow_pathway = torch.index_select(clip, 1, index)

    return [slow_pathway.unsqueeze(0), fast_pathway.unsqueeze(0)]  # [1, C, T, H, W]

def predict(video_path, model_path="slowfast_crime.pth"):
    model = load_model(model_path)
    slow, fast = preprocess_video(video_path)
    with torch.no_grad():
        outputs = model([slow, fast])
        pred_class = outputs.argmax().item()
        print(f"🎯 Predicted Class: {IDX_TO_CLASS[pred_class]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <video_path>")
    else:
        predict(sys.argv[1])
