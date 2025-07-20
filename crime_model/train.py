import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms.functional import resize
from tqdm import tqdm
import glob

VIDEO_CLASSES = ['Assault', 'Robbery', 'Vandalism', 'Fighting', 'Normal']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VIDEO_CLASSES)}

# ---------------------
# Dataset
# ---------------------
class CrimeVideoDataset(Dataset):
    def __init__(self, root_dir, clip_duration=2.0):
        self.video_paths = []
        self.labels = []
        self.clip_duration = clip_duration

        for label in VIDEO_CLASSES:
            class_folder = os.path.join(root_dir, label)
            for path in glob.glob(os.path.join(class_folder, "*.mp4")):
                self.video_paths.append(path)
                self.labels.append(CLASS_TO_IDX[label])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            video = EncodedVideo.from_path(path)
            clip = video.get_clip(start_sec=0, end_sec=self.clip_duration)["video"]

            # Convert shape to [C, T, H, W]
            if clip.ndim == 5:
                clip = clip.permute(0, 4, 1, 2, 3).squeeze(0)
            elif clip.ndim != 4:
                raise ValueError(f"Unexpected shape: {clip.shape}")

            clip = clip.float() / 255.0  # normalize

            # Ensure clip has 32 frames (Fast pathway)
            target_frames = 32
            c, t, h, w = clip.shape

            if t < target_frames:
                # pad frames at the end
                pad = target_frames - t
                pad_tensor = torch.zeros((c, pad, h, w))
                clip = torch.cat([clip, pad_tensor], dim=1)
            elif t > target_frames:
                clip = clip[:, :target_frames, :, :]

            # Resize spatial dimensions
            clip = resize(clip, [224, 224])  # Resize HxW

            # Fast and Slow pathways
            alpha = 4
            fast_pathway = clip  # [3, 32, 224, 224]
            index = torch.linspace(0, target_frames - 1, target_frames // alpha).long()
            slow_pathway = torch.index_select(clip, 1, index)  # [3, 8, 224, 224]

            return [slow_pathway, fast_pathway], label

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.video_paths))

# ---------------------
# Model
# ---------------------
def get_model(num_classes):
    model = slowfast_r50(pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    return model

# ---------------------
# Training Loop
# ---------------------
def train():
    dataset = CrimeVideoDataset("datasets/UCF-Crime-Subset", clip_duration=2.0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(VIDEO_CLASSES)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        total_loss = 0
        model.train()
        for clips, labels in tqdm(dataloader):
            slow_path, fast_path = clips
            slow_path = slow_path.to(device)
            fast_path = fast_path.to(device)
            labels = labels.to(device)

            outputs = model([slow_path, fast_path])
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "slowfast_crime.pth")
    print("✅ Model saved as 'slowfast_crime.pth'")


if __name__ == "__main__":
    train()
