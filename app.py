import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import tempfile

# ==========================================
# 1. MODEL ARCHITECTURES (Copied from your Notebooks)
# ==========================================

# --- A. UNET (Area Detection) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f*2, f))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            if x.shape != skip.shape:
                x = TF.resize(x, skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

# --- B. REGRESSOR (Line Detection) ---
class RunwayRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Use standard ResNet18
        self.backbone = models.resnet18(pretrained=False) 
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 2. SETUP & INFERENCE
# ==========================================

DEVICE = "cpu" # Use "cuda" if you have GPU hardware on HF

# Initialize Models
unet_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
reg_model = RunwayRegressor().to(DEVICE)

# Load Weights (Ensure these files exist in your HF Space)
try:
    unet_model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
    print("UNet weights loaded.")
except Exception as e:
    print(f"Error loading UNet: {e}")

try:
    reg_model.load_state_dict(torch.load("best_regressor.pth", map_location=DEVICE))
    print("Regressor weights loaded.")
except Exception as e:
    print(f"Error loading Regressor: {e}")

unet_model.eval()
reg_model.eval()

def process_frame(frame):
    """
    Takes a raw RGB frame, runs both models, and overlays results.
    """
    # 1. Resize to Model Input Size (640x360 matches your Regressor training)
    # UNet handles dynamic sizes, so 640x360 works for both.
    MODEL_W, MODEL_H = 640, 360
    h_orig, w_orig = frame.shape[:2]
    
    img_resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    
    # 2. Preprocess (Normalize 0-1, CHW)
    # Note: Your notebook used image / 255.0 manually
    img_tensor = torch.tensor(img_resized).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # --- Run UNet (Area) ---
        mask_logits = unet_model(img_tensor)
        # Sigmoid because it was trained with BCEWithLogits
        mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
        mask_binary = (mask_prob > 0.5).astype(np.uint8)
        
        # --- Run Regressor (Lines) ---
        coords = reg_model(img_tensor).cpu().numpy()[0]

    # 3. Visualization
    
    # A. Draw Area (Green Mask)
    # Create green overlay where mask is 1
    green_mask = np.zeros_like(img_resized)
    green_mask[mask_binary == 1] = [0, 255, 0] # RGB Green
    
    # Blend: 70% Original, 30% Green
    overlay = cv2.addWeighted(img_resized, 1.0, green_mask, 0.3, 0)
    
    # B. Draw Lines (Red, Blue, Green)
    # Denormalize coordinates
    c = coords.copy()
    c[0::2] *= MODEL_W # X
    c[1::2] *= MODEL_H # Y
    c = c.astype(int)
    
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)] # Left=Red, Right=Blue, Center=Green
    
    for i in range(3):
        start_idx = i * 4
        # Draw thick lines
        cv2.line(overlay, 
                 (c[start_idx], c[start_idx+1]), 
                 (c[start_idx+2], c[start_idx+3]), 
                 colors[i], 3)

    # 4. Resize back to original video size (Optional, keeps quality)
    final_output = cv2.resize(overlay, (w_orig, h_orig))
    
    return final_output

def analyze_video(video_path):
    if video_path is None:
        return None
        
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Gradio requires a file path for output
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV is BGR -> Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        processed_frame = process_frame(frame)
        
        # Convert RGB -> BGR for VideoWriter
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        out.write(processed_frame)
        
    cap.release()
    out.release()
    return output_path

# ==========================================
# 3. GRADIO UI
# ==========================================
iface = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload Cockpit View"),
    outputs=gr.Video(label="AI Analysis"),
    title="✈️ Runway AI Assistant",
    description="Detects Runway Area (Green) and Center/Edge Lines (Red/Blue/Green) using ResNet18 + U-Net.",
    examples=[] 
)

if __name__ == "__main__":
    iface.launch()