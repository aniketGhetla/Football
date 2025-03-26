import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn

# Heatmap generation 
def generate_team_heatmap(player_positions, field_size=(100, 100)):
    heatmap = np.zeros(field_size, dtype=np.float32)
    for x, y in player_positions:
        if 0 <= x < field_size[1] and 0 <= y < field_size[0]:
            heatmap[int(y), int(x)] += 1
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap = (heatmap / heatmap.max()) * 255
    return heatmap.astype(np.uint8)

#  CNN Model Loader
def load_formation_model(model_path='models/best_formation_model.pth', num_classes=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Predict Formation 
def predict_formation_from_heatmap(model, heatmap):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406],   
                             [0.229, 0.224, 0.225])
    ])

    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    input_tensor = transform(heatmap_pil).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    return pred.item()
