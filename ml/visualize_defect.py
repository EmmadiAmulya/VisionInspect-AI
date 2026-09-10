import os
import cv2
import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, 3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64, 3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3, 3, stride=2,
                padding=1, output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "bottle_autoencoder_v2.pth"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test",
    "broken_small",
    "000.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")

model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("V2 model loaded.")


image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found:")
    print(IMAGE_PATH)
    raise SystemExit


image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = cv2.resize(
    image,
    (224, 224)
)

original = image.copy()

image_tensor = (
    torch.tensor(
        image,
        dtype=torch.float32
    )
    / 255.0
)

image_tensor = image_tensor.permute(
    2, 0, 1
).unsqueeze(0)


with torch.no_grad():

    reconstructed = model(
        image_tensor
    )


reconstructed = (
    reconstructed.squeeze(0)
    .permute(1, 2, 0)
    .numpy()
)

reconstructed = (
    reconstructed * 255
).clip(0, 255).astype(np.uint8)


difference = cv2.absdiff(
    original,
    reconstructed
)

difference_gray = cv2.cvtColor(
    difference,
    cv2.COLOR_RGB2GRAY
)

heatmap = cv2.applyColorMap(
    difference_gray,
    cv2.COLORMAP_JET
)


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "original.png"
    ),
    cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "reconstructed.png"
    ),
    cv2.cvtColor(
        reconstructed,
        cv2.COLOR_RGB2BGR
    )
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "difference.png"
    ),
    cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2BGR
    )
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "heatmap.png"
    ),
    heatmap
)


print()
print("Visualization completed!")
print()
print("Files saved in:")
print(OUTPUT_DIR)
print()
print("Created:")
print("original.png")
print("reconstructed.png")
print("difference.png")
print("heatmap.png")