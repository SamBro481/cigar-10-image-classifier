import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    outputs = model(img)
    probabilities = torch.softmax(outputs, dim=1)[0]
    confidences = {classes[i]: float(probabilities[i]) for i in range(10)}
    return confidences

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload an image or try one of the examples below.",
    examples=[
        ["dog.jpeg"],
        ["ship.jpeg"]
    ]
)

iface.launch()

