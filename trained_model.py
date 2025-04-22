import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.fcl = nn.Linear(24 * 29 * 29, 120)
        self.fcl2 = nn.Linear(120, 84)
        self.fcl3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fcl(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x


def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    image = image.unsqueeze(0)
    return image


def predict(model, image_path):
    class_names = ['car', 'truck']

    image = prepare_image(image_path)

    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

    return predicted_class, confidence


if __name__ == "__main__":
    model = NeuralNet()
    model.load_state_dict(torch.load("trained_model_20_epochs.pth"))
    #uncomment line to use 30 epochs version and comment previous line
    #model.load_state_dict(torch.load("trained_model_30_epochs.pth"))


    image_path = "example/06997.jpeg"
    predicted_class, confidence = predict(model, image_path)
    print(f"The image is predicted to be a {predicted_class} with {confidence:.2f}% confidence")

    test_images = ['example/06997.jpeg',
              'example/06999.jpeg',
              'example/07001.jpeg',
              'example/07007.jpeg',
              'example/07008.jpeg',
              'example/07009.jpeg',
              'example/07010.jpeg',
              'example/07013.jpeg',
              'example/07014.jpeg',
              'example/07015.jpeg',
              'example/07021.jpeg',
              'example/07026.jpeg',
              'example/07031.jpeg',
              'example/07032.jpeg']

    for img_path in test_images:
        predicted_class, confidence = predict(model, img_path)
        print(f"Image {img_path}: {predicted_class} ({confidence:.2f}%)")

