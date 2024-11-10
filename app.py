from flask import Flask, render_template, request, flash
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from io import BytesIO

app = Flask(__name__)

# Define the DeeperCNN model class
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Load two models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = DeeperCNN().to(device)
model1.load_state_dict(torch.load('model1.pth', map_location=device, weights_only=True))
model1.eval()





class DeeperCNN2(nn.Module):
    def __init__(self, num_classes=2):
        super(DeeperCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model2 = DeeperCNN2().to(device)
model2.load_state_dict(torch.load('complex_model.pth', map_location=device, weights_only=True))
model2.eval()

# Define image transformation
img_width, img_height = 128, 128

data_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['There is no Wind Turbine in the image', 'The image does have a Wind Turbine']

def predict(image: Image.Image, model):
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    predicted_class_name = class_names[predicted_class.item()]
    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    if 'image' not in request.files:
        flash('No file part')
        return render_template('index.html', message='No file uploaded')

    file = request.files['image']

    if file.filename == '':
        flash('No selected file')
        return render_template('index.html', message='No file selected')

    if file:
        # Open the image using PIL
        image = Image.open(file.stream)
        
        if 'model1' in request.form:
            output = predict(image, model1)
        elif 'model2' in request.form:
            output = predict(image, model2)
        else:
            output = 'No model selected'

        return render_template('index.html', message='Prediction:', pred_class=output)

if __name__ == '__main__':
    app.run()



