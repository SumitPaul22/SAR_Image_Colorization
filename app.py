import os
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

categories = ['agri', 'barrenland', 'grassland', 'urban']

# Simple Classifier for image categorization
class Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(Classifier, self).__init__()
        
        # First Convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces size to 128x128
        
        # Second Convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces size to 64x64
        
        # Third Convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Reduces size to 32x32
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Apply convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
# Define DCT Residual Block
class DCTResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCTResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dct = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Placeholder for DCT
        
    def forward(self, x):
        dct_features = self.dct(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + dct_features)

# Define Light-ASPP
class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.atrous_block2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.atrous_block1(x)
        x3 = self.atrous_block2(x)
        x4 = self.atrous_block3(x)
        x5 = self.global_avg_pool(x)
        x5 = self.conv2(x5)
        return x1 + x2 + x3 + x4 + x5

# Define CCMB
class CCMB(nn.Module):
    def __init__(self, in_channels):
        super(CCMB, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, 3)  # Mapping to 3 color channels (RGB)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        color_info = self.fc(x)
        return color_info.view(-1, 3, 1, 1)

# Define Generator with skip connections (U-Net style)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.encoder1 = DCTResidualBlock(1, 32)
        self.encoder2 = DCTResidualBlock(32, 64)
        self.encoder3 = LightASPP(64, 128)
        self.encoder4 = DCTResidualBlock(128, 256)

        # Decoder with skip connections
        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_layer = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  # Final layer to output RGB

    def forward(self, x):
        # Encode
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        encoded = self.encoder4(e3)
        
        # Decoder
        d1 = F.relu(self.decoder1(encoded))
        d2 = F.relu(self.decoder2(d1))
        d3 = F.relu(self.decoder3(d2))
        decoded = self.final_layer(d3)

        # Resize to 256x256 if needed
        decoded = F.interpolate(decoded, size=(256, 256), mode='bilinear', align_corners=True)

        return decoded

# Define the transformations for input SAR images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Same normalization as during training
])

# Load the saved model
checkpoint_path = r"F:\DataD\Trash\Alien\model_epoch_53.pth"  # Path to the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)
classifier = Classifier().to(device)
model = Generator().to(device)
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['generator'])
model.eval()
classifier.load_state_dict(checkpoint['classifier'])
classifier.eval()

# Define unnormalization for displaying images
def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Inverse of normalization
    return tensor.clamp(0, 1)  # Ensure values are in [0, 1]

# Home route for file upload
@app.route('/')
def index():
    return render_template('index.html')

# Function to process large images
def process_large_image(image, tile_size=(256, 256)):
    width, height = image.size
    tiles = []
    for i in range(0, height, tile_size[1]):
        for j in range(0, width, tile_size[0]):
            box = (j, i, min(j + tile_size[0], width), min(i + tile_size[1], height))
            tile = image.crop(box)
            tiles.append((tile, box))  # Store the tile and its box
    return tiles

# Function to reconstruct image from tiles
def reconstruct_image(tiles, output_size):
    output_image = Image.new("RGB", output_size)
    for tile, box in tiles:
        output_image.paste(tile, box)
    return output_image

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            try:
                # Process the uploaded image
                img = Image.open(file).convert('L')  # Convert to grayscale
                original_size = img.size  # Store the original size
                img = img.resize((256, 256))  # Resize the image
                input_tensor = transform(img).unsqueeze(0).to(device)  # Transform and add batch dimension
                
                # print("Input tensor shape:", input_tensor.shape)  # Debugging
                
                # Clear CUDA cache before processing
                torch.cuda.empty_cache()
                
                # Generate the colorized image
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                
                # print("Output tensor shape:", output_tensor.shape)  # Debugging
                
                # Classify the output tensor
                classifier_input = output_tensor[0].unsqueeze(0)  # Ensure it's batch size 1
                classifier_output = classifier(classifier_input.detach())
                
                # Convert logits to probabilities and get the predicted class
                prediction = torch.argmax(classifier_output[0]).item()

                # Unnormalize the output image
                output_tensor = unnormalize(output_tensor[0].cpu(), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                output_image = transforms.ToPILImage()(output_tensor)
                
                # Resize output image to original size
                output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
                
                # Save the image in memory
                img_io = io.BytesIO()
                output_image.save(img_io, 'PNG')
                img_io.seek(0)
                
                print(f'prediction: {categories[prediction]}')
                # Return the colorized image
                return send_file(img_io, mimetype='image/png')
            except Exception as e:
                print(f"Error during image processing: {e}")  # Debugging
                return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)