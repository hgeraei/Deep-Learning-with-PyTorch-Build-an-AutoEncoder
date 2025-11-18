📘 Deep Learning with PyTorch — AutoEncoder for Image Denoising

This project demonstrates how to build and train a convolutional autoencoder using PyTorch for the task of image denoising.
The model learns to reconstruct clean MNIST images from artificially generated noisy inputs.

🧠 What is an AutoEncoder?

An autoencoder is a neural network that learns to compress data (encoder) and then reconstruct it (decoder).
It is widely used for:

Image denoising

Segmentation

Compression

Super-resolution

Feature learning

<p align="center"> <img src="denoising_autoencoder.png" width="500"> </p>
📂 Dataset

The notebook uses:

mnist_dataset.csv


Each row represents a flattened 28×28 grayscale MNIST image.
Images are reshaped and normalized before training.
Random Gaussian noise is added:

noisy_img = img + noise_factor * np.random.randn(*img.shape)

🧰 Project Contents
autoencoder/
│
├── autoencoder.ipynb        # Main notebook
├── mnist_dataset.csv        # Flattened MNIST images
├── helper.py                # Utilities (show_image, transforms)
├── denoising_autoencoder.png
├── dataset.png
├── README.md
└── requirements.txt

🧪 Model Architecture
Encoder
nn.Conv2d(1, 16, 3, padding=1)
nn.ReLU()
nn.MaxPool2d(2)
nn.Conv2d(16, 32, 3, padding=1)
nn.ReLU()
nn.MaxPool2d(2)

Decoder
nn.ConvTranspose2d(32, 16, 2, stride=2)
nn.ReLU()
nn.ConvTranspose2d(16, 1, 2, stride=2)
nn.Sigmoid()

🏋️ Training the AutoEncoder

Loss:

criterion = nn.MSELoss()


Optimizer:

optimizer = optim.Adam(model.parameters(), lr=0.001)


Training loop:

for noisy_image, image in trainloader:
    pred = model(noisy_image)
    loss = criterion(pred, image)

📊 Results

After training, the notebook shows:

Noisy input

Original image

Denoised reconstruction

Example usage:

show_image(noisy_image[index], image[index], pred_image)

▶️ How to Run
1. Install dependencies

Create a requirements.txt file:

torch
torchvision
numpy
pandas
matplotlib
tqdm
torchsummary


Install them:

pip install -r requirements.txt

2. Launch Jupyter Notebook
jupyter notebook autoencoder.ipynb


Run all cells to train and test the autoencoder.

💡 Applications

Autoencoders can be used for:

Image denoising

Anomaly detection

Latent feature extraction

Image compression

Data pre-processing

📜 License

MIT License (or modify as needed)

If you want, I can also generate:

📦 A zipped ready-made repo
📄 A more visual README (with GIFs)
✨ A version for public portfolio use

Just ask!
