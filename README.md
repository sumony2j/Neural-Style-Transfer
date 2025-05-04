# Neural Style Transfer

This repository implements **Neural Style Transfer** (NST) using a pre-trained model from TensorFlow Hub. The model is used to apply the style of one image (typically an artwork) onto the content of another image (usually a photograph). This process enables the creation of stunning art by blending the content and style of two different images.

## 🔍 What is Neural Style Transfer?

**Neural Style Transfer** is a deep learning technique that merges the content of one image with the style of another image. This method is based on convolutional neural networks (CNNs) and works by minimizing the difference between the **content representation** of the content image and the **style representation** of the style image.

In simpler terms, NST allows you to take a photograph (content) and apply the visual style of a famous painting (style) to create a new image that combines both elements.

## 🧠 Model Used

This implementation uses a pre-trained **Arbitrary Image Stylization** model from TensorFlow Hub:

- **Model URL**: [TensorFlow Hub: Arbitrary Image Stylization](https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/TensorFlow1/variations/256/versions/2)
- The model is trained to transfer style between arbitrary content and style images using a fast and efficient approach.

## ⚙️ How It Works

### **Mathematical Foundation of Neural Style Transfer**

Neural Style Transfer works by minimizing a loss function that balances two objectives:

1. **Content Loss**: Ensures that the content of the output image remains similar to the content image.
2. **Style Loss**: Ensures that the style of the output image remains similar to the style image.

The loss function is defined as:

**L_total = α * L_content + β * L_style**

Where:
- `L_content` is the content loss.
- `L_style` is the style loss.
- `α` and `β` are hyperparameters that control the influence of each loss.

#### **1. Content Loss**

Content loss compares the feature maps (representations) of the content image and the generated image at a specific layer of the neural network. It is defined as:

**L_content = 1/2 * Σ (C_ij - G_ij)^2**

Where:
- `C_ij` is the content feature map of the content image at a specific layer.
- `G_ij` is the feature map of the generated image at the same layer.

The goal is to minimize the difference between the content image and the generated image's features.

#### **2. Style Loss**

Style loss compares the correlation between different features (the Gram matrix) of the style image and the generated image. The Gram matrix captures the texture and patterns in the style image. It is defined as:

**L_style = 1 / (4 * N^2 * M^2) * Σ (G^S_ij - G^T_ij)^2**

Where:
- `G^S` is the Gram matrix of the style image.
- `G^T` is the Gram matrix of the generated image.
- `N` is the number of feature maps, and `M` is the number of pixels.

The goal is to minimize the difference between the style image and the generated image's style.

## 🧩 Requirements

To run the Neural Style Transfer code, you need to install the following dependencies:

```bash
pip install tensorflow tensorflow-hub matplotlib numpy
```

## 📊 Example Usage

To run Neural Style Transfer with your own images:

1. **Load the model and images.**
2. **Apply style transfer** to your content image using the style image.
3. **Display the results.**

The code is implemented in a Jupyter notebook `Neural_Style_Transfer.ipynb`. You can load your content and style images and visualize the results.

## 📁 Project Structure

- `Neural_Style_Transfer.ipynb` - The Jupyter notebook containing the entire implementation.
- `README.md` - This file.

## 🛠️ Framework

- **TensorFlow 1.x**
- **TensorFlow Hub** for model loading
- **Matplotlib** for displaying images
- **NumPy** for image handling

## ✅ Summary

This project demonstrates how to perform Neural Style Transfer by leveraging a pre-trained TensorFlow model. By blending the content of one image with the style of another, you can create unique and artistic images.



