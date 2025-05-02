import streamlit as st
import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
def load_image(uploaded_file, max_size=400, shape=None):
    image = Image.open(uploaded_file).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = [shape[-2], shape[-1]]  # (height, width)

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# Display tensor image
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image

# Feature extraction
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Streamlit app
st.title("🎨 Neural Style Transfer with PyTorch")

content_file = st.file_uploader("Upload Content Image", type=['jpg', 'png'])
style_file = st.file_uploader("Upload Style Image", type=['jpg', 'png'])

if content_file and style_file:
    content_img = load_image(content_file)
    style_img = load_image(style_file, shape=content_img.shape[-2:])

    st.image(np.array(Image.open(content_file)), caption="Content Image", width=300)
    st.image(np.array(Image.open(style_file)), caption="Style Image", width=300)

    if st.button("Start Style Transfer"):
        # Load model
        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        # Layers
        content_layer = 'conv4_2'
        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # Features
        content_features = get_features(content_img, vgg)
        style_features = get_features(style_img, vgg)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

        # Initialize target
        target = content_img.clone().requires_grad_(True).to(device)

        # Weights
        style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5,
                         'conv4_1': 0.3, 'conv5_1': 0.1}
        content_weight = 1e4
        style_weight = 1e2

        # Optimizer
        optimizer = optim.Adam([target], lr=0.003)
        steps = 200

        progress_bar = st.progress(0)
        status_text = st.empty()

        for step in range(1, steps + 1):
            optimizer.zero_grad()

            target_features = get_features(target, vgg)

            content_loss = torch.mean((target_features[content_layer] - content_features[content_layer].detach()) ** 2)

            style_loss = 0
            for layer in style_layers:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer].detach()
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (target_feature.shape[1] ** 2)

            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            optimizer.step()

            # Clamp
            with torch.no_grad():
                target.clamp_(0, 1)

            if step % 10 == 0:
                progress_bar.progress(step / steps)
                status_text.text(f"Step {step}/{steps}, Loss: {total_loss.item():.4f}")

        st.subheader("Result Image")
        result_image = tensor_to_image(target)
        st.image(result_image, caption="Stylized Image", use_column_width=True)

else:
    st.info("Upload both content and style images to begin.")

