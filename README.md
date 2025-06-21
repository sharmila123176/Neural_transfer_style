# Neural_transfer_style
COMPANY NAME : CODTECH IT SOLUTIONS

NAME : PULAMARASETTI SHARMILA

INTERN ID : CODF88

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 4WEEKS

# SMALL DISCRIPTION OF THE NEURAL STYLE TRANSFER :

This Streamlit application performs Neural Style Transfer using a pretrained VGG19 model from PyTorch. The user uploads a content image (such as a regular photograph) and a style image (such as a painting or artistic texture). The app preprocesses these images by resizing and normalizing them to match the input requirements of the VGG19 model. Then, it builds a custom model by extracting layers from VGG19 and inserting special modules that compute content loss and style loss. Content loss ensures that the generated image preserves the important structures of the content image, while style loss ensures that the textures and colors resemble those of the style image. The image generation is treated as an optimization problem, where the input image pixels are adjusted using the LBFGS optimizer to minimize a weighted combination of content and style losses. After several optimization steps, the final stylized image is displayed alongside the original content and style images. This project uses key libraries such as Streamlit for the interface, PyTorch and Torchvision for model handling, PIL for image loading, and NumPy for tensor operations. The entire app can be easily installed by setting up the environment with a few dependencies and can be launched with a single streamlit run command.


 # Technologies Used
 Frontend / User Interface
 Deep Learning Framework
 Image Processing
 Optimization
 Environment & Deployment
 Optional / Supporting Tools


 # OUTPUT :
 ![Screenshot (30)](https://github.com/user-attachments/assets/30a70be3-0718-4e0d-96f5-08f0399eb780)
 ![Screenshot (31)](https://github.com/user-attachments/assets/319bb6bb-2cec-42a4-99c6-7ecc2d2944c9)
 ![Screenshot (32)](https://github.com/user-attachments/assets/af3943e3-79b7-40ab-a804-8a24a90fd659)
 ![Screenshot (33)](https://github.com/user-attachments/assets/4530097b-707d-4669-ba2e-85a6bb8a5fc3)
 ![Screenshot (34)](https://github.com/user-attachments/assets/5a5dfc77-aa77-40f1-8e09-a7be09285e84)
