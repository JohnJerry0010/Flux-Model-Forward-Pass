# **Synthetic Image Generation, Preprocessing, and Flux Model Forward Pass**

##  Objective  
This project demonstrates the generation of synthetic images using Stable Diffusion, image preprocessing, and a **Flux-based neural network** in Julia to perform a forward pass.  

---

##  **Setup Instructions**  

### ** Clone the Repository**  
```bash
git clone https://github.com/JohnJerry0010/Flux-Model-Forward-Pass.git
cd Flux-Model-Forward-Pass


##  **Python (For Image Generation & Preprocessing)##  **
pip install diffusers opencv-python numpy torch torchvision

Open Julia and run:
using Pkg
Pkg.add(["Flux", "Images", "CUDA"])

Run the Flux model:
cd("C:/Users/hp/Untitled Folder")
include("flux_model.jl")



 Project Breakdown

1️⃣ Synthetic Image Generation

🔹 Description:

This module focuses on generating synthetic images using Stable Diffusion, a state-of-the-art deep learning model for text-to-image generation. The goal is to create high-quality images based on user-defined prompts.

🔹 Approach:

Integrated the diffusers library to access the Stable Diffusion model.

Selected a descriptive text prompt (e.g., "A futuristic city at sunset").

Generated and saved three images based on the input prompt.

Stored the images in the synthetic_images/ directory for further processing.

🔹 Tools Used:

Python: Programming language used to implement the image generation.

Hugging Face diffusers: Provides pre-trained Stable Diffusion models.

OpenCV: Used for image manipulation and file handling.

2️⃣ Image Preprocessing

🔹 Description:

This step refines the generated images to prepare them for use in a Flux-based deep learning model. Preprocessing ensures that the images are of consistent size and format, allowing efficient input handling by the model.

🔹 Steps:

✔ Resizing: Each image is resized to 224×224 pixels to maintain uniformity.✔ Normalization: Pixel values are scaled between 0 and 1 to improve model performance.✔ Grayscale Conversion: The images are converted to grayscale, reducing complexity and computational cost.

🔹 Tools Used:

Python: For implementing preprocessing logic.

OpenCV: Used to resize, normalize, and convert images to grayscale.

NumPy: Utilized for handling image data arrays efficiently.

3️⃣ Minimal Flux Model: Forward Pass

🔹 Description:

A simple Convolutional Neural Network (CNN) is implemented using Flux.jl, a deep learning framework in Julia. The preprocessed images are passed through the model to generate predictions.

🔹 Model Architecture:

✔ Convolutional Layer → Activation (ReLU) → Dense Layer → Softmax Output

🔹 Steps:

✔ Loading Preprocessed Images: The images are read into Julia and formatted as tensors.✔ Tensor Conversion: Images are transformed into the correct shape for Flux processing.✔ Forward Pass Execution: The model processes the image and outputs a probability distribution over possible classes.

🔹 Output:

A Softmax probability distribution representing the likelihood of different classes based on the image input.

🔹 Tools Used:

Julia: The programming language used for implementing the model.

Flux.jl: A deep learning library for Julia.

CUDA (optional): Used for GPU acceleration to improve model performance.

⚠️ Challenges Faced

✔ Dimension Errors: Addressed mismatches in the input shape, ensuring compatibility between image tensors and Flux.✔ Working with Julia: Debugged model structure and refined tensor operations.✔ CUDA Compatibility: Ensured cuDNN installation for leveraging GPU acceleration when available.
