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



