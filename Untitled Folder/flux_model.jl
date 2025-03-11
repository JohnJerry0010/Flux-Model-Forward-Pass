using Flux
using Images
using CUDA

# Step 1: Define the Flux model architecture (Simple CNN)
function build_model()
    model = Chain(
        Conv((3, 3), 1=>16, pad=1, relu),  # 1 input channel (Grayscale)
        MaxPool((2, 2)),  
        Conv((3, 3), 16=>32, pad=1, relu),  
        MaxPool((2, 2)),  
        Flux.flatten,  
        Dense(32 * 56 * 56, 10),  # Fully connected layer with 10 output classes
        softmax  # Softmax activation for classification
    )
    return model
end

# Step 2: Load the preprocessed grayscale image
function load_preprocessed_image(image_path)
    img = Images.load(image_path)  # Load image
    gray_img = Gray.(img)  # Convert to grayscale
    img_array = Float32.(gray_img) ./ 255.0  # Convert to Float32 and normalize

    return reshape(img_array, size(img_array, 1), size(img_array, 2), 1, 1)  # Correct shape
end


# Step 3: Load the preprocessed image
image_path = "C:/Users/hp/Downloads/preprocessed_img/preprocessed_img/synthetic_img_1.png"
img_tensor = load_preprocessed_image(image_path)

# Step 4: Build the model
model = build_model()

# Step 5: Perform forward pass (run the model on the image)
output = model(img_tensor)

# Step 6: Print the output (Softmax probabilities)
println("🔹 Model Output (Softmax Probabilities): ", output)