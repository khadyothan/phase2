import torch
import torchvision.transforms as transforms
from torchvision import models

# Function to get intermediate layer output using a hook
def getHook(model, layer_name):
    intermediate_outputs = []
    def hook_fn(module, input, output):
        intermediate_outputs.append(output)

    target_layer = None
    
    # Find the target layer by name in the model's named modules
    for name, layer in model.named_modules():
        if name == layer_name:
            target_layer = layer

    if target_layer is not None:
        hook = target_layer.register_forward_hook(hook_fn)
        return intermediate_outputs, hook
    else:
        return None  

# Function to extract ResNet features from an image
def resnet_features(image):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  
    
    # Initialize lists to store intermediate layer outputs and hooks
    outputs = []
    
    # Apply transformations to the input image
    image = transform(image).unsqueeze(0)
    
    # Get hooks for the desired layers
    avgpool_outputs, hook = getHook(model, "avgpool")
    layer3_outputs = getHook(model, "layer3")
    fc_outputs = getHook(model, "fc")
    
    if hook is not None:
        with torch.no_grad():
            # Forward pass through the model
            output = model(image)
            
            # Extract intermediate layer outputs
            avgpool_feature_descriptor = avgpool_outputs[0]
            layer3_feature_descriptor = layer3_outputs[0]
            fc_feature_descriptor = fc_outputs[0]
    
    # Return the extracted feature descriptors
    return layer3_feature_descriptor[0].view(1024, -1).mean(dim=1).tolist(), avgpool_feature_descriptor[0].view(-1, 2).mean(dim=1).tolist(), fc_feature_descriptor[0].squeeze().tolist()
