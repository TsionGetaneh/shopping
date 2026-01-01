from PIL import Image
import torchvision.transforms as transforms
import torch

# Convert image to tensor
def preprocess_image(image_input, image_size=(256, 192)):
    """Preprocess image - accepts either file path (str) or PIL Image object.
    
    Args:
        image_input: File path (str) or PIL Image
        image_size: (height, width) tuple for tensor dimensions. Default (256, 192)
    
    Returns:
        Tensor of shape [1, 3, H, W] where H=256, W=192 (normalized to [-1, 1])
    """
    if isinstance(image_input, str):
        # File path
        image = Image.open(image_input).convert("RGB")
    else:
        # Already a PIL Image
        image = image_input.convert("RGB")
    
    # Resize: PIL uses (width, height), but image_size is (height, width) for tensor
    # So we need to swap: resize to (W, H) = (192, 256) for PIL
    image = image.resize((image_size[1], image_size[0]))  # PIL (W, H) = (192, 256)
    transform = transforms.Compose([
        transforms.ToTensor(),           # convert to tensor [C, H, W] = [3, 256, 192]
        transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1,1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # add batch dim: [1, 3, 256, 192]
    return image_tensor

# Convert tensor back to PIL image
def tensor_to_image(tensor):
    """Convert tensor [1, C, H, W] to PIL Image.
    
    Args:
        tensor: Tensor in range [-1, 1] or [0, 1]
    
    Returns:
        PIL Image in RGB format
    """
    # Remove batch dimension and detach from computation graph
    tensor = tensor.squeeze(0).detach().cpu()
    
    # Check if tensor is in [-1, 1] range (normalized) or [0, 1] range
    if tensor.min() < 0:
        # Normalized to [-1, 1], convert to [0, 1]
        tensor = tensor * 0.5 + 0.5
    
    # Clamp to valid range [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    image = transforms.ToPILImage()(tensor)
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image
