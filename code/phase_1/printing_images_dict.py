import numpy as np
import matplotlib.pyplot as plt

def print_images(image_dict, heading, target_size=(224, 224)):
    num_images = len(image_dict)
    if num_images == 0:
        return
    
    if num_images == 10:
        cols = 5
        rows = 2
    else:
        cols = 3
        rows = 2
        
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i, (image_id, info_dict) in enumerate(image_dict.items()):
        col = i % cols
        row = i // cols
        image = info_dict['image']
        distance = info_dict['score']
        resized_image = image.resize(target_size)
        image_array = np.array(resized_image)
        axes[row, col].imshow(image_array)
        axes[row, col].set_title(f"ID: {image_id}\nSim/Dist Score: {distance:.2f}")
        axes[row, col].axis("off")

    fig.suptitle(heading, fontsize=16)
    plt.tight_layout()
    plt.show()
