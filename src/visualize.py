import matplotlib.pyplot as plt
import random
import pandas as pd
import os
from PIL import Image
from config import Config

def visualize_dataset_samples(csv_path, img_root, num_samples=3):
    """
    Loads random samples from the dataset and displays the image with its caption.
    """
    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Check columns
    if 'image_path' not in df.columns or 'caption' not in df.columns:
        print("Error: CSV must contain 'image_path' and 'caption' columns.")
        return

    # Pick random samples
    indices = random.sample(range(len(df)), min(len(df), num_samples))
    
    # Create Plot
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if num_samples == 1: axes = [axes] # Handle single sample edge case

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        img_name = row['image_path']
        caption = row['caption']
        
        full_img_path = os.path.join(img_root, img_name)
        
        try:
            image = Image.open(full_img_path).convert("RGB")
            axes[i].imshow(image)
            axes[i].axis("off")
            
            # Wrap text nicely
            wrapped_caption = "\n".join([caption[j:j+40] for j in range(0, len(caption), 40)])
            axes[i].set_title(wrapped_caption, fontsize=10)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, "Image Not Found", ha='center')
            print(f"Could not load {full_img_path}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Visualizing Training Data...")
    visualize_dataset_samples(Config.TRAIN_DATA_PATH, Config.IMAGE_ROOT)