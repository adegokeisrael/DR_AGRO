import os
import torch
from torch.utils.data import Dataset
from transformers import (
    SiglipProcessor, 
    SiglipModel, 
    TrainingArguments, 
    Trainer, 
    default_data_collator
)
from datasets import load_dataset
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "google/siglip-base-patch16-224" # Base model
OUTPUT_DIR = "./siglip-agric-finetuned"
DATA_FILE = "path/to/your/dataset.csv"      # CSV with 'image_path' and 'caption'
IMAGE_COLUMN = "image_path"
TEXT_COLUMN = "caption"
BATCH_SIZE = 8  # Adjust based on your VRAM (SigLIP is heavy)
EPOCHS = 3
LEARNING_RATE = 5e-6 # Low LR is crucial for fine-tuning pre-trained models

# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class AgricImageTextDataset(Dataset):
    """
    Custom Dataset to handle loading images and tokenizing text on the fly.
    """
    def __init__(self, data_file, processor, root_dir=None):
        # Load data using HuggingFace Datasets for efficiency
        self.dataset = load_dataset("csv", data_files=data_file)["train"]
        self.processor = processor
        self.root_dir = root_dir # If image paths in CSV are relative

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Load Image
        img_path = item[IMAGE_COLUMN]
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        try:
            # Convert to RGB to ensure 3 channels (handling Greyscale/RGBA)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image in case of error to prevent crash
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 2. Load Text
        text = item[TEXT_COLUMN]

        # 3. Process using SigLIP Processor (Handles resizing, normalizing, tokenizing)
        # padding="max_length" ensures consistent tensor shapes for batching
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            max_length=64 # Short captions usually suffice for Agric data
        )

        # Remove the batch dimension added by the processor (1, C, H, W) -> (C, H, W)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs

# ==========================================
# 3. SETUP & TRAINING
# ==========================================
def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Processor and Model
    print("Loading Model and Processor...")
    processor = SiglipProcessor.from_pretrained(MODEL_ID)
    model = SiglipModel.from_pretrained(MODEL_ID)
    
    # Freeze vision encoder layers (Optional strategy)
    # Often getting good results involves freezing the lower layers and only training 
    # the top layers and the projection heads to save memory and prevent catastrophic forgetting.
    # for param in model.vision_model.parameters():
    #     param.requires_grad = False

    # Prepare Dataset
    print("Preparing Dataset...")
    train_dataset = AgricImageTextDataset(DATA_FILE, processor)

    # Define Data Collator
    # This function stacks the dictionary of tensors into a batch
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        remove_unused_columns=False, # Important for custom datasets
        push_to_hub=False,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True if torch.cuda.is_bf16_supported() else False, # Use BF16 if Ampere GPU
        fp16=False if torch.cuda.is_bf16_supported() else True, # Else use FP16
        dataloader_num_workers=4,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # Start Training
    print("Starting Training...")
    trainer.train()

    # Save the final fine-tuned model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()