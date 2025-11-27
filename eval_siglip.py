import torch
from transformers import SiglipProcessor, SiglipModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "./siglip-agric-finetuned" # Path to your saved model now
TEST_DATA_FILE = "path/to/your/test_dataset.csv" # Separate test set
BATCH_SIZE = 16

# ==========================================
# 2. EVALUATION DATASET
# ==========================================
class EvalDataset(Dataset):
    def __init__(self, data_file, root_dir=None):
        self.df = pd.read_csv(data_file)
        self.root_dir = root_dir
        self.images = self.df['image_path'].tolist()
        self.captions = self.df['caption'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        caption = self.captions[idx]
        return image, caption

# ==========================================
# 3. METRIC CALCULATION
# ==========================================
def compute_recall_at_k(similarity_matrix, k_vals=[1, 5, 10]):
    """
    Computes Recall@K for Image-to-Text retrieval.
    Args:
        similarity_matrix: (N_images, N_texts) matrix of dot products.
                           Assumes diagonal elements (i, i) are the ground truth matches.
    """
    num_samples = similarity_matrix.shape[0]
    results = {}
    
    # Get indices of sorted scores (descending)
    # topk indices: (N, N)
    _, indices = torch.topk(similarity_matrix, k=max(k_vals), dim=1)
    
    # Ground truth is simply the index itself (0 for 0th row, 1 for 1st row...)
    ground_truth = torch.arange(num_samples, device=similarity_matrix.device).view(-1, 1)
    
    for k in k_vals:
        # Check if ground truth index is within the top k predictions
        # We look at the first k columns of the indices matrix
        hits = (indices[:, :k] == ground_truth).any(dim=1)
        recall = hits.float().mean().item()
        results[f"R@{k}"] = recall
        
    return results

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_PATH} on {device}...")
    
    model = SiglipModel.from_pretrained(MODEL_PATH).to(device)
    processor = SiglipProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    dataset = EvalDataset(TEST_DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_image_embeds = []
    all_text_embeds = []

    print("Generating Embeddings...")
    with torch.no_grad():
        for images, texts in tqdm(dataloader):
            # Process inputs
            inputs = processor(
                text=texts, 
                images=images, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt",
                max_length=64
            ).to(device)

            # Get Embeddings
            outputs = model(**inputs)
            
            # Normalize embeddings (Critical for Cosine Similarity/Dot Product)
            # SigLIP outputs are usually already normalized, but we ensure it here.
            img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

            all_image_embeds.append(img_embeds)
            all_text_embeds.append(txt_embeds)

    # Concatenate all batches
    all_image_embeds = torch.cat(all_image_embeds)
    all_text_embeds = torch.cat(all_text_embeds)

    print(f"Embeddings shape: {all_image_embeds.shape}")

    # Compute Similarity Matrix (Dot Product)
    # Shape: (num_samples, num_samples)
    print("Computing Similarity Matrix...")
    logits_per_image = torch.matmul(all_image_embeds, all_text_embeds.t())
    
    # SigLIP usually includes a learned temperature (logit_scale) and bias.
    # For raw retrieval ranking based on cosine similarity, pure dot product is sufficient,
    # but applying the scale makes the distribution sharper.
    logit_scale = model.logit_scale.exp()
    logits_per_image = logits_per_image * logit_scale + model.logit_bias

    # Compute Metrics
    print("Calculating Recall@K...")
    metrics = compute_recall_at_k(logits_per_image, k_vals=[1, 5, 10])

    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v*100:.2f}%")
        
    # Optional: Diagonal accuracy (How often is the correct caption the absolute #1 match?)
    # This is essentially R@1 but formatted differently
    labels = torch.arange(len(all_image_embeds)).to(device)
    acc = (logits_per_image.argmax(dim=1) == labels).float().mean()
    print(f"Exact Match Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()