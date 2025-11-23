import os

class Config:
    # ==========================
    # PATHS
    # ==========================
    # Root directory for images (leave empty if CSV has absolute paths)
    IMAGE_ROOT = "./data/images" 
    
    # Path to the CSV files
    TRAIN_DATA_PATH = "./data/train_dataset.csv"
    TEST_DATA_PATH = "./data/test_dataset.csv"
    
    # Where to save the model
    OUTPUT_DIR = "./output/siglip-agric-finetuned"
    
    # ==========================
    # MODEL HYPERPARAMETERS
    # ==========================
    MODEL_ID = "google/siglip-base-patch16-224"
    MAX_LENGTH = 64  # Max token length for captions
    
    # ==========================
    # TRAINING SETTINGS
    # ==========================
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    seed = 42
    
    # Number of workers for data loading (adjust based on CPU cores)
    NUM_WORKERS = 4 

    @classmethod
    def create_output_dir(cls):
        """Creates the output directory if it doesn't exist."""
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
            print(f"Created output directory: {cls.OUTPUT_DIR}")