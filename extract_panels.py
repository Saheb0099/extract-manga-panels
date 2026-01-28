import warnings
# Silence "FutureWarning" noise to keep terminal clean
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import re
import gc
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

# ==========================================
#               CONFIGURATION
# ==========================================
MODEL_PATH = "./magiv2"  
INPUT_DIR = "input_manga"       
SUBFOLDER_NAME = "cleaned"      
OUTPUT_DIR = "extracted_panels"
PADDING = -10  # Negative = Crop INWARDS (removes borders)
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')
# ==========================================


def extract_page_number(filename):
    """Finds the first number in a filename (e.g. '01.jpg' -> 1)."""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def get_optimal_device():
    """Auto-selects the fastest hardware (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clean_memory(device):
    """
    Forces the hardware to release unused memory.
    Run this periodically to prevent 'Out of Memory' crashes.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    elif device == "cpu":
        gc.collect()


def main():
    # 1. Setup Hardware
    device = get_optimal_device()
    print(f"🚀 Status: Hardware acceleration set to [{device.upper()}]")

    # 2. Load Model
    print(f"📂 Loading MAGI v2 model from {MODEL_PATH}...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            local_files_only=True
        ).to(device).eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Critical Error: Could not load model.\n   Details: {e}")
        return

    # 3. Check Input Directory
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"⚠️  Folder '{INPUT_DIR}' created.")
        print(f"   Please structure files as: {INPUT_DIR}/<ChapterFolder>/{SUBFOLDER_NAME}/<images>")
        return

    # 4. Find Chapters
    chapter_folders = sorted([f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))])
    
    if not chapter_folders:
        print(f"⚠️  No chapter folders found in '{INPUT_DIR}'!")
        return

    print(f"📚 Found {len(chapter_folders)} chapters. Starting job...\n")

    # 5. Process Loop
    for chapter_name in chapter_folders:
        chapter_cleaned_path = os.path.join(INPUT_DIR, chapter_name, SUBFOLDER_NAME)
        chapter_out_path = os.path.join(OUTPUT_DIR, chapter_name)
        
        if not os.path.exists(chapter_cleaned_path):
            print(f"🔸 Skipping '{chapter_name}': Missing '{SUBFOLDER_NAME}' folder.")
            continue

        print(f"🔹 Processing Chapter: {chapter_name}")

        image_files = sorted([f for f in os.listdir(chapter_cleaned_path) if f.lower().endswith(VALID_EXTENSIONS)])
        
        if not image_files:
            print(f"   ⚠️  No images found in {chapter_cleaned_path}")
            continue

        if not os.path.exists(chapter_out_path):
            os.makedirs(chapter_out_path)

        # Iterate through images with a counter
        for count, filename in enumerate(image_files):
            img_path = os.path.join(chapter_cleaned_path, filename)
            page_num = extract_page_number(filename)
            
            try:
                pil_image = Image.open(img_path).convert("RGB")
                image_np = np.array(pil_image)
            except Exception as e:
                print(f"   Skipping corrupt file {filename}: {e}")
                continue

            # --- PREDICTION ---
            with torch.no_grad():
                results = model.predict_detections_and_associations([image_np])
                panels = results[0]['panels']

            if not panels:
                print(f"   [Page {page_num:03d}] No panels detected.")
                continue

            print(f"   [Page {page_num:03d}] Saving {len(panels)} panels...")
            
            # --- CROP & SAVE ---
            for i, box in enumerate(panels):
                x1, y1, x2, y2 = map(int, box)
                
                # Apply Negative Padding
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(pil_image.width, x2 + PADDING)
                y2 = min(pil_image.height, y2 + PADDING)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = pil_image.crop((x1, y1, x2, y2))
                save_name = f"{page_num:03d}.{i+1:03d}.png"
                crop.save(os.path.join(chapter_out_path, save_name))

            # --- SMART MEMORY CLEANER ---
            # Every 20 pages, flush the memory to prevent leaks
            if (count + 1) % 20 == 0:
                clean_memory(device)
                # Optional: Print a tiny dot to show cleanup happened
                # print(".", end="", flush=True) 

        print(f"✅ Finished {chapter_name}\n")

    print("🎉 All chapters processed successfully!")

if __name__ == "__main__":
    main()