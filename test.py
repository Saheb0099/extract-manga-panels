import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

# --- CONFIGURATION ---
MODEL_PATH = "./magiv2"  
INPUT_DIR = "input_manga"       # Top level folder
SUBFOLDER_NAME = "cleaned"      # The specific subfolder where images live
OUTPUT_DIR = "extracted_panels"
PADDING = -10  # Negative = Crop INWARDS (removes borders)

def main():
    # 1. Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Status: Using {device.upper()} for processing.")

    # 2. Load Local Model
    print(f"📂 Loading MAGI v2 from {MODEL_PATH}...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            local_files_only=True
        ).to(device).eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Check Input Directory
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"⚠️  Created folder '{INPUT_DIR}'.")
        print(f"   Structure expected: {INPUT_DIR}/<ChapterNumber>/{SUBFOLDER_NAME}/<images>")
        return

    # 4. Find Chapter Folders
    # We look for any folder inside input_manga (e.g., "1", "Chapter 2")
    chapter_folders = sorted([f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))])
    
    if not chapter_folders:
        print(f"⚠️  No chapter folders found in '{INPUT_DIR}'!")
        return

    print(f"📚 Found {len(chapter_folders)} chapters. Looking for '{SUBFOLDER_NAME}' folders inside them...\n")

    # 5. Process Each Chapter
    for chapter_name in chapter_folders:
        # Construct path: input_manga/1/cleaned
        chapter_cleaned_path = os.path.join(INPUT_DIR, chapter_name, SUBFOLDER_NAME)
        
        # Output path: extracted_panels/1
        chapter_out_path = os.path.join(OUTPUT_DIR, chapter_name)
        
        # Check if "cleaned" folder actually exists
        if not os.path.exists(chapter_cleaned_path):
            print(f"🔸 Skipping '{chapter_name}': Could not find '{SUBFOLDER_NAME}' folder inside.")
            continue

        print(f"🔹 Processing Chapter: {chapter_name}")

        image_files = sorted([f for f in os.listdir(chapter_cleaned_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        if not image_files:
            print(f"   ⚠️  No images found in {chapter_cleaned_path}")
            continue

        for filename in image_files:
            img_path = os.path.join(chapter_cleaned_path, filename)
            page_name = os.path.splitext(filename)[0]
            
            # Result: extracted_panels/1/Page_01/
            page_out_dir = os.path.join(chapter_out_path, page_name)
            
            try:
                pil_image = Image.open(img_path).convert("RGB")
                image_np = np.array(pil_image)
            except Exception as e:
                print(f"   Skipping {filename}: {e}")
                continue

            # Run MAGI
            with torch.no_grad():
                results = model.predict_detections_and_associations([image_np])
                panels = results[0]['panels']

            if not panels:
                print(f"   [Page {filename}] No panels detected.")
                continue

            if not os.path.exists(page_out_dir):
                os.makedirs(page_out_dir)

            print(f"   [Page {filename}] Saving {len(panels)} panels...")
            
            for i, box in enumerate(panels):
                x1, y1, x2, y2 = map(int, box)
                
                # Apply Padding
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(pil_image.width, x2 + PADDING)
                y2 = min(pil_image.height, y2 + PADDING)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = pil_image.crop((x1, y1, x2, y2))
                save_name = f"panel_{i+1:02d}.png"
                crop.save(os.path.join(page_out_dir, save_name))
        
        print(f"✅ Finished {chapter_name}\n")

    print("🎉 All chapters processed!")

if __name__ == "__main__":
    main()