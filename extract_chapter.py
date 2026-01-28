import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

# --- CONFIGURATION ---
MODEL_PATH = "./magiv2"  
INPUT_DIR = "input_pages"
OUTPUT_DIR = "extracted_panels"
PADDING = -10  # Negative = Crop INWARDS (removes black borders)

def main():
    # 1. Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Status: Using {device.upper()} for processing.")

    # 2. Load the Local Model
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

    # 3. Prepare Directories
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"⚠️  Created folder '{INPUT_DIR}'. Please drop your manga images inside and run again.")
        return

    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    
    if not image_files:
        print(f"⚠️  No images found in '{INPUT_DIR}'!")
        return

    # 4. Process Every Page
    print(f"🔍 Found {len(image_files)} pages. Starting extraction...\n")

    for filename in image_files:
        img_path = os.path.join(INPUT_DIR, filename)
        page_name = os.path.splitext(filename)[0]
        page_out_dir = os.path.join(OUTPUT_DIR, page_name)
        
        try:
            pil_image = Image.open(img_path).convert("RGB")
            image_np = np.array(pil_image)
        except Exception as e:
            print(f"   Skipping {filename}: {e}")
            continue

        # Run MAGI Prediction
        with torch.no_grad():
            results = model.predict_detections_and_associations([image_np])
            panels = results[0]['panels']

        if not panels:
            print(f"   [Page {filename}] No panels detected.")
            continue

        # Save the Crops
        if not os.path.exists(page_out_dir):
            os.makedirs(page_out_dir)

        print(f"   [Page {filename}] Saving {len(panels)} panels...")
        
        for i, box in enumerate(panels):
            x1, y1, x2, y2 = map(int, box)
            
            # Apply Padding
            # Negative padding (-5) increases x1 and decreases x2 (cropping inwards)
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(pil_image.width, x2 + PADDING)
            y2 = min(pil_image.height, y2 + PADDING)
            
            # Safety Check: If crop is too aggressive and inverts the box, skip it
            if x2 <= x1 or y2 <= y1:
                print(f"      Warning: Panel {i+1} was too small to crop with padding.")
                continue

            crop = pil_image.crop((x1, y1, x2, y2))
            
            save_name = f"panel_{i+1:02d}.png"
            crop.save(os.path.join(page_out_dir, save_name))

    print("\n🎉 Extraction Complete! Check the 'extracted_panels' folder.")

if __name__ == "__main__":
    main()