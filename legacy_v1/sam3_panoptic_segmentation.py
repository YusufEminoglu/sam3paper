import os
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# IMPORTANT: Provide your Hugging Face Token here to utilize the SAM 3 model framework.
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "sam3_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sam3_masks(hf_token=HF_TOKEN):
    print("Initializing Meta's Segment Anything Model 3 (SAM 3) Framework...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device Interface: {device.upper()}")

    print("Loading SAM 3 from HuggingFace Architecture (This may take initialization time)...")
    try:
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(device)
    except Exception as e:
        print("ERROR: Authorization failed. Please ensure a valid Hugging Face token is provided.")
        return

    # Standardized Academic Categorical Palette (R, G, B, Alpha/Opacity)
    CATEGORIES = {
        'person': (0, 100, 255, 140),          # BLUE
        'car': (255, 0, 0, 140),               # RED
        'truck': (255, 0, 0, 140),             # RED
        'motorcycle': (255, 0, 0, 140),        # RED
        'chair': (255, 140, 0, 140),           # ORANGE (Commercial Encroachment)
        'table': (255, 140, 0, 140),           # ORANGE
        'potted plant': (0, 255, 0, 140),      # GREEN (Urban Furniture)
        'trash can': (128, 128, 128, 140),     # GRAY 
        'dumpster': (128, 128, 128, 140),      # GRAY
        'utility pole': (128, 0, 128, 140),    # PURPLE 
        'street light': (128, 0, 128, 140),    # PURPLE
        'utility box': (128, 0, 128, 140),     # PURPLE
        'electrical cabinet': (128, 0, 128, 140)# PURPLE
    }

    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    total_images = len(image_files)

    print(f"\nProcessing {total_images} observational nodes using Zero-Shot capabilities...\n")

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMG_DIR, img_name)
        out_path = os.path.join(OUTPUT_DIR, f"sam3_{img_name}")

        # Skipping dynamically if already processed
        if os.path.exists(out_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Trigger zero-shot capabilities iterating over analytical categories
            for label_name, color in CATEGORIES.items():
                inputs = processor(images=img, text=label_name, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]

                masks = results["masks"].cpu().numpy()
                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()

                for i in range(len(masks)):
                    if scores[i] > 0.60: # Threshold Confidence Metric
                        # 1. Pixel Extrapolation (SAM 3 Mask overlay)
                        mask = masks[i]
                        solid_color = Image.new('RGBA', img.size, color)
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                        overlay.paste(solid_color, (0, 0), mask_img)

                        # 2. Semantic Labelling
                        box = boxes[i]
                        xmin, ymin, xmax, ymax = box
                        display_text = f"{label_name.upper()} ({int(scores[i] * 100)}%)"
                        text_length = len(display_text) * 6

                        draw.rectangle([(xmin, max(0, ymin - 15)), (xmin + text_length, max(15, ymin))],
                                       fill=color[:3] + (255,))
                        draw.text((xmin + 2, max(0, ymin - 15)), display_text, fill="white")

            # Compound raster composite and save structured artifact
            final_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            final_img.save(out_path)

            print(f"[{idx + 1}/{total_images}] Segmented: {img_name}")

        except Exception as e:
            print(f"Analysis iteration failed on ({img_name}): {e}")

    print(f"\nZero-Shot Semantic Segmentation successfully preserved under '{OUTPUT_DIR}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zero-Shot Segregation Script.')
    parser.add_argument('--token', type=str, default=HF_TOKEN, help='HuggingFace Auth Token')
    args = parser.parse_args()
    generate_sam3_masks(args.token)
