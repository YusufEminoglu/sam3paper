import os
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "pixel_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pixel_masks():
    print("Initiating Baseline Panoptic Segmentation Protocol...")
    print("Pre-loading Object Detection Architecture (Panoptic Masking)...")

    # Employ baseline semantic pixel segmentation logic
    segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

    # Semantic Categorical Mask Rendering Variables (R, G, B, Alpha)
    COLORS = {
        'person': (0, 100, 255, 140),          # Flow dynamics (non-encroachment)
        'car': (255, 0, 0, 140),               # Red obstruction
        'truck': (255, 0, 0, 140),
        'motorcycle': (255, 0, 0, 140),
        'chair': (255, 140, 0, 140),           # Orange commercial obstruction
        'dining table': (255, 140, 0, 140),
        'umbrella': (255, 140, 0, 140),
        'potted plant': (0, 255, 0, 140),      # Green urban furniture
        'bench': (0, 255, 0, 140)
    }

    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    total_images = len(image_files)

    print(f"Executing standard semantic overlay extraction on {total_images} node snapshots...\n")

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMG_DIR, img_name)
        out_path = os.path.join(OUTPUT_DIR, f"masked_{img_name}")

        if os.path.exists(out_path):
            continue

        try:
            img = Image.open(img_path).convert("RGBA")
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            predictions = segmenter(img.convert("RGB"))

            for pred in predictions:
                label = pred['label']
                
                # Topological overlap detection
                if label in COLORS:
                    mask = pred['mask'] 
                    solid_color = Image.new('RGBA', img.size, COLORS[label])
                    overlay.paste(solid_color, (0, 0), mask)

            final_img = Image.alpha_composite(img, overlay).convert("RGB")
            final_img.save(out_path)

            print(f"[{idx + 1}/{total_images}] Baseline segmentation isolated: {img_name}")

        except Exception as e:
            print(f"Iteration fault ({img_name}): {e}")

    print(f"\nPhase 1 Panoptic processing successfully committed to:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    generate_pixel_masks()
