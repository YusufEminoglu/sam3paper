import os
import torch
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# IMPORTANT: Provide your Hugging Face Token here to utilize the SAM 3 model framework.
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: User should ensure ecw_results.csv is either populated initially or dynamically created.
CSV_INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "gsv_nodes.csv") 
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
CSV_OUTPUT = os.path.join(BASE_DIR, "data", "processed", "sam3_results.csv")

def generate_sam3_metrics_csv(hf_token=HF_TOKEN):
    print("Initiating SAM 3 Orthogonal Metrics Extractor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Executing core model allocations...")
    try:
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(device)
    except Exception as e:
        print("ERROR: Authorization failed. Please ensure a valid Hugging Face token is provided.")
        return

    # Attempt to load baseline topological definitions
    try:
        df_base = pd.read_csv(CSV_INPUT_PATH)
    except FileNotFoundError:
        print(f"Failed to load baseline topological matrix at {CSV_INPUT_PATH}")
        return

    # Semantic categories for functional obstruction measurements
    CATEGORIES = [
        'person', 'car', 'truck', 'motorcycle', 'chair', 'table',
        'potted plant', 'trash can', 'dumpster', 'utility pole',
        'street light', 'utility box', 'electrical cabinet'
    ]

    # Baseline computational pixel density projection logic (GSV constants)
    PIXELS_PER_SQ_METER = 10240.0

    results = []
    total_rows = len(df_base)

    print(f"\nConstructing physical dimensions mapping for {total_rows} nodal structures...\n")

    for index, row in df_base.iterrows():
        axis_id = row.get('Axis_ID', index)
        dist = row.get('distance', 0)
        width_cad = float(row.get('Width_Cad', 5.0)) # Fallback nominal width
        morphology = row.get('Morphology', 'Unknown')

        # Generate contextual matrices
        node_data = {
            'Axis_ID': axis_id,
            'Distance_m': dist,
            'Morphology': morphology,
            'Width_Cad_m': width_cad,
        }

        for cat in CATEGORIES:
            node_data[f"{cat}_Count"] = 0
            node_data[f"{cat}_Area_m2"] = 0.0

        clutter_left = 0.0
        clutter_right = 0.0

        for side in ['Left', 'Right']:
            img_name = f"Axis{axis_id}_Dist{dist}_{side}.jpg"
            img_path = os.path.join(IMG_DIR, img_name)

            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")

            for label_name in CATEGORIES:
                inputs = processor(images=img, text=label_name, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                res = processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]

                masks = res["masks"].cpu().numpy()
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()

                for i in range(len(masks)):
                    if scores[i] > 0.60:
                        # Tabular enumeration and measurement updates
                        node_data[f"{label_name}_Count"] += 1

                        mask = masks[i]
                        white_pixels = np.sum(mask)
                        area_m2 = white_pixels / PIXELS_PER_SQ_METER
                        node_data[f"{label_name}_Area_m2"] += round(area_m2, 2)

                        # Formulate obstruction boundaries for Phantom Capacity logic
                        if label_name != 'person':  # Transient flow entity exemption
                            box = boxes[i]
                            pixel_width = box[2] - box[0]
                            # Functional projection matrix coefficient
                            physical_width = (pixel_width * 5.0) / 600.0

                            if side == 'Left':
                                clutter_left += physical_width
                            else:
                                clutter_right += physical_width

        clutter_left = min(clutter_left, 4.0)
        clutter_right = min(clutter_right, 4.0)

        # Phantom ECW Deduction
        ecw = width_cad - (clutter_left + clutter_right)
        ecw = max(ecw, 0.0)

        node_data['Clutter_Left_m'] = round(clutter_left, 2)
        node_data['Clutter_Right_m'] = round(clutter_right, 2)
        node_data['ECW_SAM3_m'] = round(ecw, 2)

        total_block_area = sum(node_data[f"{c}_Area_m2"] for c in CATEGORIES if c != 'person')
        node_data['Total_Obstacle_Area_m2'] = round(total_block_area, 2)

        results.append(node_data)

    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(CSV_OUTPUT, index=False)
    print(f"\nArtifact finalized. ECW spatial matrices structured in: {CSV_OUTPUT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM 3 ECW Spatial Matrix Extractor.')
    parser.add_argument('--token', type=str, default=HF_TOKEN, help='HuggingFace Auth Token')
    args = parser.parse_args()
    generate_sam3_metrics_csv(args.token)
