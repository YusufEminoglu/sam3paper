import pandas as pd
import requests
import os
import argparse

# --- CONFIGURATION ---
# Note: Provide your Google Street View API Key either directly here or via arguments.
API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
# Adjust directories relative to repository root
CSV_PATH = "../data/raw/gsv_nodes.csv"
OUTPUT_DIR = "../data/images/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def harvest_gsv_images(api_key=API_KEY):
    print("Reading spatial nodes CSV dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: {CSV_PATH} not found. Please ensure the CSV is placed in data/raw/")
        return

    total_nodes = len(df)
    print(f"Detected {total_nodes} nodes. Initiating data harvesting...\n")

    for index, row in df.iterrows():
        axis_id = row['Axis_ID']
        node_id = row['distance']
        lat = row['Lat']
        lon = row['Lon']
        angle = row['angle']  # Tangent angle indicating axis direction

        # Orthogonal correction to capture pedestrian cross-sections (facades)
        heading_right = (angle + 90) % 360
        heading_left = (angle - 90) % 360

        # Right Facade URL Payload
        url_right = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lon}&heading={heading_right}&pitch=0&fov=90&key={api_key}"
        filename_right = os.path.join(OUTPUT_DIR, f"Axis{axis_id}_Dist{node_id}_Right.jpg")

        # Left Facade URL Payload
        url_left = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lon}&heading={heading_left}&pitch=0&fov=90&key={api_key}"
        filename_left = os.path.join(OUTPUT_DIR, f"Axis{axis_id}_Dist{node_id}_Left.jpg")

        # Acquire Right Facade Imagery
        if not os.path.exists(filename_right):
            response_r = requests.get(url_right)
            if response_r.status_code == 200:
                with open(filename_right, 'wb') as file:
                    file.write(response_r.content)
            else:
                print(f"ERROR: Failed to acquire right facade for Axis {axis_id}. Status: {response_r.status_code}")

        # Acquire Left Facade Imagery
        if not os.path.exists(filename_left):
            response_l = requests.get(url_left)
            if response_l.status_code == 200:
                with open(filename_left, 'wb') as file:
                    file.write(response_l.content)
            else:
                print(f"ERROR: Failed to acquire left facade for Axis {axis_id}. Status: {response_l.status_code}")

        print(f"Success: Node {node_id}m at Axis {axis_id} | Orthogonal views captured.")

    print(f"\nData acquisition complete. Vistas saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated GSV Data Acquisition for Phantom Capacity Analysis.')
    parser.add_argument('--apikey', type=str, default=API_KEY, help='Google Street View API Key')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    harvest_gsv_images(args.apikey)
