import os
import pandas as pd
import matplotlib.pyplot as plt
import folium
from transformers import pipeline
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_INPUT = os.path.join(BASE_DIR, "data", "processed", "sam3_results.csv") # Using SAM3 metrics
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "supplementary")

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_bottleneck_map(df):
    """Generates an HTML interactive topological map of spatial constraints."""
    print("Rendering Geospatial Encroachment Topologies...")
    
    if len(df) == 0:
        return

    center_lat = df['Lat'].mean() if 'Lat' in df.columns else 38.4237
    center_lon = df['Lon'].mean() if 'Lon' in df.columns else 27.1428

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles='CartoDB positron')

    for index, row in df.iterrows():
        # Fallbacks for missing coordinates
        lat = row.get('Lat', center_lat)
        lon = row.get('Lon', center_lon)
        ecw = float(row.get('ECW_SAM3_m', 0.0))

        if ecw == 0.0:
            color, radius, fill_opacity = 'red', 8, 0.9
        elif ecw < 1.5:
            color, radius, fill_opacity = 'orange', 6, 0.7
        else:
            color, radius, fill_opacity = 'green', 4, 0.5

        popup_text = f"""
        <div style="font-family: Arial; font-size: 12px; width: 200px;">
            <h4 style="margin-bottom: 5px;">Axis {row.get('Axis_ID')} - {row.get('Distance_m')}m</h4>
            <b>Morphology:</b> {row.get('Morphology')}<br>
            <b style="color:{color};">Functional ECW: {ecw} m</b>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon], radius=radius, color=color,
            fill=True, fill_color=color, fill_opacity=fill_opacity,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"ECW: {ecw}m"
        ).add_to(m)

    map_path = os.path.join(OUTPUT_DIR, "interactive_bottleneck_map.html")
    m.save(map_path)
    print(f"Topological node map exported: {map_path}")

def diagnose_bottlenecks(df):
    """Identifies root cause categorization for phantom capacity nodes utilizing AI."""
    print("Diagnosing structural causes for absolute bottlenecks (ECW < 1.5m)...")
    
    bottlenecks = df[df['ECW_SAM3_m'] < 1.5]
    print(f"Critical clearance failure identified at {len(bottlenecks)} cross-sections.")

    try:
        object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    except Exception as e:
        print(f"Model allocation warning: {e}")
        return

    encroachment_types = {
        "Cafe & Commercial": 0,
        "Vehicle Parking": 0,
        "Urban Furniture": 0,
    }

    def categorize_obstacle(label):
        if label in ['chair', 'dining table', 'umbrella']: return "Cafe & Commercial"
        elif label in ['car', 'truck', 'motorcycle', 'bus']: return "Vehicle Parking"
        elif label in ['potted plant', 'bench', 'fire hydrant', 'stop sign']: return "Urban Furniture"
        return "Other"

    # Deep inference on critical segments
    for index, row in bottlenecks.iterrows():
        axis_id = row.get('Axis_ID')
        dist = row.get('Distance_m')

        img_right = os.path.join(IMG_DIR, f"Axis{axis_id}_Dist{dist}_Right.jpg")
        img_left = os.path.join(IMG_DIR, f"Axis{axis_id}_Dist{dist}_Left.jpg")

        for img_path in [img_right, img_left]:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                predictions = object_detector(img)

                for obj in predictions:
                    if obj['score'] > 0.70:
                        cat = categorize_obstacle(obj['label'])
                        if cat != "Other":
                            encroachment_types[cat] += 1

    # Visualizing inference outcomes
    labels = [k for k, v in encroachment_types.items() if v > 0]
    sizes = [v for k, v in encroachment_types.items() if v > 0]

    if sizes:
        plt.figure(figsize=(8, 8))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
                wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        plt.title("Root Causes of Phantom Capacity Encroachment", fontsize=14, fontweight='bold')
        
        plot_path = os.path.join(OUTPUT_DIR, "Phantom_Capacity_Causation_Pie.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Structural causation inference chart compiled: {plot_path}")

if __name__ == "__main__":
    if os.path.exists(CSV_INPUT):
        data_matrix = pd.read_csv(CSV_INPUT)
        create_bottleneck_map(data_matrix)
        diagnose_bottlenecks(data_matrix)
    else:
        print(f"Warning: Primary dataset {CSV_INPUT} missing. Execute extraction protocols prior to deep diagnosis.")
