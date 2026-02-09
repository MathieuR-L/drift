
import os
import argparse
from tqdm import tqdm
import imageio
import numpy as np
from PIL import Image

def make_video(image_dir, output_path, fps=10):
    print(f"Collecting images from {image_dir}...")
    files = [f for f in os.listdir(image_dir) if f.startswith("step_") and f.endswith(".png")]
    
    # Sort by step number: step_100.png -> 100
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not files:
        print("No images found!")
        return

    print(f"Found {len(files)} images. Processing...")
    
    # Use imageio writer
    writer = imageio.get_writer(output_path, fps=fps)
    
    for f in tqdm(files):
        path = os.path.join(image_dir, f)
        img = imageio.imread(path)
        writer.append_data(img)
        
    writer.close()
    print(f"Done! Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="results", help="Directory with step_*.png files")
    parser.add_argument("--output_path", type=str, default="training.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    args = parser.parse_args()
    
    make_video(args.image_dir, args.output_path, args.fps)
