import cv2
import os
from pathlib import Path

def convert_webp_to_jpg(input_dir: str = "source_image"):
    """
    Convert all .webp files in the input directory to .jpg format.
    
    Args:
        input_dir: Directory containing the .webp files
    """
    # Create output directory if it doesn't exist
    output_dir = "source_image_jpg"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .webp files
    webp_files = list(Path(input_dir).glob("*.webp"))
    
    print(f"Found {len(webp_files)} .webp files to convert")
    
    for webp_path in webp_files:
        # Read the webp image
        img = cv2.imread(str(webp_path))
        if img is None:
            print(f"Failed to read {webp_path}")
            continue
            
        # Create output filename
        output_filename = webp_path.stem + ".jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save as jpg
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"Converted {webp_path.name} to {output_filename}")
        else:
            print(f"Failed to convert {webp_path.name}")

if __name__ == "__main__":
    convert_webp_to_jpg() 