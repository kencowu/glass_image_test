import cv2
import numpy as np
from typing import List
import os
from rembg import remove, new_session
from PIL import Image

class BackgroundRemover:
    def __init__(self):
        """
        Initialize the background remover.
        """
        self.output_dir = "background_removal_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")
        self.session = new_session()

    def _create_phone_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask for the phone using rembg.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where phone is white (255) and background is black (0)
        """
        try:
            # Convert BGR to RGB for rembg
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Remove background
            output = remove(pil_image, session=self.session)
            
            # Convert back to numpy array
            output_np = np.array(output)
            
            # Convert to binary mask
            mask = cv2.cvtColor(output_np, cv2.COLOR_RGBA2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            print(f"Warning: Background removal failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def process_image(self, image_path: str) -> None:
        """
        Process a single image to remove the background.
        
        Args:
            image_path: Path to the input image
        """
        print(f"Reading image from: {image_path}")
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        print(f"Image read successfully. Shape: {image.shape}")
        
        # Create mask
        mask = self._create_phone_mask(image)
        
        # Save the mask
        output_path = os.path.join(self.output_dir, f"mask_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, mask)
        print(f"Mask saved to: {output_path}")

def main():
    # Initialize remover
    remover = BackgroundRemover()
    
    # Set input directory
    input_dir = "detection_output"
    image_file = "preprocessed_image2_phone_1.jpg"
    
    print(f"\nProcessing image: {image_file}")
    
    # Process the image
    try:
        image_path = os.path.join(input_dir, image_file)
        remover.process_image(image_path)
        print(f"Successfully processed {image_file}")
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")
    
    print(f"\nProcessing complete!")

if __name__ == "__main__":
    main() 