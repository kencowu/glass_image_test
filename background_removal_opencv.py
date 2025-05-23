import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import imutils

class BackgroundRemover:
    def __init__(self, output_dir: str = "processed_phones"):
        """
        Initialize the background remover.
        
        Args:
            output_dir: Directory to save processed images
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")
        
        # Parameters for phone detection
        self.min_phone_area = 10000  # Reduced minimum area
        self.max_phone_area = 1000000  # Increased maximum area
        self.min_aspect_ratio = 0.3  # Reduced minimum aspect ratio
        self.max_aspect_ratio = 0.8  # Increased maximum aspect ratio

    def detect_phones(self, image: np.ndarray) -> List[Dict]:
        """
        Detect phones in the image using contour detection.
        
        Args:
            image: Input image
            
        Returns:
            List of dictionaries containing phone information
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect components
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phones = []
        for contour in contours:
            # Calculate area and aspect ratio
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Check if contour matches phone characteristics
            if (self.min_phone_area < area < self.max_phone_area and 
                self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                phones.append({
                    'contour': contour,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
                print(f"Found phone: area={area:.2f}, aspect_ratio={aspect_ratio:.2f}, w={w}, h={h}")
        
        return phones

    def remove_background(self, image: np.ndarray, phone_info: Dict) -> np.ndarray:
        """
        Remove background from a phone image.
        
        Args:
            image: Input image
            phone_info: Dictionary containing phone information
            
        Returns:
            Image with transparent background
        """
        # Create a mask for the phone
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [phone_info['contour']], -1, 255, -1)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Create RGBA image
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on mask
        rgba[:, :, 3] = mask
        
        # Crop the image to the phone region with padding
        x, y, w, h = phone_info['x'], phone_info['y'], phone_info['w'], phone_info['h']
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        cropped = rgba[y1:y2, x1:x2]
        
        return cropped

    def process_image(self, image_path: str) -> List[str]:
        """
        Process a single image to remove backgrounds from phones.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of paths to the processed images
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return []
        
        # Detect phones
        phones = self.detect_phones(image)
        if not phones:
            print(f"No phones detected in {image_path}")
            return []
        
        processed_paths = []
        for i, phone in enumerate(phones, 1):
            print(f"\nProcessing phone {i}:")
            
            # Remove background
            processed = self.remove_background(image, phone)
            
            # Save the processed image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            # Remove 'preprocessed_' prefix if it exists
            if base_name.startswith('preprocessed_'):
                base_name = base_name[13:]
            output_path = os.path.join(self.output_dir, f"{base_name}_phone_{i}.png")
            cv2.imwrite(output_path, processed)
            processed_paths.append(output_path)
            
            print(f"Saved processed phone {i} to: {output_path}")
        
        return processed_paths

    def process_directory(self, input_dir: str = "preprocessed_images") -> List[str]:
        """
        Process all images in the input directory.
        
        Args:
            input_dir: Directory containing input images
            
        Returns:
            List of paths to all processed images
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")
        
        all_processed_paths = []
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        print(f"\nFound {len(image_files)} images to process:")
        for img in image_files:
            print(f"- {img}")
        
        for image_file in image_files:
            try:
                image_path = os.path.join(input_dir, image_file)
                print(f"\nProcessing {image_file}...")
                processed_paths = self.process_image(image_path)
                all_processed_paths.extend(processed_paths)
                print(f"Successfully processed {len(processed_paths)} phones from {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        return all_processed_paths

def main():
    # Initialize background remover
    remover = BackgroundRemover()
    
    # Process all images in preprocessed_images directory
    try:
        print("\nStarting background removal...")
        processed_paths = remover.process_directory()
        print(f"\nProcessing complete!")
        print(f"Successfully processed {len(processed_paths)} phones from all images")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 