from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple
import os

class YOLOPhoneDetector:
    def __init__(self, target_size: Tuple[int, int] = (300, 600)):
        """
        Initialize the YOLO phone detector.
        
        Args:
            target_size: Tuple of (width, height) for the output images
        """
        self.model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model
        self.target_size = target_size
        self.output_dir = "yolo_processed_phones"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def detect_phones(self, image_path: str) -> List[np.ndarray]:
        """
        Detect and extract individual phone images using YOLOv8.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of cropped and processed phone images
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Run YOLOv8 inference
        results = self.model(image)
        
        # Process detections
        phone_images = []
        for i, detection in enumerate(results[0].boxes.data):
            # Get box coordinates
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Check if detected object is a cell phone (class 67 in COCO dataset)
            if cls == 67:  # Cell phone class
                # Add padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Crop phone image
                phone_img = image[y1:y2, x1:x2]
                
                # Process and resize
                processed_phone = self._process_phone_image(phone_img)
                phone_images.append(processed_phone)
                
                # Save the processed phone image
                output_path = os.path.join(self.output_dir, f"phone_{i+1}.jpg")
                cv2.imwrite(output_path, processed_phone)
                print(f"Saved phone {i+1} to: {output_path}")
                print(f"Confidence: {conf:.2f}")
        
        return phone_images
    
    def _process_phone_image(self, phone_img: np.ndarray) -> np.ndarray:
        """
        Process a single phone image to standardize its size.
        
        Args:
            phone_img: Cropped phone image
            
        Returns:
            Processed phone image
        """
        # Get current dimensions
        h, w = phone_img.shape[:2]
        aspect = w / float(h)
        
        # Calculate new dimensions
        new_h = self.target_size[1]
        new_w = int(new_h * aspect)
        
        if new_w > self.target_size[0]:
            new_w = self.target_size[0]
            new_h = int(new_w / aspect)
        
        # Resize the image
        resized = cv2.resize(phone_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black background of target size
        final = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Center the image
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        
        # Paste the resized image
        final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return final

def detect_phones_yolo(image_path: str) -> List[str]:
    """
    Process an image containing multiple phones using YOLOv8.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of paths to the saved phone images
    """
    detector = YOLOPhoneDetector()
    phone_images = detector.detect_phones(image_path)
    
    return [os.path.join(detector.output_dir, f"phone_{i+1}.jpg") 
            for i in range(len(phone_images))]

if __name__ == "__main__":
    # Example usage
    image_path = "modern-touch-screen-smartphone-broken-600nw-2524629559.webp"
    try:
        saved_paths = detect_phones_yolo(image_path)
        print(f"\nProcessed {len(saved_paths)} phones:")
        for path in saved_paths:
            print(f"- {path}")
    except Exception as e:
        print(f"Error processing image: {e}") 