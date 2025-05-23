from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple
import os
import sys
from rembg import remove, new_session
from PIL import Image
import imutils

class YOLOPhoneDetector:
    def __init__(self, target_size: Tuple[int, int] = (300, 600), conf_threshold: float = 0.1):
        """
        Initialize the YOLO phone detector.
        
        Args:
            target_size: Tuple of (width, height) for the output images
            conf_threshold: Confidence threshold for detections
        """
        print("Initializing YOLO model...")
        # Load the YOLOv8m model (medium size, better accuracy)
        self.model = YOLO('yolov8m.pt')
        # Set model parameters
        self.model.conf = conf_threshold  # Lower confidence threshold
        self.model.iou = 0.45  # IOU threshold for NMS
        self.model.agnostic = True  # Class-agnostic NMS
        self.model.multi_label = True  # Allow multiple labels per box
        print("YOLO model loaded successfully")
        self.target_size = target_size
        self.output_dir = "processed_phones"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")
        
        # Initialize rembg session
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

    def _calculate_phone_angle(self, binary_image: np.ndarray) -> float:
        """
        Calculate the angle of the phone using PCA on the binary image points.
        
        Args:
            binary_image: Binary image of the phone
            
        Returns:
            Angle in degrees
        """
        # Find points that belong to the phone (white pixels)
        y_coords, x_coords = np.nonzero(binary_image)
        coords = np.column_stack((x_coords, y_coords))
        
        # Calculate PCA
        mean = np.mean(coords, axis=0)
        coords_centered = coords - mean
        cov = np.cov(coords_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Get the angle from the principal component
        principal_vector = eigenvectors[:, -1]  # Vector with largest eigenvalue
        angle = np.degrees(np.arctan2(principal_vector[1], principal_vector[0]))
        
        # Normalize angle to be between 0 and 180 degrees
        angle = angle % 180
        
        # Additional check to ensure phone is not upside down
        # Calculate the center of mass of the phone
        center_y = np.mean(y_coords)
        height = binary_image.shape[0]
        
        # If the center of mass is in the bottom half, flip the phone
        if center_y > height / 2:
            angle = (angle + 180) % 180
            
        return angle

    def _process_phone_image(self, phone_img: np.ndarray) -> np.ndarray:
        """
        Process a single phone image to standardize its size and orientation.
        
        Args:
            phone_img: Cropped phone image
            
        Returns:
            Processed phone image
        """
        # Get the initial mask using rembg
        mask = self._create_phone_mask(phone_img)
        
        # Calculate the angle
        current_angle = self._calculate_phone_angle(mask)
        print(f"Detected phone angle: {current_angle:.2f} degrees")
        
        # Rotate if needed
        if abs(current_angle - 90) > 0.5:
            rotation_angle = 90 - current_angle
            print(f"Rotating image by {rotation_angle:.2f} degrees to make it vertical")
            rotated = imutils.rotate_bound(phone_img, rotation_angle)
            # Update mask after rotation
            mask = self._create_phone_mask(rotated)
            print("Image has been rotated to vertical orientation")
        else:
            print("Image is already vertical (within 0.5 degrees), no rotation needed")
            rotated = phone_img
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(rotated.shape[1] - x, w + 2 * padding)
            h = min(rotated.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = rotated[y:y+h, x:x+w]
            # Update mask for cropped region
            mask = mask[y:y+h, x:x+w]
        else:
            cropped = rotated
            mask = self._create_phone_mask(cropped)
        
        # Resize to target size while maintaining aspect ratio
        h, w = cropped.shape[:2]
        aspect = w / float(h)
        
        # Calculate new dimensions
        new_h = self.target_size[1]
        new_w = int(new_h * aspect)
        
        if new_w > self.target_size[0]:
            new_w = self.target_size[0]
            new_h = int(new_w / aspect)
        
        # Resize the image and mask
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create a transparent background
        final = np.zeros((self.target_size[1], self.target_size[0], 4), dtype=np.uint8)
        
        # Convert resized image to RGBA
        if resized.shape[2] == 3:
            resized_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
        else:
            resized_rgba = resized
        
        # Apply the mask to create transparency
        resized_rgba[:, :, 3] = resized_mask
        
        # Calculate position to paste the resized image
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        
        # Paste the resized image onto the transparent background
        final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_rgba
        
        return final

    def detect_phones(self, image_path: str) -> List[np.ndarray]:
        """
        Detect and extract individual phone images using YOLOv8.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of cropped and processed phone images
        """
        print(f"Reading image from: {image_path}")
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        print(f"Image read successfully. Shape: {image.shape}")
            
        # Run YOLOv8 inference with specific parameters
        print("Running YOLO inference...")
        results = self.model(image, verbose=True)
        print("YOLO inference completed")
        
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Process detections
        phone_images = []
        print(f"Number of detections: {len(results[0].boxes.data)}")
        
        # Print all detections for debugging
        print("\nAll detections:")
        for i, detection in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            print(f"Detection {i+1}: Class {cls}, Confidence {conf:.2f}")
            
            # Draw all detections on visualization image
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"Class {cls:.0f} ({conf:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if detected object is a cell phone (class 67 in COCO dataset)
            if cls == 67:  # Cell phone class
                print(f"Processing phone detection {i+1}")
                # Add padding
                padding = 20  # Increased padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Crop phone image using bounding box
                phone_img = image[y1:y2, x1:x2]
                
                # Process and resize with background removal
                processed_phone = self._process_phone_image(phone_img)
                phone_images.append(processed_phone)
                
                # Save the processed phone image
                output_path = os.path.join(self.output_dir, f"phone_{i+1}.png")
                cv2.imwrite(output_path, processed_phone)
                print(f"Saved phone {i+1} to: {output_path}")
                print(f"Confidence: {conf:.2f}")
        
        # Save visualization image with unique name
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(self.output_dir, f"{base_name}_detection_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"\nSaved detection visualization to: {vis_path}")
        
        # Print COCO class names for reference
        print("\nCOCO class names for reference:")
        coco_names = self.model.names
        for class_id, class_name in coco_names.items():
            print(f"Class {class_id}: {class_name}")
        
        return phone_images

def detect_phones_yolo(image_path: str) -> List[str]:
    """
    Process an image containing multiple phones using YOLOv8.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of paths to the saved phone images
    """
    print(f"Starting phone detection for image: {image_path}")
    detector = YOLOPhoneDetector(conf_threshold=0.1)  # Lower confidence threshold
    phone_images = detector.detect_phones(image_path)
    
    return [os.path.join(detector.output_dir, f"phone_{i+1}.png") 
            for i in range(len(phone_images))]

def main():
    # Set input directory
    input_dir = "preprocessed_images"
    
    # Get all image files from input directory
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    
    print(f"\nFound {len(image_files)} images to process:")
    for img in image_files:
        print(f"- {img}")
    
    # Process each image
    total_processed = 0
    for image_file in image_files:
        try:
            image_path = os.path.join(input_dir, image_file)
            print(f"\nProcessing {image_file}...")
            saved_paths = detect_phones_yolo(image_path)
            total_processed += len(saved_paths)
            print(f"Successfully processed {len(saved_paths)} phones from {image_file}")
            for path in saved_paths:
                print(f"  - Saved to: {path}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {total_processed} phones from {len(image_files)} images")

if __name__ == "__main__":
    main() 