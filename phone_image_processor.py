import cv2
import numpy as np
import os
from typing import List, Tuple
import imutils

class PhoneImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (300, 600)):
        """
        Initialize the phone image processor.
        
        Args:
            target_size: Tuple of (width, height) for the output images
        """
        self.target_size = target_size
        self.output_dir = "processed_phones"
        self.reference_angle = None  # Store the angle of the first phone
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_phones(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect and extract individual phone images from the input image.
        
        Args:
            image: Input image containing multiple phones
            
        Returns:
            List of cropped and processed phone images
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 30, 150)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and merge nearby contours
        min_area = width * height * 0.05  # Minimum 5% of image area
        phone_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Print debug information
                print(f"Found contour: area={area:.2f}, aspect_ratio={aspect_ratio:.2f}, w={w}, h={h}")
                
                # Check if it's likely a phone
                if 0.3 < aspect_ratio < 0.8:
                    phone_contours.append(contour)
                    print(f"Accepted as phone: aspect_ratio={aspect_ratio:.2f}, area={area:.2f}")
        
        # Sort contours from left to right
        phone_contours = sorted(phone_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Extract and process phone images
        phone_images = []
        for i, contour in enumerate(phone_contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding (percentage based)
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            
            # Ensure we don't go out of bounds
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(width, x + w + padding_x)
            y2 = min(height, y + h + padding_y)
            
            # Extract phone image
            phone_img = image[y1:y2, x1:x2]
            
            print(f"\nProcessing phone {i+1}:")
            print(f"Crop dimensions: x={x1}-{x2}, y={y1}-{y2}")
            print(f"Original size: {phone_img.shape}")
            
            # Process the phone image
            processed_phone = self._process_phone_image(phone_img)
            phone_images.append(processed_phone)
            
            # Save the processed phone image
            output_path = os.path.join(self.output_dir, f"phone_{i+1}.png")
            cv2.imwrite(output_path, processed_phone)
            
            # Calculate percentage of background pixels
            total_pixels = processed_phone.shape[0] * processed_phone.shape[1]
            background_pixels = np.sum(processed_phone[:, :, 3] == 0)  # Count transparent pixels
            background_percentage = (background_pixels / total_pixels) * 100
            print(f"Background percentage: {background_percentage:.2f}%")
            print(f"Saved to: {output_path}")
            print(f"Processed size: {processed_phone.shape}")
        
        return phone_images

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

    def _create_phone_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask for the phone using color and edge information.
        
        Args:
            image: Input image
            
        Returns:
            Binary mask where phone is white (255) and background is black (0)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find the largest contour (should be the phone)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        return mask

    def _process_phone_image(self, phone_img: np.ndarray) -> np.ndarray:
        """
        Process a single phone image to standardize its size and orientation.
        
        Args:
            phone_img: Cropped phone image
            
        Returns:
            Processed phone image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(phone_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate the angle of the phone
        current_angle = self._calculate_phone_angle(binary)
        print(f"Detected phone angle: {current_angle:.2f} degrees")
        
        # Rotate the image to align with vertical orientation
        if abs(current_angle - 90) > 0.5:  # Only rotate if not close to vertical
            rotation_angle = 90 - current_angle
            print(f"Rotating image by {rotation_angle:.2f} degrees to make it vertical")
            rotated = imutils.rotate_bound(phone_img, rotation_angle)
            # Update binary image and gray after rotation
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            print("Image has been rotated to vertical orientation")
        else:
            print("Image is already vertical (within 0.5 degrees), no rotation needed")
            rotated = phone_img
        
        # Create a mask for the phone
        mask = self._create_phone_mask(rotated)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (should be the phone)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding to ensure we capture the entire phone
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
        new_h = self.target_size[1]  # Set height to target height
        new_w = int(new_h * aspect)  # Calculate width to maintain aspect ratio
        
        if new_w > self.target_size[0]:
            new_w = self.target_size[0]  # Cap width at target width
            new_h = int(new_w / aspect)  # Recalculate height
        
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
        
        # Calculate position to paste the resized image (center it)
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        
        # Paste the resized image onto the transparent background
        final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_rgba
        
        return final

def process_phone_image(image_path: str, output_dir: str = "processed_phones") -> List[str]:
    """
    Process an image containing multiple phones and save individual phone images.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save processed phone images
        
    Returns:
        List of paths to the saved phone images
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Initialize processor
    processor = PhoneImageProcessor()
    
    # Process the image
    phone_images = processor.detect_phones(image)
    
    # Get the paths of saved images
    saved_paths = []
    for i in range(len(phone_images)):
        saved_paths.append(os.path.join(output_dir, f"phone_{i+1}.png"))
    
    return saved_paths

if __name__ == "__main__":
    # Example usage
    image_path = "modern-touch-screen-smartphone-broken-600nw-2524629559.webp"  # Using the webp image file
    try:
        saved_paths = process_phone_image(image_path)
        print(f"Processed {len(saved_paths)} phones:")
        for path in saved_paths:
            print(f"- {path}")
    except Exception as e:
        print(f"Error processing image: {e}") 