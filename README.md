# Phone Image Processor

A Python tool for processing phone images with background removal using the `rembg` library.

## Features

- Automatic phone detection and extraction from images
- Background removal using the `rembg` library
- Image rotation and orientation correction
- Size standardization with aspect ratio preservation
- Transparent background output

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pillow
- imutils
- rembg

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phone-image-processor.git
cd phone-image-processor
```

2. Create and activate a conda environment:
```bash
conda create -n phone_processor python=3.10
conda activate phone_processor
```

3. Install the required packages:
```bash
pip install opencv-python numpy pillow imutils rembg
```

## Usage

1. Place your input images in the `source_image_jpg` directory.

2. Run the script:
```bash
python phone_image_processor_rembg.py
```

3. Processed images will be saved in the `processed_phones` directory.

## How it Works

1. The script scans the `source_image_jpg` directory for JPG/JPEG files.
2. For each image:
   - Detects phones using contour analysis
   - Removes background using `rembg`
   - Corrects orientation if needed
   - Standardizes size while maintaining aspect ratio
   - Saves the processed image with a transparent background

## Output

- Processed images are saved as PNG files with transparent backgrounds
- Each phone is saved as a separate file in the `processed_phones` directory
- The script maintains the original aspect ratio while standardizing the size

## License

MIT License

## Acknowledgments

- [rembg](https://github.com/danielgatis/rembg) for background removal
- OpenCV for image processing 