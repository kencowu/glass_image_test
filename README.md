# Phone Image Processing Pipeline

This project provides a pipeline for processing phone images, including preprocessing and background removal. It uses OpenCV for image processing and is designed to handle multiple phones in a single image.

## Features

- Image preprocessing with adaptive thresholding
- Phone detection using contour analysis
- Background removal with transparency
- Support for multiple phones in a single image
- Maintains aspect ratio during processing

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- imutils

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your source images in the `source_image` directory.

2. Run the preprocessing script:
```bash
python image_preprocessor.py
```

3. Run the background removal script:
```bash
python background_removal_opencv.py
```

The processed images will be saved in the `processed_phones` directory.

## Project Structure

- `image_preprocessor.py`: Handles image preprocessing
- `background_removal_opencv.py`: Removes backgrounds from detected phones
- `source_image/`: Directory for input images
- `preprocessed_images/`: Directory for preprocessed images
- `processed_phones/`: Directory for final output images

## License

MIT License 