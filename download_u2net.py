import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # U2Net model URL
    model_url = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net/u2net.pth"
    
    # Download the model
    print("Downloading U2Net model weights...")
    download_file(model_url, "u2net.pth")
    print("Download complete!")

if __name__ == "__main__":
    main() 