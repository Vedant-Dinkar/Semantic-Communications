import requests
import base64
import csv

# Path to the image file
IMAGE_PATH = "test_image.png"

# Output CSV file
OUTPUT_PATH = "test_base64.txt"

def main():
    try:
        # Encode the image and get the latent vector
        with open(IMAGE_PATH, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        with open(OUTPUT_PATH, 'w+') as f:
            f.write(image_base64)
        print(f"Encoded image saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()