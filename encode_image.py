import requests
import base64
import csv

# API endpoint
API_URL = "https://kbtxcr05-4050.inc1.devtunnels.ms/encode"

# Path to the image file
IMAGE_PATH = "test_image.png"

# Output CSV file
OUTPUT_CSV = "encoded_vector.csv"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
 
    payload = {"image_base64": image_base64}
    
    # Send the POST request
    response = requests.post(API_URL, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract the latent vector (base64-encoded NPY)
        latent_base64 = response.json().get("latent_base64")
        if latent_base64:
            return latent_base64
        else:
            raise ValueError("No latent_base64 found in the response.")
    else:
        raise Exception(f"Failed to encode image. Status code: {response.status_code}, Response: {response.text}")

def save_latent_to_csv(latent_base64, output_csv):
    # Decode the base64-encoded latent vector
    print(latent_base64)
    print(type(latent_base64))
    latent_bytes = base64.b64decode(latent_base64)
    
    # Convert the bytes to a list of floats (assuming NPY format)
    import numpy as np
    latent_vector = np.frombuffer(latent_bytes, dtype=np.float32)
    
    # Save the latent vector to a CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(latent_vector)

def main():
    try:
        # Encode the image and get the latent vector
        latent_base64 = encode_image(IMAGE_PATH)
        
        # Save the latent vector to a CSV file
        save_latent_to_csv(latent_base64, OUTPUT_CSV)
        
        print(f"Encoded vector saved to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()