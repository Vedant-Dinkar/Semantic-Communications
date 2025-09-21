
import base64
import requests
import os

DATASET_DIR = "./semacomm-master/dataset"
OUTPUT_DIR = "./semacomm-master/noised_dataset"
NOISE_LEVEL = 30.0
IMAGE_FORMAT = "png"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith(('.jpg')):
        imagepath = os.path.join(DATASET_DIR, filename)
        with open(imagepath, "rb") as img_file:
            imagebase64 = base64.b64encode(img_file.read()).decode('utf-8')

        body = {
            "image": imagebase64,
            "noise_level": NOISE_LEVEL,
            "image_format": IMAGE_FORMAT
        }

        response = requests.post("http://localhost:4040/process_image", json=body)

        if response.status_code == 500:
            print(f"Skipping {filename} (server error 500).")
        else:
            response.raise_for_status()

            response_json = response.json()
            reconstructed_base64 = response_json["reconstructed_image"]

            outputimagepth = os.path.join(OUTPUT_DIR, filename)
            with open(outputimagepth, "wb") as out_file:
                out_file.write(base64.b64decode(reconstructed_base64))


