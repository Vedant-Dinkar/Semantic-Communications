import base64
import requests

# 1. Read image from disk and convert to base64 string
imagepath = "/home/network/Documents/Semantic Communications/test_image (5).png"
with open(imagepath, "rb") as img_file:
    imagebase64 = base64.b64encode(img_file.read()).decode('utf-8')

# 2. Prepare the POST request body
body = {
    "image": imagebase64,
    "noise_level": 30.0,  # example noise level, adjust as needed
    "image_format": "png"
}

for i in range(1):
    # 3. Send POST request to the local FlowMo API
    response = requests.post("http://localhost:4040/process_image", json=body)
    response.raise_for_status()  # raise exception if request failed
    response_json = response.json()

    # 4. Extract reconstructed image base64 string from response
    reconstructed_base64 = response_json["reconstructed_image"]

    # 5. Decode and write the output image to disk
    outputimagepth = f"/home/network/Documents/Semantic Communications/output_image (Chappan Output - {i}_NOISE).png"
    with open(outputimagepth, "wb") as out_file:
        out_file.write(base64.b64decode(reconstructed_base64))

