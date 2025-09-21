import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directory to save images
output_dir = './dataset'
os.makedirs(output_dir, exist_ok=True)

TARGET_IMAGES = 5000
BATCH_LIMIT = 100  # max supported by API
MAX_WORKERS = 50

def get_image_list(page, limit=BATCH_LIMIT):
    """Fetch list of images metadata from Picsum."""
    url = f"https://picsum.photos/v2/list?page={page}&limit={limit}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_and_save_image(item, index):
    """Download one image and save with proper filename."""
    try:
        img_id = item["id"]
        width = item["width"]
        height = item["height"]
        download_url = item["download_url"]  # full-resolution
        # Request fixed size (keeps it small, e.g. 256x256)
        url = f"https://picsum.photos/id/{img_id}/256/256.jpg"

        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            filename = f"{256}x{256}_{img_id}_{index}.jpg"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Error saving image {item['id']} at index {index}: {e}")
    return False

def main():
    downloaded = 0
    page = 1
    index = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while downloaded < TARGET_IMAGES:
            print(f"Fetching metadata page {page}...")
            items = get_image_list(page)

            futures = [executor.submit(fetch_and_save_image, item, index + i)
                       for i, item in enumerate(items)]

            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                    print(f"✅ Saved {downloaded}/{TARGET_IMAGES}")
                    if downloaded >= TARGET_IMAGES:
                        break

            index += len(items)
            page += 1

    print(f"\n🎉 Done! {downloaded} images saved in {output_dir}")

if __name__ == "__main__":
    main()
