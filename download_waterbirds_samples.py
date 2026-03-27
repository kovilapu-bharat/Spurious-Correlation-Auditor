import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from wilds import get_dataset
import os

print("Downloading Stanford WILDS Waterbirds Dataset...")
print("This may take a few minutes depending on your internet connection (approx 1GB).")

# This will download the dataset to ./data/waterbirds_v1.0 by default
dataset = get_dataset(dataset="waterbirds", download=True)

print(f"✅ Success! Total images in dataset: {len(dataset)}")

output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

# Extract the first 5 images for easy testing in the UI
# In WILDS: y=1 means waterbird, y=0 means landbird
for i in range(5):
    img, y, metadata = dataset[i]
    label = "waterbird" if y == 1 else "landbird"
    img.save(f"{output_dir}/{label}_sample_{i}.jpg")

print(f"📁 Extracted 5 sample images to the '{output_dir}/' folder.")
print("You can upload these directly into the React UI to test for Spurious Background Correlations!")
