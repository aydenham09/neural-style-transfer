"""Download dataset images for Neural Style Transfer.
Downloads 3 content landscape photos and 3 style paintings from official sources.
"""

import os
import urllib.request
from PIL import Image

DATA_ITEMS = [
    {
        "category": "style",
        "key": "van_gogh",
        "title": "Wheat Field with Cypresses",
        "artist": "Vincent van Gogh",
        "year": 1889,
        "source_page": "https://www.metmuseum.org/art/collection/search/436535",
        "download_url": "https://images.metmuseum.org/CRDImages/ep/original/DP-42549-001.jpg",
        "license": "Met Open Access (CC0 / Public Domain)",
        "destination": "data/style/style_van_gogh.jpg",
        "filename": "style_van_gogh.jpg",
    },
    {
        "category": "style",
        "key": "hokusai",
        "title": "The Great Wave off Kanagawa (Under the Wave off Kanagawa)",
        "artist": "Katsushika Hokusai",
        "year": "ca. 1830-1832",
        "source_page": "https://commons.wikimedia.org/wiki/File:Tsunami_by_hokusai_19th_century.jpg",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/a/a5/Tsunami_by_hokusai_19th_century.jpg",
        "license": "Public Domain",
        "destination": "data/style/style_hokusai.jpg",
        "filename": "style_hokusai.jpg",
    },
    {
        "category": "style",
        "key": "kandinsky",
        "title": "Composition 8 (Komposition 8)",
        "artist": "Wassily Kandinsky",
        "year": 1923,
        "source_page": "https://commons.wikimedia.org/wiki/File:Kandinsky_-_Composition_8,_July_1923.jpg",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Kandinsky_-_Composition_8%2C_July_1923.jpg",
        "license": "Public Domain",
        "destination": "data/style/style_kandinsky.jpg",
        "filename": "style_kandinsky.jpg",
    },
    {
        "category": "content",
        "key": "mountain",
        "title": "View of Mount Rainier",
        "artist": "WqemenW (Contributor)",
        "year": 2023,
        "source_page": "https://commons.wikimedia.org/wiki/File:View_of_Mount_Rainier.jpg",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/c/ce/View_of_Mount_Rainier.jpg",
        "license": "Creative Commons CC0 1.0 Universal Public Domain Dedication",
        "destination": "data/content/landscape_mountain.jpg",
        "filename": "landscape_mountain.jpg",
    },
    {
        "category": "content",
        "key": "lake",
        "title": "Crater Lake from Rim",
        "artist": "United States Geological Survey (USGS)",
        "year": 1982,
        "source_page": "https://commons.wikimedia.org/wiki/File:Crater_Lake_from_rim-USGS.jpg",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/e/e1/Crater_Lake_from_rim-USGS.jpg",
        "license": "Public Domain (US Federal Government work)",
        "destination": "data/content/landscape_lake.jpg",
        "filename": "landscape_lake.jpg",
    },
    {
        "category": "content",
        "key": "forest",
        "title": "Coastal Redwood Forest (REDW9339)",
        "artist": "National Park Service (NPS) Digital Image Archives",
        "year": "Unknown (NPS Archive)",
        "source_page": "https://commons.wikimedia.org/wiki/File:Redwood_National_Park_REDW9339.jpg",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Redwood_National_Park_REDW9339.jpg",
        "license": "Public Domain (US Federal Government work)",
        "destination": "data/content/landscape_forest.jpg",
        "filename": "landscape_forest.jpg",
    },
]


def download_all():
    os.makedirs("data/content", exist_ok=True)
    os.makedirs("data/style", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    results = []
    for item in DATA_ITEMS:
        dest = item["destination"]
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"File already exists: {dest}")
        else:
            print(f"Downloading {item['title']} -> {dest} ...")
            req = urllib.request.Request(item["download_url"], headers=headers)
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                f.write(resp.read())

        with Image.open(dest) as img:
            w, h = img.size
            mode = img.mode
            fmt = img.format

        file_size_kb = round(os.path.getsize(dest) / 1024, 2)
        print(f"  Verified {dest}: {w}x{h} px, format={fmt}, mode={mode}, size={file_size_kb} KB")

        item_result = {
            "title": item["title"],
            "creator": item["artist"],
            "year": item["year"],
            "category": item["category"],
            "filename": item["filename"],
            "path": item["destination"],
            "source_url": item["source_page"],
            "download_url": item["download_url"],
            "license": item["license"],
            "dimensions": {"width": w, "height": h},
            "channels": len(img.getbands()),
            "format": fmt,
            "size_kb": file_size_kb,
        }
        results.append(item_result)

    import yaml
    metadata_path = "data/metadata/dataset_metadata.yaml"
    metadata_content = {
        "content_images": [r for r in results if r["category"] == "content"],
        "style_images": [r for r in results if r["category"] == "style"]
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata_content, f, sort_keys=False, allow_unicode=True)
    print(f"Saved dataset metadata to {metadata_path}")
    return results


if __name__ == "__main__":
    download_all()
