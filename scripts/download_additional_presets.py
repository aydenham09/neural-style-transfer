"""Download additional landscape presets to reach at least 8 presets total.

Downloads:
1. landscape_beach_waves.jpg (Pantai berombak)
2. landscape_rocky_sea.jpg (Laut berbatu)
3. landscape_rice_terrace.jpg (Sawah atau padang rumput)
4. landscape_sunset.jpg (Senja / Sunset)
5. landscape_interior_forest.jpg (Hutan pedalaman berkabut)

All from Wikimedia Commons with CC0 / Public Domain / open licenses.
"""

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

headers = {
    "User-Agent": "NeuralStyleTransferResearch/1.0 (contact: muham@example.org) Python-urllib/3.11"
}

ADDITIONAL_PRESETS = [
    {
        "filename": "landscape_beach_waves.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/91/Cloudy_beach_with_wrack_and_crashing_waves.jpg",
        "title": "Cloudy Beach with Crashing Waves",
        "creator": "James Wheeler",
        "year": "2016",
        "license": "Creative Commons CC0 1.0 Universal Public Domain Dedication",
        "source_url": "https://commons.wikimedia.org/wiki/File:Cloudy_beach_with_wrack_and_crashing_waves.jpg",
        "description": "Pantai berombak dengan deburan ombak berbusa putih memecah di bibir pantai pasir basah",
    },
    {
        "filename": "landscape_rocky_sea.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Rocky_coast_%28Unsplash%29.jpg",
        "title": "Rocky Coast at Queen's Bath",
        "creator": "Austin Schmid",
        "year": "2016",
        "license": "Creative Commons CC0 1.0 Universal Public Domain Dedication",
        "source_url": "https://commons.wikimedia.org/wiki/File:Rocky_coast_(Unsplash).jpg",
        "description": "Laut berbatu dengan tebing karang vulkanik gelap dan semburan ombak laut biru",
    },
    {
        "filename": "landscape_rice_terrace.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Rice_terraces_in_Indonesia-IMG_3082.jpg",
        "title": "Rice Terraces in Indonesia",
        "creator": "chensiyuan",
        "year": "2011",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
        "source_url": "https://commons.wikimedia.org/wiki/File:Rice_terraces_in_Indonesia-IMG_3082.jpg",
        "description": "Hamparan persawahan berundak hijau subur dan padang berumput tropis di lereng bukit",
    },
    {
        "filename": "landscape_sunset.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/df/Landscape_sunset.jpg",
        "title": "Golden Landscape Sunset",
        "creator": "Tomas Sobek",
        "year": "2010",
        "license": "Creative Commons CC0 1.0 Universal Public Domain Dedication",
        "source_url": "https://commons.wikimedia.org/wiki/File:Landscape_sunset.jpg",
        "description": "Pemandangan senja dengan semburat langit jingga keemasan dan siluet bukit bergelombang",
    },
    {
        "filename": "landscape_interior_forest.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Cloudy_forest_%28Unsplash%29.jpg",
        "title": "Misty Evergreen Interior Forest",
        "creator": "Kalen Emsley",
        "year": "2016",
        "license": "Creative Commons CC0 1.0 Universal Public Domain Dedication",
        "source_url": "https://commons.wikimedia.org/wiki/File:Cloudy_forest_(Unsplash).jpg",
        "description": "Pedalaman hutan cemara berkabut tebal dengan kanopi pohon rapat dan lumut basah",
    },
]


def download_presets():
    out_dir = Path("data/content")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading Additional Landscape Presets (Reaching >= 8 Presets)...")
    print("=" * 70)

    for item in ADDITIONAL_PRESETS:
        dest = out_dir / item["filename"]
        if dest.is_file() and dest.stat().st_size > 1000:
            print(f"[Exists] {item['filename']} already downloaded ({dest.stat().st_size / 1024:.1f} KB)")
            continue

        print(f"[Downloading] {item['title']} -> {dest} ...")
        time.sleep(2)
        req = urllib.request.Request(item["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        with open(dest, "wb") as f:
            f.write(data)

        # Verification with PIL
        with Image.open(dest) as img:
            img.verify()
            w, h = img.size

        # Reopen to get mode
        with Image.open(dest) as img:
            mode = img.mode

        sha256 = hashlib.sha256(data).hexdigest()
        print(f"  -> OK! Size: {len(data) / 1024:.1f} KB | Resolution: {w}x{h} | Mode: {mode} | SHA-256: {sha256[:16]}...")

    print("=" * 70)
    print("All additional landscape presets downloaded and verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    download_presets()

