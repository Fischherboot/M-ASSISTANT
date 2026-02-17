#!/usr/bin/env python3
"""
M-ASSISTANT Icon Replacer
--------------------------
Place your logo.png (1000x1000) in the same folder as this script,
then run:  python3 replace_icons.py
"""

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed.")
    print("Run: pip install Pillow")
    exit(1)

import os
import sys

LOGO_FILE = "logo.png"
SIZES = {
    "app/src/main/res/mipmap-mdpi":    48,
    "app/src/main/res/mipmap-hdpi":    72,
    "app/src/main/res/mipmap-xhdpi":   96,
    "app/src/main/res/mipmap-xxhdpi":  144,
    "app/src/main/res/mipmap-xxxhdpi": 192,
}

if not os.path.exists(LOGO_FILE):
    print(f"Error: {LOGO_FILE} not found in current directory.")
    print("Place your logo.png next to this script and try again.")
    sys.exit(1)

print(f"Loading {LOGO_FILE}...")
logo = Image.open(LOGO_FILE).convert("RGBA")
print(f"Source size: {logo.size[0]}x{logo.size[1]}")

for folder, size in SIZES.items():
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    resized = logo.resize((size, size), Image.LANCZOS)
    resized.save(f"{folder}/ic_launcher.png", "PNG")
    resized.save(f"{folder}/ic_launcher_round.png", "PNG")
    print(f"✓ {folder}/ic_launcher.png  ({size}x{size})")

print("\nAll icons replaced successfully!")
print("Now rebuild the APK in Android Studio or with: ./gradlew assembleDebug")
