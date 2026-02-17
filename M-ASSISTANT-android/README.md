# M-ASSISTANT — Build Guide

**Developer:** Moritz · onlymoritz.de  
**Version:** 1.0  
**Target:** Android tablets (API 24+, Android 7.0+)

---

## What this app does

- Opens fullscreen (truly immersive — no status/navigation bars)
- Displays a configurable web address with full permissions
- Camera, microphone, geolocation, storage — all auto-granted, no prompts
- All SSL/TLS errors are ignored — works with HTTP and self-signed certs
- Auto-restarts on crash
- Auto-launches after device reboot
- Bottom-right ⚙️ settings button: change URL + zoom level (0.1x – 10.0x)

---

## Prerequisites

You need **Android Studio** or just the **command-line build tools**.

### Option A — Android Studio (recommended, easiest)

1. Download Android Studio: https://developer.android.com/studio
2. Install it, open it, let it download the SDK
3. That's it — you're ready

### Option B — Command line only

1. Install JDK 17:
   - **Windows:** https://adoptium.net/
   - **macOS:** `brew install openjdk@17`
   - **Linux:** `sudo apt install openjdk-17-jdk`

2. Install Android SDK command-line tools:
   - Download from: https://developer.android.com/studio#command-tools
   - Extract to `~/android-sdk/cmdline-tools/latest/`
   - Accept licenses: `sdkmanager --licenses`
   - Install build tools: `sdkmanager "build-tools;34.0.0" "platforms;android-34"`

3. Set environment variables:
   ```
   export ANDROID_HOME=~/android-sdk
   export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
   export PATH=$PATH:$ANDROID_HOME/platform-tools
   ```

---

## Adding your logo (optional but recommended)

Your `logo.png` (1000×1000) goes here — run this script or do it manually:

```bash
# Place your logo.png in the project root, then run:
python3 replace_icons.py
```

**OR manually copy:**
```
logo.png  →  app/src/main/res/mipmap-mdpi/ic_launcher.png     (resize to 48×48)
logo.png  →  app/src/main/res/mipmap-hdpi/ic_launcher.png     (resize to 72×72)
logo.png  →  app/src/main/res/mipmap-xhdpi/ic_launcher.png    (resize to 96×96)
logo.png  →  app/src/main/res/mipmap-xxhdpi/ic_launcher.png   (resize to 144×144)
logo.png  →  app/src/main/res/mipmap-xxxhdpi/ic_launcher.png  (resize to 192×192)
```
Same for `ic_launcher_round.png` in each folder.

**Easiest way:** In Android Studio → right-click `res` → New → Image Asset → choose your logo.png → it generates all sizes automatically.

---

## Building the APK

### Via Android Studio

1. Open Android Studio
2. **File → Open** → select the `M-ASSISTANT` folder
3. Wait for Gradle sync (first time downloads ~1 GB)
4. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
5. APK is at: `app/build/outputs/apk/debug/app-debug.apk`

### Via command line

```bash
cd M-ASSISTANT

# On Windows:
gradlew.bat assembleDebug

# On Mac/Linux:
chmod +x gradlew
./gradlew assembleDebug
```

APK ends up at: `app/build/outputs/apk/debug/app-debug.apk`

---

## Installing on your tablet

### Via ADB (USB)

1. On your tablet: **Settings → About tablet → tap "Build number" 7 times**
2. **Settings → Developer options → USB debugging → ON**
3. Connect USB cable to your PC
4. Install ADB:
   - Windows: https://developer.android.com/tools/releases/platform-tools
   - macOS: `brew install android-platform-tools`
   - Linux: `sudo apt install adb`
5. Run:
   ```bash
   adb devices          # should show your tablet
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

### Via file transfer

1. Copy `app-debug.apk` to your tablet (USB file transfer / email / cloud)
2. On tablet: **Settings → Security → Unknown sources → ON** (or "Install unknown apps")
3. Open the APK file on tablet → Install

---

## First launch

1. Open **M-ASSISTANT** on your tablet
2. It will ask for permissions — **Allow All**
3. The app opens your configured URL fullscreen
4. Tap the small ⚙️ button (bottom-right) to change URL or zoom

---

## Changing URL and zoom

1. Tap ⚙️ bottom-right corner
2. **SERVER URL:** enter your address, e.g. `http://192.168.188.71:8765/`
3. **SCREEN ZOOM:** drag slider (0.1x to 10.0x, steps of 0.1)
4. Tap **SAVE & APPLY** → app reloads with new settings

---

## Updating version number

Before building, open `app/build.gradle`:
```gradle
versionCode 1        ← increment for each release (integer)
versionName "1.0"    ← change to "1.1", "2.0" etc.
```

And in `SettingsActivity.java`:
```java
private static final String VERSION = "1.0";  ← update this
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Gradle sync fails | Check internet, try **File → Invalidate Caches** |
| `ANDROID_HOME` not found | Set it as described in Prerequisites |
| Tablet not detected via ADB | Try different USB cable, enable USB debugging |
| App crashes on launch | Check logcat: `adb logcat -s M-ASSISTANT` |
| Black screen | Make sure URL is reachable from tablet's network |
| Camera/mic not working | Grant permissions in Settings → Apps → M-ASSISTANT → Permissions |

---

## File structure

```
M-ASSISTANT/
├── app/
│   ├── src/main/
│   │   ├── AndroidManifest.xml          ← permissions, activities
│   │   ├── java/de/onlymoritz/massistant/
│   │   │   ├── MainActivity.java        ← main WebView + crash handler
│   │   │   ├── SettingsActivity.java    ← URL + zoom settings
│   │   │   └── BootReceiver.java        ← auto-start after reboot
│   │   └── res/
│   │       ├── layout/
│   │       │   ├── activity_main.xml    ← WebView layout
│   │       │   └── activity_settings.xml
│   │       ├── values/
│   │       │   ├── strings.xml
│   │       │   ├── styles.xml
│   │       │   └── colors.xml
│   │       ├── xml/
│   │       │   └── network_security_config.xml  ← allow all HTTP
│   │       └── mipmap-*/
│   │           └── ic_launcher*.png     ← app icons
│   └── build.gradle
├── build.gradle
├── settings.gradle
├── gradle.properties
└── README.md  ← you are here
```

---

## Logo icon replacement script (replace_icons.py)

Save this as `replace_icons.py` in the project root and run `python3 replace_icons.py` with `logo.png` present:

```python
from PIL import Image
import shutil

logo = Image.open("logo.png").convert("RGBA")

sizes = {
    "app/src/main/res/mipmap-mdpi":    48,
    "app/src/main/res/mipmap-hdpi":    72,
    "app/src/main/res/mipmap-xhdpi":   96,
    "app/src/main/res/mipmap-xxhdpi":  144,
    "app/src/main/res/mipmap-xxxhdpi": 192,
}

for folder, size in sizes.items():
    resized = logo.resize((size, size), Image.LANCZOS)
    resized.save(f"{folder}/ic_launcher.png")
    resized.save(f"{folder}/ic_launcher_round.png")
    print(f"✓ {folder} → {size}x{size}")

print("Done! Rebuild the app.")
```

---

*M-ASSISTANT v1.0 · Developer: Moritz · onlymoritz.de*
