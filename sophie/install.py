#!/usr/bin/env python3
"""Sophie v3.3 – Automatisches Installations-Script"""
import subprocess
import sys
import urllib.request
import os
from pathlib import Path

# Farben für die Ausgabe
G = "\033[92m"  # Grün
Y = "\033[93m"  # Gelb
R = "\033[91m"  # Rot
C = "\033[96m"  # Cyan
B = "\033[1m"   # Fett
X = "\033[0m"   # Reset

def ok(m): print(f"{G}  v{X} {m}")
def info(m): print(f"{C}  >{X} {m}")
def warn(m): print(f"{Y}  !{X} {m}")
def err(m): print(f"{R}  X{X} {m}")
def run(cmd): 
    # pip output unterdrücken, außer es gibt Fehler
    return subprocess.run(cmd, shell=True, check=False)

print(f"\n{B}{C}=== SOPHIE v3.3 Setup (NLP Upgrade) ==={X}\n")

# 1. Ordnerstruktur anlegen
# ---------------------------------------------------------
info("Prüfe Ordnerstruktur...")
dirs = ["data", "static", "model", "voice"]
for d in dirs:
    Path(d).mkdir(exist_ok=True)
ok("Ordner erstellt/geprüft")

# 2. Python-Pakete installieren
# ---------------------------------------------------------
# sentence-transformers & torch sind die schwersten Brocken
pkgs = [
    "numpy", 
    "scipy", 
    "scikit-learn",          # Für den TF-IDF Fallback
    "aiohttp", 
    "websockets", 
    "openwakeword", 
    "onnxruntime",
    "openai-whisper", 
    "torch", 
    "torchaudio",
    "sentence-transformers", # NEU: Für die Intent-Engine
    "pyttsx3",               # TTS Fallback
    "requests"               # Oft benötigt für Downloads
]

print(f"\n{B}Installiere Python-Bibliotheken (das kann dauern)...{X}")
for p in pkgs:
    info(f"Installiere {p}...")
    r = run(f'{sys.executable} -m pip install "{p}"')
    if r.returncode == 0:
        ok(p)
    else:
        warn(f"Installation von {p} gab einen Fehler zurück.")

# 3. openWakeWord Ressourcen
# ---------------------------------------------------------
print(f"\n{B}Lade openWakeWord Ressourcen...{X}")
try:
    import openwakeword
    res_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    res_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
    
    needed = ["melspectrogram.onnx", "embedding_model.onnx"]
    for fname in needed:
        dest = res_dir / fname
        if dest.exists():
            ok(f"{fname} bereit.")
            continue
        try:
            info(f"Lade {fname} herunter...")
            urllib.request.urlretrieve(base_url + fname, str(dest))
            ok(f"{fname} geladen.")
        except Exception as e:
            warn(f"Download Fehler {fname}: {e}")
except ImportError:
    warn("openwakeword konnte nicht importiert werden (Installation fehlgeschlagen?).")
except Exception as e:
    warn(f"openWakeWord Setup Fehler: {e}")

# 4. Sentence-Transformer Modell vorladen
# ---------------------------------------------------------
# Damit Sophie beim ersten Start nicht ewig hängt, laden wir das Modell jetzt.
ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"\n{B}Lade NLP-Modell: {ST_MODEL}...{X}")
try:
    from sentence_transformers import SentenceTransformer
    # Cache-Folder wird automatisch von library genutzt
    model = SentenceTransformer(ST_MODEL)
    ok("Sentence-Transformer Modell erfolgreich geladen.")
except ImportError:
    err("sentence-transformers Paket fehlt.")
except Exception as e:
    warn(f"Konnte Modell nicht vorladen: {e}\nSophie wird es beim Start versuchen.")

# 5. Whisper Modell vorladen
# ---------------------------------------------------------
WHISPER_MODEL = "small"
print(f"\n{B}Lade Whisper-Modell: {WHISPER_MODEL}...{X}")
try:
    import whisper
    # Das lädt es in den Cache (~/.cache/whisper oder Userprofile)
    whisper.load_model(WHISPER_MODEL)
    ok(f"Whisper {WHISPER_MODEL} bereit.")
except Exception as e:
    warn(f"Whisper Download Fehler: {e}")

# 6. Check Wakeword Modell (Custom)
# ---------------------------------------------------------
# Dateiname aus der Config von sophie.py v3.3
WAKE_MODEL = "model/sofie_20260213_173731.onnx"
print(f"\n{B}Prüfe Wakeword-Modell...{X}")
if Path(WAKE_MODEL).exists():
    ok(f"Gefunden: {WAKE_MODEL}")
else:
    warn(f"Datei nicht gefunden: {WAKE_MODEL}")
    info("Bitte stelle sicher, dass deine .onnx Datei im Ordner 'model/' liegt")
    info("und passe ggf. 'CONFIG[\"wake_model_path\"]' in sophie.py an.")

# 7. Abschluss
# ---------------------------------------------------------
print(f"\n{G}{B}Installation abgeschlossen!{X}")
print(f"""
  Starten mit:  {C}{sys.executable} sophie.py{X}
  
  Features:
  - {B}NLP{X}:    Sentence-Transformer ({ST_MODEL})
  - {B}ASR{X}:    Whisper ({WHISPER_MODEL}) + Fuzzy-Fixes
  - {B}TTS{X}:    API (Port 8020) oder Fallback pyttsx3
  - {B}Web{X}:    http://localhost:8765

  Hinweis: Falls du 'torch' mit CUDA-Support möchtest (für Nvidia GPUs),
  musst du PyTorch manuell gemäß https://pytorch.org/get-started/locally/ installieren.
""")