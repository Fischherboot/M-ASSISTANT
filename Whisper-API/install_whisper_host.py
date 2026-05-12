#!/usr/bin/env python3
"""Sophie Whisper-API – Host-Installation (GPU)"""
import subprocess
import sys

# Farben
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
B = "\033[1m"
X = "\033[0m"

def ok(m): print(f"{G}  v{X} {m}")
def info(m): print(f"{C}  >{X} {m}")
def warn(m): print(f"{Y}  !{X} {m}")
def run(cmd): 
    return subprocess.run(cmd, shell=True, check=False)

print(f"\n{B}{C}=== Sophie Whisper-API – Host Setup (GPU) ==={X}\n")

# 1. Pakete installieren
# ---------------------------------------------------------
pkgs = [
    "numpy",
    "openai-whisper",    # Whisper ASR
    "torch",             # PyTorch (CUDA)
    "torchaudio",
    "fastapi",           # API-Framework
    "uvicorn[standard]", # ASGI-Server
]

print(f"{B}Installiere Python-Bibliotheken...{X}")
for p in pkgs:
    info(f"Installiere {p}...")
    r = run(f'{sys.executable} -m pip install "{p}"')
    if r.returncode == 0:
        ok(p)
    else:
        warn(f"Installation von {p} gab einen Fehler zurück.")

# 2. Whisper vorladen
# ---------------------------------------------------------
WHISPER_MODEL = "small"
print(f"\n{B}Lade Whisper-Modell: {WHISPER_MODEL}...{X}")
try:
    import whisper
    whisper.load_model(WHISPER_MODEL)
    ok(f"Whisper {WHISPER_MODEL} bereit.")
except Exception as e:
    warn(f"Whisper Download Fehler: {e}")

# 3. CUDA-Check
# ---------------------------------------------------------
print(f"\n{B}Prüfe GPU...{X}")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        ok(f"CUDA verfügbar: {gpu_name}")
    else:
        warn("CUDA NICHT verfügbar – Whisper läuft auf CPU (langsam!)")
        info("Installiere PyTorch mit CUDA: https://pytorch.org/get-started/locally/")
except ImportError:
    warn("PyTorch nicht gefunden.")

# 4. Fertig
# ---------------------------------------------------------
print(f"\n{G}{B}Host-Installation abgeschlossen!{X}")
print(f"""
  Starten mit:  {C}{sys.executable} whisper_api.py{X}
  
  Optionen:
    --port 3502      (Standard)
    --model small    (oder medium/large für bessere Qualität)
    --device cuda    (oder cpu)
  
  Test:
    curl http://localhost:3502/health

  {Y}Hinweis:{X} Falls du PyTorch mit CUDA-Support brauchst,
  installiere es manuell: https://pytorch.org/get-started/locally/
  Beispiel: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
""")
