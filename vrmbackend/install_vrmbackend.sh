#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/root/M-ASSISTANT/vrmbackend"
PY_FILE="vrm_avatar_server.py"
VENV_DIR="$TARGET_DIR/.venv"
SERVICE_NAME="vrmbackend"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PORT="8000"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Bitte als root ausführen."
  exit 1
fi

echo "==> Installiere Systempakete"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  build-essential \
  libportaudio2 \
  portaudio19-dev \
  libasound2-dev \
  libsndfile1

mkdir -p "$TARGET_DIR"

# Falls die Python-Datei noch nicht im Ziel liegt, versuche sie aus dem aktuellen Ordner zu kopieren
if [[ ! -f "$TARGET_DIR/$PY_FILE" ]]; then
  if [[ -f "./$PY_FILE" ]]; then
    echo "==> Kopiere $PY_FILE nach $TARGET_DIR"
    cp -f "./$PY_FILE" "$TARGET_DIR/$PY_FILE"
  else
    echo "FEHLER: $TARGET_DIR/$PY_FILE fehlt."
    echo "Leg die Datei genau dort ab oder starte dieses Skript im Ordner, in dem $PY_FILE liegt."
    exit 1
  fi
fi

cd "$TARGET_DIR"

echo "==> Lege virtuelle Umgebung an"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Aktualisiere pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "==> Installiere Python-Abhängigkeiten"
pip install --upgrade numpy sounddevice

# Optionale requirements-Datei zum späteren Nachziehen
cat > "$TARGET_DIR/requirements.txt" <<REQ
numpy
sounddevice
REQ

# Harte Checks für Assets, damit der Service nicht dumm losrennt und dann stirbt
if [[ ! -f "$TARGET_DIR/model.vrm" ]]; then
  echo "WARNUNG: $TARGET_DIR/model.vrm fehlt. Der Dienst startet sonst nicht sinnvoll."
fi

mkdir -p "$TARGET_DIR/animations"
for anim in Idle.fbx Idle2.fbx Idle3.fbx; do
  if [[ ! -f "$TARGET_DIR/animations/$anim" ]]; then
    echo "WARNUNG: $TARGET_DIR/animations/$anim fehlt."
  fi
done

echo "==> Schreibe systemd-Service"
cat > "$SERVICE_FILE" <<SERVICE
[Unit]
Description=VRM Avatar Backend
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$TARGET_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/python $TARGET_DIR/$PY_FILE
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE

echo "==> Aktiviere und starte Service"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo
 echo "Fertig."
echo "Status prüfen:   systemctl status $SERVICE_NAME --no-pager -l"
echo "Logs ansehen:    journalctl -u $SERVICE_NAME -f"
echo "Im Browser:      http://$(hostname -I | awk '{print $1}'):$PORT"
echo
