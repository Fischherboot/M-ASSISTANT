#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="/opt/massis-whisper-host"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="massis-whisper-api"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
APP_USER="root"
APP_GROUP="root"
PORT="3502"
MODEL="small"
HOST="0.0.0.0"
PYTHON_BIN="python3"
CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu124"

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
BOLD="\033[1m"
RESET="\033[0m"

ok()   { echo -e "${GREEN}[OK]${RESET} $*"; }
info() { echo -e "${CYAN}[..]${RESET} $*"; }
warn() { echo -e "${YELLOW}[!!]${RESET} $*"; }
fail() { echo -e "${RED}[XX]${RESET} $*"; exit 1; }

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    fail "Bitte als root ausführen."
  fi
}

install_apt_packages() {
  info "Installiere Systempakete ..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    curl \
    ca-certificates
  ok "Systempakete installiert."
}

check_nvidia() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    info "NVIDIA erkannt:"
    nvidia-smi || true
    return 0
  fi

  warn "nvidia-smi wurde nicht gefunden."
  warn "Ich installiere hier absichtlich KEINEN NVIDIA-Treiber blind auf dem Proxmox-Host."
  warn "Wenn die Treiber noch fehlen oder kaputt sind, läuft der Service notfalls auf CPU."
  return 1
}

prepare_app_dir() {
  info "Lege App-Verzeichnis an: ${APP_DIR}"
  mkdir -p "$APP_DIR"

  if [[ -f "./whisper_api.py" ]]; then
    cp -f ./whisper_api.py "$APP_DIR/whisper_api.py"
  elif [[ -f "/root/whisper_api.py" ]]; then
    cp -f /root/whisper_api.py "$APP_DIR/whisper_api.py"
  else
    fail "whisper_api.py nicht gefunden. Lege install.sh und whisper_api.py in denselben Ordner oder nach /root."
  fi

  chmod 755 "$APP_DIR"
  chmod 644 "$APP_DIR/whisper_api.py"
  ok "Dateien bereitgestellt."
}

setup_venv() {
  info "Erstelle Python-Venv ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  info "Aktualisiere pip/setuptools/wheel ..."
  python -m pip install --upgrade pip setuptools wheel

  if check_nvidia; then
    info "Installiere PyTorch mit CUDA-Unterstützung ..."
    python -m pip install --upgrade --index-url "$CUDA_WHL_INDEX" torch torchaudio
  else
    info "Installiere Standard-PyTorch ..."
    python -m pip install --upgrade torch torchaudio
  fi

  info "Installiere Python-Abhängigkeiten ..."
  python -m pip install --upgrade numpy openai-whisper fastapi 'uvicorn[standard]'

  ok "Python-Umgebung fertig."
}

preload_model() {
  info "Lade Whisper-Modell vor (${MODEL}) ..."
  "$VENV_DIR/bin/python" - <<PY
import whisper
whisper.load_model("${MODEL}")
print("Whisper-Modell ${MODEL} ist geladen.")
PY
  ok "Whisper-Modell vorbereitet."
}

write_service() {
  info "Schreibe systemd-Service: ${SERVICE_FILE}"
  cat > "$SERVICE_FILE" <<EOF_SERVICE
[Unit]
Description=Moritz Whisper API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=NVIDIA_VISIBLE_DEVICES=all
ExecStart=${VENV_DIR}/bin/python ${APP_DIR}/whisper_api.py --host ${HOST} --port ${PORT} --model ${MODEL}
Restart=always
RestartSec=3
TimeoutStartSec=120
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF_SERVICE

  chmod 644 "$SERVICE_FILE"
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME"
  systemctl restart "$SERVICE_NAME"
  ok "Service aktiviert und gestartet."
}

show_status() {
  echo
  ok "Fertig."
  echo
  echo -e "${BOLD}Wichtige Befehle:${RESET}"
  echo "  systemctl status ${SERVICE_NAME} --no-pager"
  echo "  journalctl -u ${SERVICE_NAME} -f"
  echo "  curl http://127.0.0.1:${PORT}/health"
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  nvidia-smi"
  fi
}

main() {
  require_root
  install_apt_packages
  prepare_app_dir
  setup_venv
  preload_model
  write_service
  show_status
}

main "$@"
