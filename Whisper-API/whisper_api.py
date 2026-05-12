#!/usr/bin/env python3
"""
Sophie Whisper-API – Standalone GPU-Transkriptions-Server
Läuft auf dem Host-System (mit GPU), wird von Sophie im LXC aufgerufen.

Starten:  python whisper_api.py [--port 3502] [--model small] [--device cuda]

Endpoint:
  POST /transcribe
    Body: raw PCM float32 @ 16kHz (application/octet-stream)
    Response: {"text": "transkribierter text"}

  GET /health
    Response: {"status": "ok", "model": "small", "device": "cuda"}
"""

import argparse
import io
import logging
import struct
import threading
import queue
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whisper-api")

# ── Konfiguration ──────────────────────────────────────────────────────────────
DEFAULT_PORT   = 3502
DEFAULT_MODEL  = "small"
DEFAULT_DEVICE = "cuda"
LANGUAGE       = "de"
WHISPER_LEN    = 480000   # 30s @ 16kHz – Whisper-Standardlänge

# ── Globale Whisper-Instanz ────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_job_queue: queue.Queue = queue.Queue()
_worker_started = False

def load_model(model_name: str, device: str):
    """Whisper-Modell laden (einmalig)."""
    global _model
    with _model_lock:
        if _model is None:
            import whisper
            log.info(f"Lade Whisper-{model_name} auf {device} ...")
            _model = whisper.load_model(model_name, device=device)
            log.info(f"Whisper-{model_name} bereit auf {device}.")
    return _model


def _pad(samples: np.ndarray) -> np.ndarray:
    """Auf Whisper-Eingabelänge padden/trimmen."""
    if len(samples) >= WHISPER_LEN:
        return samples[:WHISPER_LEN]
    return np.pad(samples, (0, WHISPER_LEN - len(samples)))


def transcribe(samples: np.ndarray) -> str:
    """Transkribiert float32 Samples mit dem geladenen Modell."""
    if _model is None:
        raise RuntimeError("Modell nicht geladen")
    try:
        result = _model.transcribe(
            _pad(samples),
            language=LANGUAGE,
            fp16=False,
            condition_on_previous_text=False,
        )
        return result.get("text", "").strip()
    except Exception as e:
        log.error(f"Transkriptions-Fehler: {e}")
        return ""


# ── Worker-Thread: serialisiert GPU-Zugriffe ──────────────────────────────────
def _worker_loop():
    """Nimmt Jobs aus der Queue, transkribiert sequenziell."""
    log.info("Whisper-Worker bereit.")
    while True:
        samples, event, result_box = _job_queue.get()
        try:
            result_box["text"] = transcribe(samples)
        except Exception as e:
            log.error(f"Worker-Fehler: {e}")
            result_box["text"] = ""
        finally:
            event.set()


def _ensure_worker():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        t = threading.Thread(target=_worker_loop, daemon=True, name="whisper-worker")
        t.start()


def transcribe_queued(samples: np.ndarray) -> str:
    """Thread-safe: Job in die Queue, auf Ergebnis warten."""
    _ensure_worker()
    event = threading.Event()
    result_box: dict = {}
    _job_queue.put((samples, event, result_box))
    event.wait(timeout=60)
    return result_box.get("text", "")


# ── FastAPI-Server ─────────────────────────────────────────────────────────────
def create_app(model_name: str, device: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Sophie Whisper-API", version="1.0")

    @app.on_event("startup")
    async def startup():
        load_model(model_name, device)
        _ensure_worker()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": model_name,
            "device": device,
            "ready": _model is not None,
        }

    @app.post("/transcribe")
    async def do_transcribe(request: Request):
        """
        Erwartet raw PCM float32 @ 16kHz als Body (application/octet-stream).
        Gibt {"text": "..."} zurück.
        """
        body = await request.body()
        if not body:
            return JSONResponse({"text": "", "error": "Kein Audio empfangen"}, status_code=400)

        # bytes → numpy float32
        try:
            samples = np.frombuffer(body, dtype=np.float32)
        except Exception as e:
            return JSONResponse({"text": "", "error": f"Audio-Dekodierung fehlgeschlagen: {e}"}, status_code=400)

        if len(samples) < 8000:  # < 0.5s @ 16kHz
            return JSONResponse({"text": "", "error": "Audio zu kurz"}, status_code=400)

        t0 = time.time()
        text = transcribe_queued(samples)
        dt = time.time() - t0

        log.info(f"Transkription ({len(samples)/16000:.1f}s Audio → {dt:.2f}s): '{text}'")
        return {"text": text}

    return app


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sophie Whisper-API (GPU)")
    parser.add_argument("--port",   type=int, default=DEFAULT_PORT,   help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL,  help=f"Whisper-Modell (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help=f"Device (default: {DEFAULT_DEVICE})")
    parser.add_argument("--host",   type=str, default="0.0.0.0",      help="Bind-Adresse (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║  Sophie Whisper-API                                  ║
║  Modell:  whisper-{args.model:<39s}║
║  Device:  {args.device:<43s}║
║  Port:    {args.port:<43d}║
╚══════════════════════════════════════════════════════╝
    """)

    import uvicorn
    app = create_app(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
