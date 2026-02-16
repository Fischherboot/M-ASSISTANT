#!/usr/bin/env python3
"""
emotion_trigger.py  —  Beispiel-Script für den VRM Avatar Server Emotion-Webhook

Verwendung:
  python emotion_trigger.py happy
  python emotion_trigger.py sad
  python emotion_trigger.py angry
  python emotion_trigger.py surprised
  python emotion_trigger.py neutral

Oder als importierbares Modul:
  from emotion_trigger import set_emotion
  set_emotion('happy')

Oder interaktive Demo:
  python emotion_trigger.py
"""

import sys
import json
import urllib.request
import urllib.error
import time

# ── Konfiguration ─────────────────────────────────────────────────────────────
SERVER_URL = 'http://localhost:8000'

VALID_EMOTIONS = ['happy', 'angry', 'sad', 'surprised', 'neutral']


def set_emotion(emotion: str, server: str = SERVER_URL) -> dict:
    """
    Sendet eine Emotion an den VRM Avatar Server.

    Args:
        emotion: Eine von: happy, angry, sad, surprised, neutral
        server:  Server-URL (default: http://localhost:8000)

    Returns:
        Server-Antwort als dict, z.B. {'ok': True, 'emotion': 'happy'}

    Raises:
        ValueError:  Unbekannte Emotion
        RuntimeError: Server nicht erreichbar
    """
    emotion = emotion.lower().strip()
    if emotion not in VALID_EMOTIONS:
        raise ValueError(f"Unbekannte Emotion '{emotion}'. Gültig: {VALID_EMOTIONS}")

    payload = json.dumps({'emotion': emotion}).encode('utf-8')
    req = urllib.request.Request(
        url=f'{server}/emotion',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Server nicht erreichbar ({server}). "
            f"Läuft vrm_avatar_server.py?  Fehler: {e}"
        )


def demo_loop(server: str = SERVER_URL):
    """Durchläuft alle Emotionen automatisch — gut zum Testen."""
    print(f"🎭  Emotion-Demo  →  {server}")
    print("     Drücke CTRL+C zum Abbrechen\n")

    sequence = [
        ('happy',     '😊  Happy     — freudig'),
        ('surprised', '😲  Surprised — überrascht'),
        ('sad',       '😢  Sad       — traurig'),
        ('angry',     '😠  Angry     — wütend'),
        ('neutral',   '😐  Neutral   — neutral'),
    ]

    while True:
        for emotion, label in sequence:
            print(f"  → {label}")
            try:
                result = set_emotion(emotion, server)
                if not result.get('ok'):
                    print(f"     ⚠️  Fehler: {result}")
            except RuntimeError as e:
                print(f"  ❌  {e}")
                return
            time.sleep(2.5)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Keine Argumente → interaktive Demo
        demo_loop()

    elif len(sys.argv) == 2:
        arg = sys.argv[1].lower()

        if arg in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)

        if arg == 'demo':
            demo_loop()
            sys.exit(0)

        # Einzelne Emotion setzen
        try:
            result = set_emotion(arg)
            print(f"✅  Emotion gesetzt: {result['emotion']}")
        except ValueError as e:
            print(f"❌  {e}")
            sys.exit(1)
        except RuntimeError as e:
            print(f"❌  {e}")
            sys.exit(1)

    else:
        print(f"Verwendung: python {sys.argv[0]} [emotion|demo]")
        print(f"Emotionen: {', '.join(VALID_EMOTIONS)}")
        sys.exit(1)
