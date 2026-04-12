#!/usr/bin/env python3
"""
trigger.py  —  CLI + Modul für den VRM Avatar Server v2

Verwendung:
  python trigger.py emotion happy
  python trigger.py emotion sad

  python trigger.py talk on          # Mund bewegt sich automatisch
  python trigger.py talk off         # Mund stoppt

  python trigger.py anim Thinking    # Thinking-Animation abspielen
  python trigger.py anim Wave        # Wave-Animation abspielen

  python trigger.py demo             # Durchläuft alles automatisch

Oder als importierbares Modul:
  from trigger import set_emotion, set_talking, set_animation
  set_emotion('happy')
  set_talking(True)
  set_animation('Wave')
"""

import sys
import json
import urllib.request
import urllib.error
import time

# ── Konfiguration ─────────────────────────────────────────────────────────────
SERVER_URL = 'http://localhost:8000'

VALID_EMOTIONS   = ['happy', 'angry', 'sad', 'surprised', 'neutral']
VALID_ANIMATIONS = ['Thinking', 'Wave', 'Idle', 'Idle2', 'Idle3']


def _post(endpoint: str, payload: dict, server: str = SERVER_URL) -> dict:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url=f'{server}{endpoint}',
        data=data,
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


def set_emotion(emotion: str, server: str = SERVER_URL) -> dict:
    emotion = emotion.lower().strip()
    if emotion not in VALID_EMOTIONS:
        raise ValueError(f"Unbekannte Emotion '{emotion}'. Gültig: {VALID_EMOTIONS}")
    return _post('/emotion', {'emotion': emotion}, server)


def set_talking(talking: bool, server: str = SERVER_URL) -> dict:
    return _post('/talking', {'talking': talking}, server)


def set_animation(name: str, server: str = SERVER_URL) -> dict:
    if name not in VALID_ANIMATIONS:
        raise ValueError(f"Unbekannte Animation '{name}'. Gültig: {VALID_ANIMATIONS}")
    return _post('/animation', {'name': name}, server)


def demo_loop(server: str = SERVER_URL):
    print(f"🎭  VRM Avatar v2 Demo  →  {server}")
    print("     Drücke CTRL+C zum Abbrechen\n")

    try:
        # 1. Emotionen durchlaufen
        print("── Emotionen ──")
        for emo in VALID_EMOTIONS:
            print(f"  🎭  {emo}")
            set_emotion(emo, server)
            time.sleep(2.5)

        # 2. Talking testen
        print("\n── Talking ──")
        print("  🗣️  Talking AN")
        set_talking(True, server)
        time.sleep(5)
        print("  🤐  Talking AUS")
        set_talking(False, server)
        time.sleep(2)

        # 3. Animationen testen
        print("\n── Animationen ──")
        for anim in ['Thinking', 'Wave']:
            print(f"  🎬  {anim}")
            set_animation(anim, server)
            time.sleep(5)

        # 4. Combo: Emotion + Talking
        print("\n── Combo: Happy + Talking ──")
        set_emotion('happy', server)
        set_talking(True, server)
        time.sleep(4)
        set_talking(False, server)
        set_emotion('neutral', server)

        print("\n✅  Demo fertig!")

    except RuntimeError as e:
        print(f"\n  ❌  {e}")
    except KeyboardInterrupt:
        print("\n  ⏹️  Abgebrochen")
        set_talking(False, server)
        set_emotion('neutral', server)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'demo':
        demo_loop()

    elif cmd == 'emotion' and len(sys.argv) == 3:
        try:
            r = set_emotion(sys.argv[2])
            print(f"✅  Emotion → {r.get('emotion')}")
        except (ValueError, RuntimeError) as e:
            print(f"❌  {e}")
            sys.exit(1)

    elif cmd in ('talk', 'talking') and len(sys.argv) == 3:
        val = sys.argv[2].lower()
        if val in ('on', 'true', '1', 'an', 'ja'):
            talking = True
        elif val in ('off', 'false', '0', 'aus', 'nein'):
            talking = False
        else:
            print(f"❌  Unbekannt: '{val}'. Nutze: on/off")
            sys.exit(1)
        try:
            r = set_talking(talking)
            print(f"✅  Talking → {'AN' if talking else 'AUS'}")
        except RuntimeError as e:
            print(f"❌  {e}")
            sys.exit(1)

    elif cmd in ('anim', 'animation') and len(sys.argv) == 3:
        try:
            r = set_animation(sys.argv[2])
            print(f"✅  Animation → {r.get('animation')}")
        except (ValueError, RuntimeError) as e:
            print(f"❌  {e}")
            sys.exit(1)

    else:
        print(f"Verwendung: python {sys.argv[0]} [emotion|talk|anim|demo] [wert]")
        print(f"Emotionen:   {', '.join(VALID_EMOTIONS)}")
        print(f"Talking:     on / off")
        print(f"Animationen: {', '.join(VALID_ANIMATIONS)}")
        sys.exit(1)
