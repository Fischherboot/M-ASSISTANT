#!/usr/bin/env python3
"""
VRM Avatar Server v2  —  mit Talking-Trigger + Smooth Animation Blending
─────────────────────────────────────────────────────────────────────────
• Windows: WASAPI Loopback — fängt ALLES ein was der PC ausgibt
• Linux:   PulseAudio Monitor-Device
• Mac:     BlackHole / Multi-Output

Features:
  - Idle-Sequencer  (Idle.fbx / Idle2.fbx / Idle3.fbx, shuffle)
  - Multi-Viseme Lipsync  (aa/ih/ou/ee/oh + FFT)
  - Realistisches Blinken
  - Erweitertes Eye-Gazing  (normal · side · up-right · down · cross · roll)
  - Emotion-Webhooks    POST /emotion     {"emotion":"happy|angry|sad|surprised|neutral"}
  - Talking-Trigger     POST /talking     {"talking":true}  /  {"talking":false}
  - Animation-Trigger   POST /animation   {"name":"Thinking"}  /  {"name":"Wave"}
  - Smooth Crossfade mit Bone-Quaternion-SLERP (kein T-Pose Snapping)
  - Dezentes animiertes Rainbow-Glow + pulsierender Hintergrund
"""

import json
import sys
import platform
import numpy as np
import sounddevice as sd
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
import queue

# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHANNELS    = 1
BLOCKSIZE   = 2048

audio_queue         = queue.Queue(maxsize=10)
current_audio_level = 0.0
current_fft_bands   = [0.0, 0.0, 0.0, 0.0]
_fft_lock           = threading.Lock()

# ── Emotion state ─────────────────────────────────────────────────────────────
current_emotion = 'neutral'
_emotion_lock   = threading.Lock()
VALID_EMOTIONS  = {'happy', 'angry', 'sad', 'surprised', 'neutral'}

# ── Talking state ─────────────────────────────────────────────────────────────
current_talking = False
_talking_lock   = threading.Lock()

# ── Animation trigger state ──────────────────────────────────────────────────
current_anim_trigger = None
_anim_lock           = threading.Lock()
VALID_ANIMATIONS     = {'Thinking', 'Wave', 'Idle', 'Idle2', 'Idle3'}


# ── FFT bands ─────────────────────────────────────────────────────────────────
def _compute_fft_bands(data: np.ndarray, sr: int):
    n = len(data)
    if n < 64:
        return [0.0, 0.0, 0.0, 0.0]
    win   = np.hanning(n)
    spec  = np.abs(np.fft.rfft(data * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    def band(f_lo, f_hi):
        m = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.sqrt(np.mean(spec[m] ** 2))) if m.any() else 0.0
    lo, mlo, mhi, hi = band(0, 860), band(860, 2150), band(2150, 3440), band(3440, 6020)
    tot = lo + mlo + mhi + hi + 1e-9
    return [lo/tot, mlo/tot, mhi/tot, hi/tot]


def audio_callback(indata, frames, time_info, status):
    global current_audio_level, current_fft_bands
    if status:
        pass
    d   = indata[:, 0] if len(indata.shape) > 1 else indata
    lvl = min(float(np.sqrt(np.mean(d ** 2))) * 20, 1.0)
    current_audio_level = lvl
    with _fft_lock:
        current_fft_bands = _compute_fft_bands(d, SAMPLE_RATE)
    try:
        audio_queue.put_nowait(lvl)
    except queue.Full:
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(lvl)
        except Exception:
            pass


# ── Windows WASAPI Loopback helper ────────────────────────────────────────────
def _find_wasapi_loopback():
    try:
        hostapis = sd.query_hostapis()
        wasapi_index = next(
            (i for i, h in enumerate(hostapis) if 'WASAPI' in h['name']), None
        )
        if wasapi_index is None:
            return None, None
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['hostapi'] != wasapi_index:
                continue
            if dev['max_input_channels'] < 1:
                continue
            if 'loopback' in dev['name'].lower():
                return i, dev['name']
        default_out = sd.default.device[1]
        if default_out is not None:
            dev = devices[default_out]
            if dev['hostapi'] == wasapi_index:
                return default_out, dev['name'] + ' (loopback)'
        return None, None
    except Exception:
        return None, None


def list_input_devices():
    print("\n📊  Verfügbare Input-Devices:")
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            mark = " ⭐" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{mark}")
            input_devices.append(i)
    print()
    return input_devices


def start_audio_capture(device_index=None):
    is_windows = platform.system() == 'Windows'
    extra_kwargs = {}

    if device_index is not None:
        dev_name = sd.query_devices(device_index)['name']
        print(f"🎤  Audio-Device [{device_index}]: {dev_name}")
    else:
        if is_windows:
            lb_index, lb_name = _find_wasapi_loopback()
            if lb_index is not None:
                print(f"🎤  Windows WASAPI Loopback: [{lb_index}] {lb_name}")
                device_index = lb_index
                extra_kwargs = {'extra_settings': sd.WasapiSettings(loopback=True)}
            else:
                try:
                    default_out = sd.default.device[1]
                    if default_out is not None:
                        print(f"⚙️   WASAPI Loopback auf Default-Output [{default_out}]")
                        device_index = default_out
                        extra_kwargs = {'extra_settings': sd.WasapiSettings(loopback=True)}
                    else:
                        print("🎤  Kein Loopback gefunden → Standard-Mikrofon")
                except AttributeError:
                    print("⚠️   WasapiSettings fehlt → pip install sounddevice --upgrade")
        else:
            print("🎤  Standard-Input-Device")

    def _run(dev_idx, kwargs):
        try:
            kw = dict(callback=audio_callback, channels=CHANNELS,
                      samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, **kwargs)
            if dev_idx is not None:
                kw['device'] = dev_idx
            with sd.InputStream(**kw):
                print("✅  Audio-Capture aktiv!\n")
                while True:
                    sd.sleep(1000)
        except Exception as e:
            print(f"❌  Audio Error: {e}")
            if kwargs:
                print("   → Retry ohne Loopback…")
                try:
                    with sd.InputStream(callback=audio_callback, channels=CHANNELS,
                                        samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE):
                        print("✅  Fallback Mikrofon aktiv!\n")
                        while True:
                            sd.sleep(1000)
                except Exception as e2:
                    print(f"❌  Audio komplett fehlgeschlagen: {e2}")

    _run(device_index, extra_kwargs)


# ── Embedded HTML / JavaScript ────────────────────────────────────────────────
HTML_CONTENT = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; overflow: hidden; }
    body { background: #000000; }
    #canvas {
      position: absolute; top: 0; left: 0;
      width: 100%; height: 100%; display: block; z-index: 1;
    }
    #glow {
      position: absolute; top: 50%; left: 50%;
      transform: translate(-50%, -40%);
      width: 900px; height: 900px; z-index: 0; pointer-events: none;
      background: radial-gradient(circle,
        rgba(140,160,255,0.35) 0%,
        rgba(120,50,255,0.15) 35%,
        rgba(0,0,0,0) 70%
      );
      filter: blur(30px);
      animation: glowPulse 6s ease-in-out infinite;
    }
    @keyframes glowPulse {
      0%, 100% { opacity: 0.6; transform: translate(-50%, -40%) scale(1.0); }
      50%      { opacity: 1.0; transform: translate(-50%, -40%) scale(1.1); }
    }
    #status {
      position: fixed; bottom: 8px; left: 50%; transform: translateX(-50%);
      color: #444; font: 11px/1 monospace; pointer-events: none; z-index: 10;
    }
  </style>
</head>
<body>
  <div id="glow"></div>
  <canvas id="canvas"></canvas>
  <div id="status">loading...</div>

  <script type="importmap">
  {
    "imports": {
      "three":            "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
      "three/addons/":    "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
      "@pixiv/three-vrm": "https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@3.4.5/lib/three-vrm.module.js"
    }
  }
  </script>

  <script type="module">
  import * as THREE from 'three';
  import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
  import { FBXLoader }  from 'three/addons/loaders/FBXLoader.js';
  import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

  // ── Config ───────────────────────────────────────────────────────────────
  const AVATAR_SCALE       = 1.8;
  const CROSSFADE_DURATION = 2.5;   // Langsames, smoothes Übergleiten

  // ── Scene ────────────────────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.1, 100);
  camera.position.set(0, 1.5, 3.5);
  camera.lookAt(0, 1.5, 0);

  const renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('canvas'),
    antialias: true, alpha: true,
    precision: 'highp', powerPreference: 'high-performance',
  });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const kl = new THREE.DirectionalLight(0xffffff, 1.0);
  kl.position.set(0, 2, 3); scene.add(kl);
  const fl = new THREE.DirectionalLight(0x8899ff, 0.3);
  fl.position.set(-2, 1, 1); scene.add(fl);

  // ── Mixamo → VRM bone map ────────────────────────────────────────────────
  const BONE_MAP = {
    mixamorigHips:'hips',mixamorigSpine:'spine',mixamorigSpine1:'chest',
    mixamorigSpine2:'upperChest',mixamorigNeck:'neck',mixamorigHead:'head',
    mixamorigLeftShoulder:'leftShoulder',mixamorigLeftArm:'leftUpperArm',
    mixamorigLeftForeArm:'leftLowerArm',mixamorigLeftHand:'leftHand',
    mixamorigLeftHandThumb1:'leftThumbMetacarpal',mixamorigLeftHandThumb2:'leftThumbProximal',
    mixamorigLeftHandThumb3:'leftThumbDistal',mixamorigLeftHandIndex1:'leftIndexProximal',
    mixamorigLeftHandIndex2:'leftIndexIntermediate',mixamorigLeftHandIndex3:'leftIndexDistal',
    mixamorigLeftHandMiddle1:'leftMiddleProximal',mixamorigLeftHandMiddle2:'leftMiddleIntermediate',
    mixamorigLeftHandMiddle3:'leftMiddleDistal',mixamorigLeftHandRing1:'leftRingProximal',
    mixamorigLeftHandRing2:'leftRingIntermediate',mixamorigLeftHandRing3:'leftRingDistal',
    mixamorigLeftHandPinky1:'leftLittleProximal',mixamorigLeftHandPinky2:'leftLittleIntermediate',
    mixamorigLeftHandPinky3:'leftLittleDistal',mixamorigRightShoulder:'rightShoulder',
    mixamorigRightArm:'rightUpperArm',mixamorigRightForeArm:'rightLowerArm',
    mixamorigRightHand:'rightHand',mixamorigRightHandThumb1:'rightThumbMetacarpal',
    mixamorigRightHandThumb2:'rightThumbProximal',mixamorigRightHandThumb3:'rightThumbDistal',
    mixamorigRightHandIndex1:'rightIndexProximal',mixamorigRightHandIndex2:'rightIndexIntermediate',
    mixamorigRightHandIndex3:'rightIndexDistal',mixamorigRightHandMiddle1:'rightMiddleProximal',
    mixamorigRightHandMiddle2:'rightMiddleIntermediate',mixamorigRightHandMiddle3:'rightMiddleDistal',
    mixamorigRightHandRing1:'rightRingProximal',mixamorigRightHandRing2:'rightRingIntermediate',
    mixamorigRightHandRing3:'rightRingDistal',mixamorigRightHandPinky1:'rightLittleProximal',
    mixamorigRightHandPinky2:'rightLittleIntermediate',mixamorigRightHandPinky3:'rightLittleDistal',
    mixamorigLeftUpLeg:'leftUpperLeg',mixamorigLeftLeg:'leftLowerLeg',
    mixamorigLeftFoot:'leftFoot',mixamorigLeftToeBase:'leftToes',
    mixamorigRightUpLeg:'rightUpperLeg',mixamorigRightLeg:'rightLowerLeg',
    mixamorigRightFoot:'rightFoot',mixamorigRightToeBase:'rightToes',
  };

  // ── FBX Retargeting ──────────────────────────────────────────────────────
  async function loadMixamoAnimation(url, vrm) {
    const asset = await new FBXLoader().loadAsync(url);
    const clip  = asset.animations[0];
    if (!clip) return null;
    const tracks=[], rInv=new THREE.Quaternion(), pRot=new THREE.Quaternion();
    const _q=new THREE.Quaternion(), _v=new THREE.Vector3();
    const mhH = asset.getObjectByName('mixamorigHips')?.position.y ?? 1;
    const vhY = vrm.humanoid?.getNormalizedBoneNode('hips')?.getWorldPosition(_v).y ?? 0;
    const hS  = Math.abs(vhY - vrm.scene.getWorldPosition(_v).y) / mhH;
    clip.tracks.forEach(t => {
      const [mx, prop] = t.name.split('.');
      const bn = BONE_MAP[mx]; if (!bn) return;
      const nn = vrm.humanoid?.getNormalizedBoneNode(bn)?.name;
      const mn = asset.getObjectByName(mx);
      if (!nn||!mn) return;
      mn.getWorldQuaternion(rInv).invert();
      mn.parent.getWorldQuaternion(pRot);
      if (t instanceof THREE.QuaternionKeyframeTrack) {
        for (let i=0;i<t.values.length;i+=4) {
          _q.set(t.values[i],t.values[i+1],t.values[i+2],t.values[i+3]);
          _q.premultiply(pRot).multiply(rInv);
          t.values[i]=_q.x;t.values[i+1]=_q.y;t.values[i+2]=_q.z;t.values[i+3]=_q.w;
        }
        tracks.push(new THREE.QuaternionKeyframeTrack(`${nn}.${prop}`,t.times,
          t.values.map((v,i)=>vrm.meta?.metaVersion==='0'&&i%2===0?-v:v)));
      } else if (t instanceof THREE.VectorKeyframeTrack && bn==='hips') {
        const vn=vrm.humanoid?.getNormalizedBoneNode(bn);
        const[rx,ry,rz]=[vn?.position.x??0,vn?.position.y??0,vn?.position.z??0];
        const my=mn.position.y;
        const val=new Float32Array(t.values.length);
        for(let i=0;i<t.values.length;i+=3){val[i]=rx;val[i+1]=ry+(t.values[i+1]-my)*hS;val[i+2]=rz;}
        tracks.push(new THREE.VectorKeyframeTrack(`${nn}.${prop}`,t.times,val));
      }
    });
    return tracks.length ? new THREE.AnimationClip('va',clip.duration,tracks) : null;
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  SMOOTH CROSSFADE ENGINE  —  SLERP-basiert, kein T-Pose Snapping
  // ════════════════════════════════════════════════════════════════════════════
  //
  //  Das Problem: Three.js AnimationMixer.crossFadeTo() interpoliert intern
  //  über Gewichte. Wenn ein Clip Bones nicht enthält die der vorherige hatte,
  //  snappen diese Bones zur Rest-Pose (T-Pose).
  //
  //  Lösung: Wir cachen die Quaternions + Positions aller Bones am Ende
  //  jeder Animation. Beim Crossfade erstellen wir einen "Bridge-Clip" der
  //  von den gecachten Werten zum neuen Clip interpoliert. So hat jeder
  //  Bone immer eine explizite Animation und es gibt keinen Snap.
  // ════════════════════════════════════════════════════════════════════════════

  let curAction = null;
  let mixer = null;
  const boneCache = new Map();   // boneName → {q: Quaternion, p: Vector3}

  function cacheBonePoses() {
    if (!vrm) return;
    vrm.scene.traverse(obj => {
      if (obj.isBone) {
        boneCache.set(obj.name, {
          q: obj.quaternion.clone(),
          p: obj.position.clone()
        });
      }
    });
  }

  // Ease-in-out Kurve (smooth-step)
  function easeInOut(t) {
    return t * t * (3 - 2 * t);
  }

  function createBridgeClip(targetClip, duration) {
    // Erstellt einen Clip der von gecachten Poses zum Start des targetClip interpoliert
    // mit 16 Zwischenframes auf einer ease-in-out Kurve für butterweiche Übergänge
    if (boneCache.size === 0) return null;

    const STEPS = 16;
    const tracks = [];
    const targetTrackMap = new Map();

    for (const t of targetClip.tracks) {
      const boneName = t.name.split('.')[0];
      const prop = t.name.split('.')[1];
      targetTrackMap.set(`${boneName}.${prop}`, t);
    }

    for (const [boneName, cached] of boneCache) {
      // Quaternion track mit ease-in-out SLERP
      const qKey = `${boneName}.quaternion`;
      const targetQTrack = targetTrackMap.get(qKey);

      if (targetQTrack) {
        const startQ = cached.q.clone();
        const endQ = new THREE.Quaternion(
          targetQTrack.values[0], targetQTrack.values[1],
          targetQTrack.values[2], targetQTrack.values[3]
        );

        const times = [];
        const values = [];
        const _tmpQ = new THREE.Quaternion();

        for (let i = 0; i <= STEPS; i++) {
          const linearT = i / STEPS;
          const easedT  = easeInOut(linearT);
          times.push(linearT * duration);
          _tmpQ.copy(startQ).slerp(endQ, easedT);
          values.push(_tmpQ.x, _tmpQ.y, _tmpQ.z, _tmpQ.w);
        }

        tracks.push(new THREE.QuaternionKeyframeTrack(qKey, times, values));
      }

      // Position track mit ease-in-out LERP
      const pKey = `${boneName}.position`;
      const targetPTrack = targetTrackMap.get(pKey);
      if (targetPTrack) {
        const startP = cached.p.clone();
        const endP = new THREE.Vector3(
          targetPTrack.values[0], targetPTrack.values[1], targetPTrack.values[2]
        );

        const times = [];
        const values = [];
        const _tmpV = new THREE.Vector3();

        for (let i = 0; i <= STEPS; i++) {
          const linearT = i / STEPS;
          const easedT  = easeInOut(linearT);
          times.push(linearT * duration);
          _tmpV.copy(startP).lerp(endP, easedT);
          values.push(_tmpV.x, _tmpV.y, _tmpV.z);
        }

        tracks.push(new THREE.VectorKeyframeTrack(pKey, times, values));
      }
    }

    if (tracks.length === 0) return null;
    return new THREE.AnimationClip('_bridge', duration, tracks);
  }

  function smoothCrossfade(newClip, fadeDuration = CROSSFADE_DURATION) {
    if (!vrm || !mixer) return;

    // Cache aktuelle Bone-Poses bevor wir irgendwas ändern
    cacheBonePoses();

    // Alle laufenden Actions sanft stoppen
    mixer.stopAllAction();

    // Bridge-Clip erstellen: von aktueller Pose → Start des neuen Clips
    const bridgeDur = fadeDuration;
    const bridgeClip = createBridgeClip(newClip, bridgeDur);

    if (bridgeClip && boneCache.size > 0) {
      // Phase 1: Bridge-Animation spielen
      const bridgeAction = mixer.clipAction(bridgeClip);
      bridgeAction.setLoop(THREE.LoopOnce, 1);
      bridgeAction.clampWhenFinished = false;
      bridgeAction.setEffectiveWeight(1);
      bridgeAction.play();

      // Phase 2: Neuen Clip parallel starten mit fadeIn
      const newAction = mixer.clipAction(newClip);
      newAction.reset();
      newAction.setEffectiveTimeScale(1);
      newAction.setEffectiveWeight(0);
      newAction.play();
      newAction.fadeIn(bridgeDur);

      // Bridge nach Ablauf stoppen
      bridgeAction.fadeOut(bridgeDur * 0.8);

      curAction = newAction;

      // Cleanup: Bridge-Action nach Ablauf entfernen
      setTimeout(() => {
        bridgeAction.stop();
        mixer.uncacheAction(bridgeClip);
        mixer.uncacheClip(bridgeClip);
      }, bridgeDur * 1000 + 200);
    } else {
      // Kein Cache vorhanden (erster Start) → normaler Start
      const newAction = mixer.clipAction(newClip);
      newAction.reset();
      newAction.setEffectiveTimeScale(1);
      newAction.setEffectiveWeight(1);
      newAction.play();
      curAction = newAction;
    }
  }

  // ── Idle Sequencer ───────────────────────────────────────────────────────
  const IDLE_URLS = ['/animations/Idle.fbx', '/animations/Idle2.fbx', '/animations/Idle3.fbx'];
  const ALL_ANIM_URLS = {
    'Idle':     '/animations/Idle.fbx',
    'Idle2':    '/animations/Idle2.fbx',
    'Idle3':    '/animations/Idle3.fbx',
    'Thinking': '/animations/Thinking.fbx',
    'Wave':     '/animations/Wave.fbx',
  };

  let vrm = null;
  const clipCache = {};

  async function getClip(url) {
    if (clipCache[url]) return clipCache[url];
    try {
      const c = await loadMixamoAnimation(url, vrm);
      if (c) clipCache[url] = c;
      return c;
    } catch(e) {
      console.warn('Anim missing:', url);
      return null;
    }
  }

  function shuffle(a) {
    const b = [...a];
    for (let i = b.length - 1; i > 0; i--) {
      const j = ~~(Math.random() * (i + 1));
      [b[i], b[j]] = [b[j], b[i]];
    }
    return b;
  }

  let idleSeq = [], idlePtr = 0, idleTimer = null;
  let isPlayingSpecial = false;   // Thinking / Wave gerade aktiv?

  function ensureMixer() {
    if (!mixer && vrm) mixer = new THREE.AnimationMixer(vrm.scene);
  }

  async function playClip(url) {
    if (!vrm) return;
    ensureMixer();
    const c = await getClip(url);
    if (c) smoothCrossfade(c);
  }

  function nextIdle() {
    if (!vrm || isPlayingSpecial) return;
    if (idlePtr >= idleSeq.length) { idleSeq = shuffle(IDLE_URLS); idlePtr = 0; }
    playClip(idleSeq[idlePtr++]);
    idleTimer = setTimeout(nextIdle, 8000);
  }

  function startIdleSequencer() {
    if (idleTimer) clearTimeout(idleTimer);
    isPlayingSpecial = false;
    idleSeq = shuffle(IDLE_URLS);
    idlePtr = 0;
    nextIdle();
  }

  // ── Animation Trigger (Thinking, Wave, etc.) ─────────────────────────────
  let lastAnimTrigger = null;

  async function checkAnimTrigger() {
    try {
      const d = await (await fetch('/animation-data')).json();
      if (d.animation && d.animation !== lastAnimTrigger) {
        lastAnimTrigger = d.animation;
        const url = ALL_ANIM_URLS[d.animation];
        if (!url) return;

        // Idle-Sequencer pausieren
        if (idleTimer) clearTimeout(idleTimer);
        isPlayingSpecial = true;

        await playClip(url);

        // Nach Clip-Dauer zurück zu Idle
        const clip = await getClip(url);
        const dur = clip ? clip.duration * 1000 : 5000;
        setTimeout(() => {
          isPlayingSpecial = false;
          lastAnimTrigger = null;
          startIdleSequencer();
        }, dur + 500);
      } else if (!d.animation) {
        lastAnimTrigger = null;
      }
    } catch(_) {}
    setTimeout(checkAnimTrigger, 200);
  }

  // ── VRM Load ─────────────────────────────────────────────────────────────
  const loader = new GLTFLoader();
  loader.register(p => new VRMLoaderPlugin(p));
  setStatus('Loading VRM…');
  loader.load('/model.vrm',
    async gltf => {
      vrm = gltf.userData.vrm;
      if (!vrm) { setStatus('❌ No VRM'); return; }
      if (VRMUtils.removeUnnecessaryVertices) VRMUtils.removeUnnecessaryVertices(vrm.scene);
      if (VRMUtils.combineSkeletons) VRMUtils.combineSkeletons(vrm.scene);
      else if (VRMUtils.removeUnnecessaryJoints) VRMUtils.removeUnnecessaryJoints(vrm.scene);
      if (VRMUtils.rotateVRM0) VRMUtils.rotateVRM0(vrm);
      vrm.scene.traverse(o => { if (o.isMesh) o.frustumCulled = false; });
      vrm.scene.position.set(0, 0, 0);
      vrm.scene.rotation.y = 0;
      vrm.scene.scale.setScalar(AVATAR_SCALE);
      scene.add(vrm.scene);
      console.log('✅ VRM | expressions:', Object.keys(vrm.expressionManager?.expressionMap ?? {}));

      // Pre-load alle Animations
      setTimeout(async () => {
        const allUrls = Object.values(ALL_ANIM_URLS);
        await Promise.allSettled(allUrls.map(u => getClip(u)));
        setStatus('ready');
        setTimeout(() => setStatus(''), 2000);
        startIdleSequencer();
        checkAnimTrigger();
      }, 200);
    },
    p => setStatus(`Loading VRM… ${(p.loaded / p.total * 100).toFixed(0)}%`),
    e => { console.error(e); setStatus('❌ Load error'); }
  );

  // ── Audio polling ─────────────────────────────────────────────────────────
  let audioLevel = 0, fftBands = [0,0,0,0];
  const glowEl = document.getElementById('glow');
  async function pollAudio() {
    try {
      const d = await (await fetch('/audio-data')).json();
      audioLevel = d.level;
      fftBands = d.bands ?? [0,0,0,0];
      if (glowEl) {
        const boost = 0.7 + audioLevel * 0.6;
        glowEl.style.opacity = Math.min(boost, 1.4).toFixed(2);
      }
    } catch(_) {}
    setTimeout(pollAudio, 50);
  }
  pollAudio();

  // ── Emotion polling ───────────────────────────────────────────────────────
  const EMOTION_SHAPES = {
    happy:     ['happy','Happy','joy','Joy'],
    angry:     ['angry','Angry','anger','Anger'],
    sad:       ['sad','Sad','sorrow','Sorrow'],
    surprised: ['surprised','Surprised','surprise','Surprise'],
    neutral:   [],
  };
  const ALL_EMO_KEYS = Object.values(EMOTION_SHAPES).flat();
  let activeEmotion = 'neutral', emoTarget = {}, emoCurrent = {};
  ALL_EMO_KEYS.forEach(k => { emoTarget[k] = 0; emoCurrent[k] = 0; });

  function setEmotionTarget(em) {
    activeEmotion = em;
    ALL_EMO_KEYS.forEach(k => emoTarget[k] = 0);
    (EMOTION_SHAPES[em] ?? []).forEach(n => emoTarget[n] = 1.0);
  }

  async function pollEmotion() {
    try {
      const d = await (await fetch('/emotion-data')).json();
      if (d.emotion !== activeEmotion) setEmotionTarget(d.emotion);
    } catch(_) {}
    setTimeout(pollEmotion, 100);
  }
  pollEmotion();

  function updateEmotion() {
    if (!vrm?.expressionManager) return;
    const em = vrm.expressionManager;
    for (const [k, t] of Object.entries(emoTarget)) {
      const c = emoCurrent[k] ?? 0, n = c + (t - c) * 0.08;
      emoCurrent[k] = n; safeSet(em, k, clamp(n));
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  TALKING ENGINE  —  Prozedurale Mundbewegungen ohne Audio
  // ════════════════════════════════════════════════════════════════════════════
  //
  //  Wenn /talking {talking:true} gesendet wird, generiert der Client
  //  natürlich aussehende Viseme-Animationen über mehrere überlagerte
  //  Sinus-Wellen mit unterschiedlichen Frequenzen und Phasen.
  //
  //  Die Idee: Echtes Sprechen hat Grundfrequenz (Kiefer auf/zu ca. 4-6 Hz),
  //  überlagert mit langsameren Formant-Wechseln (welcher Vokal, 1-2 Hz)
  //  und gelegentlichen Pausen.
  // ════════════════════════════════════════════════════════════════════════════

  let isTalking = false;
  let talkTime  = 0;      // Akkumulierte Zeit seit Talking-Start

  // Pseudo-Random mit Seed für konsistente aber natürliche Patterns
  function hash(n) {
    let x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }

  // Smooth Noise (Value-Noise 1D)
  function smoothNoise(t) {
    const i = Math.floor(t);
    const f = t - i;
    const u = f * f * (3 - 2 * f);  // smoothstep
    return hash(i) * (1 - u) + hash(i + 1) * u;
  }

  async function pollTalking() {
    try {
      const d = await (await fetch('/talking-data')).json();
      isTalking = !!d.talking;
      if (!isTalking) talkTime = 0;
    } catch(_) {}
    setTimeout(pollTalking, 80);
  }
  pollTalking();

  let pAa=0, pIh=0, pOu=0, pEe=0, pOh=0;

  function updateTalkingMouth(dt) {
    if (!vrm?.expressionManager) return;
    const em = vrm.expressionManager;

    if (!isTalking) {
      // Mund sanft schließen wenn nicht mehr talking
      const fade = 0.95;  // Sehr langsam schließen
      pAa *= fade; pIh *= fade; pOu *= fade; pEe *= fade; pOh *= fade;
      safeSet(em, 'aa', pAa); safeSet(em, 'ih', pIh);
      safeSet(em, 'ou', pOu); safeSet(em, 'ee', pEe); safeSet(em, 'oh', pOh);
      return false;  // nicht aktiv
    }

    talkTime += dt;
    const t = talkTime;

    // ── Jaw cycle: Kiefer öffnen/schließen 2-3 Hz (langsames natürliches Sprechen) ──
    const jawFreq = 2.0 + smoothNoise(t * 0.3) * 1.0;  // 2.0-3.0 Hz
    const jawBase = Math.sin(t * jawFreq * Math.PI * 2) * 0.5 + 0.5;
    const jawOpen = jawBase * (0.5 + smoothNoise(t * 1.2) * 0.45);  // Mund weiter auf (max ~0.95)

    // ── Pausen einfügen: natürliches Sprechen hat Mikro-Pausen ──
    const pauseWave = smoothNoise(t * 0.5);
    const isPause = pauseWave < 0.15;  // ~15% der Zeit kurze Pause
    const pauseMult = isPause ? 0.05 : 1.0;

    // ── Viseme-Selektion: welcher Vokal dominiert (wechselt langsam ~0.8 Hz) ──
    const visemePhase  = t * 0.8;
    const visemeSelect = smoothNoise(visemePhase);

    // Verschiedene Viseme-Gewichte basierend auf Phase
    let tAa, tIh, tOu, tEe, tOh;

    if (visemeSelect < 0.25) {
      // "aa" dominant — offener Mund
      tAa = jawOpen * 0.80;
      tIh = jawOpen * 0.10;
      tOu = jawOpen * 0.05;
      tEe = jawOpen * 0.04;
      tOh = jawOpen * 0.15;
    } else if (visemeSelect < 0.45) {
      // "ee/ih" dominant — breiter Mund
      tAa = jawOpen * 0.15;
      tIh = jawOpen * 0.60;
      tOu = jawOpen * 0.03;
      tEe = jawOpen * 0.50;
      tOh = jawOpen * 0.08;
    } else if (visemeSelect < 0.65) {
      // "ou" dominant — runder Mund
      tAa = jawOpen * 0.12;
      tIh = jawOpen * 0.05;
      tOu = jawOpen * 0.70;
      tEe = jawOpen * 0.03;
      tOh = jawOpen * 0.40;
    } else if (visemeSelect < 0.82) {
      // "oh" dominant
      tAa = jawOpen * 0.20;
      tIh = jawOpen * 0.06;
      tOu = jawOpen * 0.15;
      tEe = jawOpen * 0.04;
      tOh = jawOpen * 0.65;
    } else {
      // Mix — Übergangslaut
      tAa = jawOpen * 0.30;
      tIh = jawOpen * 0.20;
      tOu = jawOpen * 0.20;
      tEe = jawOpen * 0.18;
      tOh = jawOpen * 0.25;
    }

    // Pause multiplier anwenden
    tAa *= pauseMult;
    tIh *= pauseMult;
    tOu *= pauseMult;
    tEe *= pauseMult;
    tOh *= pauseMult;

    // Smoothing (stärkere Glättung für langsamere, weichere Bewegung)
    const s = 0.55;
    pAa += (tAa - pAa) * (1 - s);
    pIh += (tIh - pIh) * (1 - s);
    pOu += (tOu - pOu) * (1 - s);
    pEe += (tEe - pEe) * (1 - s);
    pOh += (tOh - pOh) * (1 - s);

    safeSet(em, 'aa', clamp(pAa));
    safeSet(em, 'ih', clamp(pIh));
    safeSet(em, 'ou', clamp(pOu));
    safeSet(em, 'ee', clamp(pEe));
    safeSet(em, 'oh', clamp(pOh));

    return true;  // aktiv
  }

  // ── Audio-basierter Lipsync (Original, nur aktiv wenn NICHT talking) ─────
  function updateLipSync() {
    if (!vrm?.expressionManager) return;
    const em = vrm.expressionManager;

    if (audioLevel < 0.02) {
      // Nur faden wenn talking-engine nicht aktiv ist
      if (!isTalking) {
        const f = 0.85;
        pAa *= f; pIh *= f; pOu *= f; pEe *= f; pOh *= f;
        safeSet(em, 'aa', pAa); safeSet(em, 'ih', pIh);
        safeSet(em, 'ou', pOu); safeSet(em, 'ee', pEe); safeSet(em, 'oh', pOh);
      }
      return;
    }

    const [nL, nML, nMH, nH] = fftBands;
    const sensitivity = 0.6;
    const a = audioLevel * sensitivity;

    let tAa = Math.min(nL * 1.4 * a * 2.0, 0.60);
    let tOh = Math.min((nL * .5 + nML * .5) * a * 1.6, 0.50);
    let tIh = Math.min((nML * .8 + nH * .4) * a * 1.6, 0.40);
    let tEe = Math.min(nMH * 1.2 * a * 1.8, 0.40);
    let tOu = Math.min((nMH * .6 + nL * .3) * a * 1.4, 0.40);

    if (tAa + tIh + tOu + tEe + tOh < 0.10) {
      tAa = Math.max(a * 0.5, 0.10);
    }

    const s = 0.35;
    pAa += (tAa - pAa) * (1 - s);
    pIh += (tIh - pIh) * (1 - s);
    pOu += (tOu - pOu) * (1 - s);
    pEe += (tEe - pEe) * (1 - s);
    pOh += (tOh - pOh) * (1 - s);

    safeSet(em, 'aa', clamp(pAa));
    safeSet(em, 'ih', clamp(pIh));
    safeSet(em, 'ou', clamp(pOu));
    safeSet(em, 'ee', clamp(pEe));
    safeSet(em, 'oh', clamp(pOh));
  }

  // ── Blink ─────────────────────────────────────────────────────────────────
  let blinkT = 0, blinkN = null;
  function detectBlink() {
    if (!vrm?.expressionManager) return null;
    for (const n of ['blink', 'Blink', 'blinkLeft', 'blinkRight', 'BLINK']) {
      try { vrm.expressionManager.setValue(n, 0); return n; } catch(_) {}
    }
    return null;
  }
  function doBlink() {
    if (!blinkN) return;
    const em = vrm.expressionManager;
    let t = 0;
    const cl = setInterval(() => {
      t += 0.1; safeSet(em, blinkN, Math.min(t, 1));
      if (t >= 1) {
        clearInterval(cl);
        setTimeout(() => {
          let o = 1;
          const op = setInterval(() => {
            o -= 0.15; safeSet(em, blinkN, Math.max(o, 0));
            if (o <= 0) clearInterval(op);
          }, 12);
        }, 80);
      }
    }, 15);
  }

  // ── Extended Eye Gazing ───────────────────────────────────────────────────
  let gazeTimer = 0, gazeHold = 2.0, gazeMode = 'normal';
  let gTX = 0, gTY = 0, gCX = 0, gCY = 0;
  let rollPhase = 'idle', rollProg = 0;

  function pickGaze() {
    const r = Math.random();
    rollPhase = 'idle'; rollProg = 0;
    if      (r < 0.40) { gazeMode = 'normal';   gTX = (Math.random() - .5) * .25; gTY = (Math.random() - .5) * .15; gazeHold = 1.5 + Math.random() * 2.5; }
    else if (r < 0.60) { gazeMode = 'side';     gTX = (Math.random() < .5 ? -1 : 1) * (.35 + Math.random() * .2); gTY = (Math.random() - .5) * .08; gazeHold = 1.0 + Math.random() * 2.0; }
    else if (r < 0.72) { gazeMode = 'up-right'; gTX = 0.28 + Math.random() * .12; gTY = -0.28 - Math.random() * .12; gazeHold = 1.2 + Math.random() * 1.5; }
    else if (r < 0.80) { gazeMode = 'down';     gTX = (Math.random() - .5) * .08; gTY = 0.22 + Math.random() * .1; gazeHold = 1.0 + Math.random() * 2.0; }
    else if (r < 0.90) { gazeMode = 'cross';    gTX = (Math.random() - .5) * .04; gTY = 0.08 + Math.random() * .04; gazeHold = 0.5 + Math.random() * 0.8; }
    else               { gazeMode = 'roll';     gTX = 0; gTY = 0; gazeHold = 2.5; rollPhase = 'rolling-up'; rollProg = 0; }
  }

  function updateGaze(dt) {
    if (!vrm?.lookAt) return;
    gazeTimer += dt;
    if (gazeTimer >= gazeHold && rollPhase === 'idle') { gazeTimer = 0; pickGaze(); }
    if (rollPhase === 'rolling-up') { rollProg += dt * 0.7; gTY = -rollProg * 0.55; if (rollProg >= 1.0) rollPhase = 'rolling-down'; }
    else if (rollPhase === 'rolling-down') { rollProg -= dt * 0.55; gTY = -rollProg * 0.55; if (rollProg <= 0) { rollPhase = 'idle'; rollProg = 0; gTY = 0; gTX = 0; gazeTimer = 0; } }
    const sp = gazeMode === 'cross' ? 0.14 : 0.04;
    gCX += (gTX - gCX) * sp; gCY += (gTY - gCY) * sp;
    vrm.lookAt.yaw = gCX; vrm.lookAt.pitch = gCY;
  }

  // ── Main loop ─────────────────────────────────────────────────────────────
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    if (!vrm) { renderer.render(scene, camera); return; }
    const dt = clock.getDelta();
    if (mixer) mixer.update(dt);
    vrm.update(dt);

    if (!blinkN) blinkN = detectBlink();
    blinkT += dt;
    if (blinkT > 3.5 + Math.random() * 1.5) { blinkT = 0; doBlink(); }

    updateGaze(dt);

    // Talking-Trigger hat Priorität über Audio-Lipsync
    const talkingActive = updateTalkingMouth(dt);
    if (!talkingActive) {
      updateLipSync();
    }

    updateEmotion();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  });

  function safeSet(em, n, v) { try { em.setValue(n, v); } catch(_) {} }
  function clamp(v, lo=0, hi=1) { return Math.max(lo, Math.min(hi, v)); }
  function setStatus(msg) { document.getElementById('status').textContent = msg; }
  </script>
</body>
</html>
"""


# ── HTTP Handler ──────────────────────────────────────────────────────────────
class VRMHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._bytes(HTML_CONTENT.encode(), 'text/html')
        elif path == '/model.vrm':
            self._file(Path(__file__).parent / 'model.vrm', 'application/octet-stream')
        elif path.startswith('/animations/'):
            name = path[len('/animations/'):]
            self._file(Path(__file__).parent / 'animations' / name, 'application/octet-stream')
        elif path == '/audio-data':
            with _fft_lock:
                bands = list(current_fft_bands)
            self._bytes(json.dumps({
                'level': float(current_audio_level),
                'bands': bands,
            }).encode(), 'application/json')
        elif path == '/emotion-data':
            with _emotion_lock:
                em = current_emotion
            self._bytes(json.dumps({'emotion': em}).encode(), 'application/json')
        elif path == '/talking-data':
            with _talking_lock:
                tk = current_talking
            self._bytes(json.dumps({'talking': tk}).encode(), 'application/json')
        elif path == '/animation-data':
            with _anim_lock:
                anim = current_anim_trigger
            self._bytes(json.dumps({'animation': anim}).encode(), 'application/json')
        else:
            self.send_error(404)

    def do_POST(self):
        global current_emotion, current_talking, current_anim_trigger
        path = urlparse(self.path).path

        if path == '/emotion':
            try:
                data = self._read_json()
                emotion = str(data.get('emotion', '')).lower().strip()
                if emotion not in VALID_EMOTIONS:
                    self._json_error(f'Unknown emotion. Valid: {sorted(VALID_EMOTIONS)}')
                    return
                with _emotion_lock:
                    current_emotion = emotion
                print(f"🎭  Emotion → {emotion}")
                self._bytes(json.dumps({'ok': True, 'emotion': emotion}).encode(), 'application/json')
            except Exception as e:
                self._json_error(str(e))

        elif path == '/talking':
            try:
                data = self._read_json()
                talking = bool(data.get('talking', False))
                with _talking_lock:
                    current_talking = talking
                state = "AN" if talking else "AUS"
                print(f"🗣️  Talking → {state}")
                self._bytes(json.dumps({'ok': True, 'talking': talking}).encode(), 'application/json')
            except Exception as e:
                self._json_error(str(e))

        elif path == '/animation':
            try:
                data = self._read_json()
                name = str(data.get('name', '')).strip()
                if name not in VALID_ANIMATIONS:
                    self._json_error(f'Unknown animation. Valid: {sorted(VALID_ANIMATIONS)}')
                    return
                with _anim_lock:
                    current_anim_trigger = name
                print(f"🎬  Animation → {name}")
                self._bytes(json.dumps({'ok': True, 'animation': name}).encode(), 'application/json')
            except Exception as e:
                self._json_error(str(e))

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)
        return json.loads(body)

    def _json_error(self, msg, status=400):
        self._bytes(json.dumps({'error': msg}).encode(), 'application/json', status=status)

    def _bytes(self, data: bytes, ct: str, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _file(self, fp: Path, ct: str):
        if not fp.exists():
            self.send_error(404, f'{fp.name} not found')
            return
        self._bytes(fp.read_bytes(), ct)

    def log_message(self, format, *args):
        pass


# ── Startup checks ────────────────────────────────────────────────────────────
def check_requirements() -> bool:
    base  = Path(__file__).parent
    vrm   = base / 'model.vrm'
    anims = base / 'animations'

    print("=" * 70)
    print("🎭  VRM AVATAR SERVER v2")
    print("   Talking-Trigger + Smooth Crossfade + Animation Webhooks")
    print("=" * 70)
    print()
    ok = True

    if vrm.exists():
        print(f"✅  model.vrm  ({vrm.stat().st_size / 1_048_576:.2f} MB)")
    else:
        print(f"❌  model.vrm nicht gefunden  ({vrm})")
        ok = False

    if anims.is_dir():
        print(f"✅  animations/")
        for n in ('Idle.fbx', 'Idle2.fbx', 'Idle3.fbx', 'Thinking.fbx', 'Wave.fbx'):
            p = anims / n
            s = f"({p.stat().st_size // 1024} KB)" if p.exists() else "fehlt"
            print(f"    {'✅' if p.exists() else '⚠️ '}  {n}  {s}")
    else:
        print(f"⚠️   animations/ Ordner fehlt")

    print()
    if not ok:
        print("💡  VRM Models:   https://hub.vroid.com/")
        print("💡  Animationen:  https://www.mixamo.com/")
        print()
    return ok


def main():
    import sys as _sys

    selected_device = None
    if len(_sys.argv) > 1:
        arg = _sys.argv[1]
        if arg in ('-h', '--help'):
            print("Verwendung:")
            print("  python vrm_avatar_server.py          → startet + zeigt alle Devices")
            print("  python vrm_avatar_server.py -N       → verwendet Device Nummer N")
            print("  python vrm_avatar_server.py --list   → zeigt nur Device-Liste")
            _sys.exit(0)
        if arg == '--list':
            list_input_devices()
            _sys.exit(0)
        if arg.startswith('-') and arg[1:].lstrip('-').isdigit():
            selected_device = int(arg.lstrip('-'))
        else:
            print(f"❌  Unbekanntes Argument: {arg}")
            _sys.exit(1)

    if selected_device is None:
        list_input_devices()
        os_name = platform.system()
        if os_name == 'Windows':
            print("ℹ️   Windows: WASAPI Loopback wird automatisch gesucht")
            if not hasattr(sd, 'WasapiSettings'):
                print("   ⚠️  sounddevice zu alt: pip install sounddevice --upgrade")
        elif os_name == 'Darwin':
            print("ℹ️   Mac: BlackHole als Standard-Input setzen")
        else:
            print("ℹ️   Linux: PulseAudio Monitor-Device als Input setzen")
        print()
    else:
        try:
            dev = sd.query_devices(selected_device)
            if dev['max_input_channels'] < 1:
                print(f"❌  Device [{selected_device}] hat keine Input-Channels!")
                _sys.exit(1)
            print(f"✅  Gewähltes Device: [{selected_device}] {dev['name']}")
        except Exception as e:
            print(f"❌  Device [{selected_device}] nicht gefunden: {e}")
            list_input_devices()
            _sys.exit(1)

    if not check_requirements():
        if input("Ohne model.vrm geht nichts. Trotzdem starten? (j/n): ").strip().lower() != 'j':
            return

    threading.Thread(
        target=start_audio_capture,
        args=(selected_device,),
        daemon=True
    ).start()

    PORT   = 8000
    server = HTTPServer(('', PORT), VRMHandler)

    print(f"🌐  http://localhost:{PORT}")
    print()
    print(f"🎬  Idle-Sequencer   (Idle1/2/3 shuffle, 8s/clip, smooth crossfade)")
    print(f"👄  Multi-Viseme      (aa/ih/ou/ee/oh + FFT)")
    print(f"👁️  Blink + Gazing   (normal · side · up-right · down · cross · roll)")
    print(f"🌈  Glow Aura")
    print()
    print(f"── Webhooks ────────────────────────────────────────────────")
    print(f"🎭  POST /emotion     {{\"emotion\": \"happy\"}}       happy|angry|sad|surprised|neutral")
    print(f"🗣️  POST /talking     {{\"talking\": true}}          Mund bewegt sich automatisch")
    print(f"🗣️  POST /talking     {{\"talking\": false}}         Mund stoppt")
    print(f"🎬  POST /animation   {{\"name\": \"Thinking\"}}       Thinking|Wave|Idle|Idle2|Idle3")
    print()
    print("⌨️   CTRL+C zum Beenden")
    print("=" * 70)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋  Shutdown…")
        server.shutdown()


if __name__ == '__main__':
    main()
