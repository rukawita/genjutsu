// Motor Neural Local de Genjutsu Studio (100% Gratis, Sin Token, En Tiempo Real)
// Basado en Google MediaPipe Tasks Vision (WebGL2 / WASM):
//  1. FaceLandmarker (478 puntos 3D faciales + blendshapes de expresión)
//  2. PoseLandmarker (33 puntos corporales 3D + máscara de segmentación)
//  3. ImageSegmenter (recorte neural de persona / eliminación de fondo cuadrado)
//  4. Mallado Delaunay de 468 vértices + Deformación Afín por Triángulos (Piecewise Affine Warp)
//  5. Transferencia de iluminación y tono de piel (Reinhard / RGB Illumination Matching)

import {
  FilesetResolver,
  FaceLandmarker,
  PoseLandmarker,
  ImageSegmenter,
} from "/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";

// Contorno ordenado oficial del óvalo facial de MediaPipe (36 puntos perimetrales)
const FACE_OVAL_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

// Subconjunto denso pero ultra-rápido de ~175 puntos clave anatómicos de los 468 de MediaPipe
// (Frente, cejas, párpados, nariz, pómulos, labios exteriores/interiores, mentón y mandíbula)
const KEY_FACE_INDICES = Array.from(
  new Set([
    ...FACE_OVAL_INDICES,
    // Cejas izquierda y derecha
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    // Ojo izquierdo
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
    // Ojo derecho
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
    // Puente y punta de la nariz
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 48, 278,
    // Labios exteriores e interiores (boca completa para hablar/sonreír)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
    // Mejillas, pómulos y surcos nasogenianos
    116, 117, 118, 119, 120, 121, 128, 245, 193, 50, 101, 36, 205, 206, 207, 187, 147,
    345, 346, 347, 348, 349, 350, 357, 465, 417, 280, 330, 266, 425, 426, 427, 411, 376,
    // Mentón y centro frente
    151, 9, 8, 199, 175, 171, 140, 369, 396,
  ]),
);

// Triangulación de Delaunay (algoritmo Bowyer-Watson 2D)
function computeDelaunayTriangles(points) {
  const n = points.length;
  if (n < 3) return [];

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const dx = maxX - minX || 1;
  const dy = maxY - minY || 1;
  const deltaMax = Math.max(dx, dy) * 10;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  const allPts = [
    ...points,
    { x: midX - 2 * deltaMax, y: midY - deltaMax },
    { x: midX, y: midY + 2 * deltaMax },
    { x: midX + 2 * deltaMax, y: midY - deltaMax },
  ];

  let triangles = [[n, n + 1, n + 2]];

  for (let i = 0; i < n; i++) {
    const px = allPts[i].x;
    const py = allPts[i].y;
    const bad = [];

    for (let t = 0; t < triangles.length; t++) {
      const [a, b, c] = triangles[t];
      const ax = allPts[a].x, ay = allPts[a].y;
      const bx = allPts[b].x, by = allPts[b].y;
      const cx = allPts[c].x, cy = allPts[c].y;

      const D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
      if (Math.abs(D) < 1e-12) continue;
      const aSq = ax * ax + ay * ay;
      const bSq = bx * bx + by * by;
      const cSq = cx * cx + cy * cy;
      const ux = (aSq * (by - cy) + bSq * (cy - ay) + cSq * (ay - by)) / D;
      const uy = (aSq * (cx - bx) + bSq * (ax - cx) + cSq * (bx - ax)) / D;
      const rSq = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
      const dSq = (px - ux) * (px - ux) + (py - uy) * (py - uy);
      if (dSq <= rSq) bad.push(t);
    }

    const polygon = [];
    for (const tIdx of bad) {
      const [a, b, c] = triangles[tIdx];
      const edges = [[a, b], [b, c], [c, a]];
      for (const [e0, e1] of edges) {
        let shared = false;
        for (const otherIdx of bad) {
          if (otherIdx === tIdx) continue;
          const ot = triangles[otherIdx];
          if (ot.includes(e0) && ot.includes(e1)) {
            shared = true;
            break;
          }
        }
        if (!shared) polygon.push([e0, e1]);
      }
    }

    const badSet = new Set(bad);
    triangles = triangles.filter((_, idx) => !badSet.has(idx));
    for (const [e0, e1] of polygon) {
      triangles.push([e0, e1, i]);
    }
  }

  return triangles.filter(([a, b, c]) => a < n && b < n && c < n);
}

// Deformación afín exacta de un triángulo (Piecewise Affine Transform)
function warpTriangle(dstCtx, srcCanvas, s0, s1, s2, d0, d1, d2) {
  const dx1 = s1.x - s0.x;
  const dy1 = s1.y - s0.y;
  const dx2 = s2.x - s0.x;
  const dy2 = s2.y - s0.y;
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-5) return;

  const a = ((d1.x - d0.x) * dy2 - (d2.x - d0.x) * dy1) / det;
  const b = ((d1.y - d0.y) * dy2 - (d2.y - d0.y) * dy1) / det;
  const c = ((d2.x - d0.x) * dx1 - (d1.x - d0.x) * dx2) / det;
  const d = ((d2.y - d0.y) * dx1 - (d1.y - d0.y) * dx2) / det;
  const e = d0.x - a * s0.x - c * s0.y;
  const f = d0.y - b * s0.x - d * s0.y;

  // Expandir 0.75px desde el centroide para evitar costuras sub-píxel entre triángulos
  const cx = (d0.x + d1.x + d2.x) / 3;
  const cy = (d0.y + d1.y + d2.y) / 3;
  const expand = (pt) => {
    const vx = pt.x - cx;
    const vy = pt.y - cy;
    const len = Math.hypot(vx, vy) || 1;
    return { x: pt.x + (vx / len) * 0.75, y: pt.y + (vy / len) * 0.75 };
  };
  const p0 = expand(d0);
  const p1 = expand(d1);
  const p2 = expand(d2);

  dstCtx.save();
  dstCtx.beginPath();
  dstCtx.moveTo(p0.x, p0.y);
  dstCtx.lineTo(p1.x, p1.y);
  dstCtx.lineTo(p2.x, p2.y);
  dstCtx.closePath();
  dstCtx.clip();
  dstCtx.setTransform(a, b, c, d, e, f);
  dstCtx.drawImage(srcCanvas, 0, 0);
  dstCtx.restore();
}

function computeFaceColorStats(ctx2d, landmarks, width, height) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const idx of FACE_OVAL_INDICES) {
    const p = landmarks[idx];
    if (!p) continue;
    const x = p.x * width;
    const y = p.y * height;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  const x0 = Math.max(0, Math.floor(minX + (maxX - minX) * 0.18));
  const y0 = Math.max(0, Math.floor(minY + (maxY - minY) * 0.18));
  const w = Math.max(4, Math.min(width - x0, Math.floor((maxX - minX) * 0.64)));
  const h = Math.max(4, Math.min(height - y0, Math.floor((maxY - minY) * 0.64)));

  const data = ctx2d.getImageData(x0, y0, w, h).data;
  let rSum = 0, gSum = 0, bSum = 0, count = 0;
  for (let i = 0; i < data.length; i += 16) {
    if (data[i + 3] < 128) continue;
    rSum += data[i];
    gSum += data[i + 1];
    bSum += data[i + 2];
    count++;
  }
  if (!count) return { mean: [140, 115, 100], std: [40, 35, 32] };
  const mr = rSum / count;
  const mg = gSum / count;
  const mb = bSum / count;

  let vr = 0, vg = 0, vb = 0;
  for (let i = 0; i < data.length; i += 16) {
    if (data[i + 3] < 128) continue;
    vr += (data[i] - mr) ** 2;
    vg += (data[i + 1] - mg) ** 2;
    vb += (data[i + 2] - mb) ** 2;
  }
  return {
    mean: [mr, mg, mb],
    std: [
      Math.max(12, Math.sqrt(vr / count)),
      Math.max(12, Math.sqrt(vg / count)),
      Math.max(12, Math.sqrt(vb / count)),
    ],
  };
}

export class GenjutsuNeuralEngine {
  constructor() {
    this.ready = false;
    this.initializing = null;
    this.faceImageDetector = null;
    this.faceVideoDetector = null;
    this.poseImageDetector = null;
    this.poseVideoDetector = null;
    this.segmenter = null;

    // Cache del personaje fuente (@image1)
    this.sourceProfile = null;
    // Suavizado temporal para evitar parpadeos entre fotogramas del video
    this.smoothedTargetLandmarks = null;
    this.smoothedColorStats = null;
    this.lastVideoTimestampMs = -1;
    this.lastDetectionResult = { faceLandmarks: null, poseLandmarks: null };

    // Canvas auxiliares reutilizables
    this.workCanvas = document.createElement("canvas");
    this.workCtx = this.workCanvas.getContext("2d", { willReadFrequently: true });
    this.maskCanvas = document.createElement("canvas");
    this.maskCtx = this.maskCanvas.getContext("2d");
    this.colorMatchedSourceCanvas = document.createElement("canvas");
    this.colorMatchedSourceCtx = this.colorMatchedSourceCanvas.getContext("2d", { willReadFrequently: true });
  }

  async init() {
    if (this.ready) return;
    if (this.initializing) return this.initializing;

    this.initializing = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "/node_modules/@mediapipe/tasks-vision/wasm"
      );

      const [faceImg, faceVid, poseImg, poseVid, seg] = await Promise.all([
        FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
          runningMode: "IMAGE",
          numFaces: 1,
          minFaceDetectionConfidence: 0.2,
          minFacePresenceConfidence: 0.2,
        }),
        FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 2,
          minFaceDetectionConfidence: 0.2,
          minFacePresenceConfidence: 0.2,
          minTrackingConfidence: 0.2,
        }),
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: "/models/pose_landmarker_lite.task", delegate: "GPU" },
          runningMode: "IMAGE",
          numPoses: 1,
        }),
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: "/models/pose_landmarker_lite.task", delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.25,
          minTrackingConfidence: 0.25,
        }),
        ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: "/models/selfie_segmenter.tflite", delegate: "GPU" },
          runningMode: "IMAGE",
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        }),
      ]);

      this.faceImageDetector = faceImg;
      this.faceVideoDetector = faceVid;
      this.poseImageDetector = poseImg;
      this.poseVideoDetector = poseVid;
      this.segmenter = seg;
      this.ready = true;
    })();

    return this.initializing;
  }

  /**
   * Analiza la foto subida por el usuario (@image1):
   *  - Extrae los 478 puntos faciales 3D y construye la malla de triángulos Delaunay
   *  - Elimina el fondo cuadrado con la red neuronal selfie_segmenter
   *  - Extrae la pose corporal de 33 puntos
   */
  async analyzeSourceImage(imgEl) {
    await this.init();
    const w = imgEl.naturalWidth || imgEl.width || 512;
    const h = imgEl.naturalHeight || imgEl.height || 512;

    // 1. Canvas con la imagen original
    const rawCanvas = document.createElement("canvas");
    rawCanvas.width = w;
    rawCanvas.height = h;
    const rawCtx = rawCanvas.getContext("2d", { willReadFrequently: true });
    rawCtx.drawImage(imgEl, 0, 0, w, h);
    const rawImageData = rawCtx.getImageData(0, 0, w, h);

    // 2. Recorte neural del personaje sin fondo (ImageSegmenter)
    const cutoutCanvas = document.createElement("canvas");
    cutoutCanvas.width = w;
    cutoutCanvas.height = h;
    const cutoutCtx = cutoutCanvas.getContext("2d", { willReadFrequently: true });
    const cutoutData = new ImageData(new Uint8ClampedArray(rawImageData.data), w, h);

    try {
      const segResult = this.segmenter.segment(imgEl);
      const mask = segResult?.confidenceMasks?.[0];
      if (mask) {
        const maskFloat = mask.getAsFloat32Array();
        for (let i = 0; i < maskFloat.length; i++) {
          // Suavizado sigmoidal en el borde para evitar recortes duros
          const conf = maskFloat[i];
          const alpha = Math.max(0, Math.min(1, (conf - 0.25) / 0.45));
          cutoutData.data[i * 4 + 3] = Math.round(alpha * 255);
        }
      }
    } catch {
      // Si el segmentador falla, aplicamos viñeta elíptica suave para nunca mostrar bordes cuadrados
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const nx = (x / w - 0.5) * 2;
          const ny = (y / h - 0.5) * 2;
          const dist = Math.hypot(nx, ny);
          const alpha = Math.max(0, Math.min(1, (1.02 - dist) * 3.2));
          cutoutData.data[(y * w + x) * 4 + 3] = Math.round(alpha * 255);
        }
      }
    }
    cutoutCtx.putImageData(cutoutData, 0, 0);

    // 3. Detección de 478 puntos faciales en @image1
    const faceRes = this.faceImageDetector.detect(imgEl);
    let landmarks = faceRes?.faceLandmarks?.[0] || null;
    const poseRes = this.poseImageDetector.detect(imgEl);
    const poseLandmarks = poseRes?.landmarks?.[0] || null;

    // Si la foto es de un personaje estilizado donde no se detectó malla, generamos una malla anatómica centrada
    if (!landmarks) {
      landmarks = this.buildSyntheticCanonicalFace(w, h, poseLandmarks);
    }

    // Construir los puntos clave en píxeles de @image1 y triangular con Delaunay
    const keyPointsPx = KEY_FACE_INDICES.map((lmIdx) => ({
      lmIdx,
      x: landmarks[lmIdx].x * w,
      y: landmarks[lmIdx].y * h,
    }));
    const rawTris = computeDelaunayTriangles(keyPointsPx);
    // Mapear cada triángulo a los índices reales de MediaPipe (0..467)
    const meshTriangles = rawTris.map(([a, b, c]) => [
      KEY_FACE_INDICES[a],
      KEY_FACE_INDICES[b],
      KEY_FACE_INDICES[c],
    ]);

    const colorStats = computeFaceColorStats(rawCtx, landmarks, w, h);

    // Centro y escala base de la cara fuente
    const ovalPts = FACE_OVAL_INDICES.map((idx) => ({
      x: landmarks[idx].x * w,
      y: landmarks[idx].y * h,
    }));
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of ovalPts) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    this.sourceProfile = {
      width: w,
      height: h,
      rawCanvas,
      rawImageData,
      cutoutCanvas,
      landmarks,
      poseLandmarks,
      meshTriangles,
      colorStats,
      faceBox: {
        cx: (minX + maxX) / 2,
        cy: (minY + maxY) / 2,
        w: Math.max(32, maxX - minX),
        h: Math.max(32, maxY - minY),
      },
    };

    this.smoothedTargetLandmarks = null;
    this.smoothedColorStats = null;
    return this.sourceProfile;
  }

  buildSyntheticCanonicalFace(w, h, pose) {
    const cx = pose?.[0]?.x ?? 0.5;
    const cy = pose?.[0]?.y ?? 0.36;
    const rx = 0.18;
    const ry = 0.22;
    const pts = [];
    for (let i = 0; i < 478; i++) {
      const angle = (i / 478) * Math.PI * 2;
      const ring = 0.25 + ((i * 37) % 100) / 135;
      pts.push({
        x: Math.max(0.05, Math.min(0.95, cx + Math.cos(angle) * rx * ring)),
        y: Math.max(0.05, Math.min(0.95, cy + Math.sin(angle) * ry * ring)),
        z: 0,
      });
    }
    return pts;
  }

  /**
   * Detecta cara (478 puntos) y cuerpo (33 puntos) en el fotograma actual del video.
   */
  detectVideoFrame(videoEl) {
    if (!this.ready || !videoEl || videoEl.readyState < 2) {
      return { faceLandmarks: null, poseLandmarks: null };
    }
    const nowMs = Math.floor(performance.now());
    if (nowMs <= this.lastVideoTimestampMs) {
      return this.lastDetectionResult;
    }
    this.lastVideoTimestampMs = nowMs;

    let faceLandmarks = null;
    let poseLandmarks = null;
    try {
      const fRes = this.faceVideoDetector.detectForVideo(videoEl, nowMs);
      if (fRes?.faceLandmarks?.length > 0) {
        faceLandmarks = fRes.faceLandmarks[0];
      }
    } catch {
      // ignore frame collision
    }
    try {
      const pRes = this.poseVideoDetector.detectForVideo(videoEl, nowMs + 1);
      if (pRes?.landmarks?.length > 0) {
        poseLandmarks = pRes.landmarks[0];
      }
    } catch {
      // ignore frame collision
    }

    this.lastDetectionResult = { faceLandmarks, poseLandmarks };
    return this.lastDetectionResult;
  }

  /**
   * Si en un fotograma el actor está lejos o de perfil y solo PoseLandmarker ve su cabeza,
   * proyecta la malla de 478 puntos sobre la cabeza del actor (nariz, ojos, orejas) para mantener el Face Swap.
   */
  synthesizeFaceFromPoseHead(poseLandmarks, srcLandmarks) {
    if (!poseLandmarks || !srcLandmarks) return null;
    const nose = poseLandmarks[0];
    const lEye = poseLandmarks[2];
    const rEye = poseLandmarks[5];
    const lEar = poseLandmarks[7];
    const rEar = poseLandmarks[8];
    if (!nose || (nose.visibility !== undefined && nose.visibility < 0.25)) return null;

    const earDist = lEar && rEar ? Math.hypot(rEar.x - lEar.x, rEar.y - lEar.y) : 0.14;
    const eyeDist = lEye && rEye ? Math.hypot(rEye.x - lEye.x, rEye.y - lEye.y) : earDist * 0.5;
    const headScale = Math.max(0.06, Math.max(earDist * 1.15, eyeDist * 2.3));
    const angle =
      lEye && rEye ? Math.atan2(rEye.y - lEye.y, rEye.x - lEye.x) + Math.PI : 0;

    const srcNose = srcLandmarks[1];
    const srcLEye = srcLandmarks[33];
    const srcREye = srcLandmarks[263];
    const srcEyeDist = Math.max(0.04, Math.hypot(srcREye.x - srcLEye.x, srcREye.y - srcLEye.y));
    const ratio = (headScale * 0.48) / srcEyeDist;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    return srcLandmarks.map((pt) => {
      const dx = (pt.x - srcNose.x) * ratio;
      const dy = (pt.y - srcNose.y) * ratio;
      return {
        x: nose.x + dx * cosA - dy * sinA,
        y: nose.y + dx * sinA + dy * cosA,
        z: pt.z,
      };
    });
  }

  /**
   * Armoniza el color e iluminación del rostro fuente (@image1) con la iluminación del actor en el video.
   */
  updateColorMatchedSource(targetColorStats, strength = 0.62) {
    const sp = this.sourceProfile;
    if (!sp) return null;

    if (!this.smoothedColorStats) {
      this.smoothedColorStats = targetColorStats;
    } else {
      for (let c = 0; c < 3; c++) {
        this.smoothedColorStats.mean[c] =
          this.smoothedColorStats.mean[c] * 0.75 + targetColorStats.mean[c] * 0.25;
        this.smoothedColorStats.std[c] =
          this.smoothedColorStats.std[c] * 0.75 + targetColorStats.std[c] * 0.25;
      }
    }

    const w = sp.width;
    const h = sp.height;
    if (this.colorMatchedSourceCanvas.width !== w || this.colorMatchedSourceCanvas.height !== h) {
      this.colorMatchedSourceCanvas.width = w;
      this.colorMatchedSourceCanvas.height = h;
    }

    const srcMean = sp.colorStats.mean;
    const srcStd = sp.colorStats.std;
    const dstMean = this.smoothedColorStats.mean;
    const dstStd = this.smoothedColorStats.std;

    const scaleR = Math.max(0.65, Math.min(1.45, dstStd[0] / srcStd[0]));
    const scaleG = Math.max(0.65, Math.min(1.45, dstStd[1] / srcStd[1]));
    const scaleB = Math.max(0.65, Math.min(1.45, dstStd[2] / srcStd[2]));

    const srcData = sp.rawImageData.data;
    const outImg = this.colorMatchedSourceCtx.createImageData(w, h);
    const dstData = outImg.data;

    for (let i = 0; i < srcData.length; i += 4) {
      const r = srcData[i];
      const g = srcData[i + 1];
      const b = srcData[i + 2];

      const nr = (r - srcMean[0]) * scaleR + dstMean[0];
      const ng = (g - srcMean[1]) * scaleG + dstMean[1];
      const nb = (b - srcMean[2]) * scaleB + dstMean[2];

      dstData[i] = Math.max(0, Math.min(255, r * (1 - strength) + nr * strength));
      dstData[i + 1] = Math.max(0, Math.min(255, g * (1 - strength) + ng * strength));
      dstData[i + 2] = Math.max(0, Math.min(255, b * (1 - strength) + nb * strength));
      dstData[i + 3] = 255;
    }

    this.colorMatchedSourceCtx.putImageData(outImg, 0, 0);
    return this.colorMatchedSourceCanvas;
  }

  /**
   * Renderiza el fotograma completo con IA en el Canvas de salida:
   *  - modo "faceswap": reemplaza la cara del actor en el video con la de @image1 (deformando los 478 puntos: boca, ojos, cejas, giro 3D)
   *  - modo "motion": transfiere el movimiento corporal y facial completo del actor en el video al personaje recortado de @image1
   *  - modo "liveportrait": anima el retrato de @image1 con todas las expresiones y giros de cabeza del video
   */
  renderFrame({ dstCtx, width, height, videoEl, mode = "faceswap", timeSec = 0 }) {
    dstCtx.save();
    dstCtx.clearRect(0, 0, width, height);

    const hasVideo = videoEl && videoEl.readyState >= 2;
    const sp = this.sourceProfile;

    // Si aún no hay video o imagen, mostrar lo que haya disponible
    if (!hasVideo && !sp) {
      dstCtx.fillStyle = "#06080f";
      dstCtx.fillRect(0, 0, width, height);
      dstCtx.restore();
      return { status: "idle" };
    }

    if (hasVideo && !sp) {
      this.drawCover(dstCtx, videoEl, videoEl.videoWidth, videoEl.videoHeight, width, height);
      dstCtx.restore();
      return { status: "video-only" };
    }

    if (!hasVideo && sp) {
      this.drawCover(dstCtx, sp.rawCanvas, sp.width, sp.height, width, height);
      dstCtx.restore();
      return { status: "image-only" };
    }

    // Ajustar dimensiones de los lienzos de trabajo
    if (this.workCanvas.width !== width || this.workCanvas.height !== height) {
      this.workCanvas.width = width;
      this.workCanvas.height = height;
      this.maskCanvas.width = width;
      this.maskCanvas.height = height;
    }

    // Dibujar primero el fotograma del video con ajuste proporcional completo
    const vidTransform = this.drawCover(
      dstCtx,
      videoEl,
      videoEl.videoWidth,
      videoEl.videoHeight,
      width,
      height
    );

    const { faceLandmarks, poseLandmarks } = this.detectVideoFrame(videoEl);

    // Convertir coordenadas normalizadas del video (0..1) al espacio de píxeles del canvas
    const mapNormToCanvas = (pt) => ({
      x: vidTransform.offsetX + pt.x * vidTransform.drawW,
      y: vidTransform.offsetY + pt.y * vidTransform.drawH,
      z: pt.z || 0,
    });

    // Determinar los 478 puntos objetivo en el fotograma actual
    let rawTargetLm = faceLandmarks;
    if (!rawTargetLm && poseLandmarks) {
      rawTargetLm = this.synthesizeFaceFromPoseHead(poseLandmarks, sp.landmarks);
    }

    // Suavizado temporal exponencial de los landmarks para movimiento fluido a 30fps
    if (rawTargetLm) {
      if (!this.smoothedTargetLandmarks || this.smoothedTargetLandmarks.length !== rawTargetLm.length) {
        this.smoothedTargetLandmarks = rawTargetLm.map((p) => ({ ...p }));
      } else {
        const alpha = 0.68;
        for (let i = 0; i < rawTargetLm.length; i++) {
          this.smoothedTargetLandmarks[i].x =
            this.smoothedTargetLandmarks[i].x * (1 - alpha) + rawTargetLm[i].x * alpha;
          this.smoothedTargetLandmarks[i].y =
            this.smoothedTargetLandmarks[i].y * (1 - alpha) + rawTargetLm[i].y * alpha;
          this.smoothedTargetLandmarks[i].z =
            (this.smoothedTargetLandmarks[i].z || 0) * (1 - alpha) + (rawTargetLm[i].z || 0) * alpha;
        }
      }
    }

    // =========================================================================
    // MODO 3: ANIMAR ROSTRO (LivePortrait 3D — Retrato cobra vida con los gestos del video)
    // =========================================================================
    if (mode === "liveportrait") {
      this.renderLivePortraitMode(dstCtx, width, height, this.smoothedTargetLandmarks, timeSec);
      dstCtx.restore();
      return { status: "liveportrait", hasFace: Boolean(rawTargetLm) };
    }

    // =========================================================================
    // MODO 2: MOTION TRANSFER (Personaje completo de @image1 imita cuerpo + cara del video)
    // =========================================================================
    if (mode === "motion" || mode === "motion-control") {
      this.renderFullCharacterMotionTransfer(
        dstCtx,
        width,
        height,
        this.smoothedTargetLandmarks,
        poseLandmarks,
        mapNormToCanvas,
        timeSec
      );
      dstCtx.restore();
      return { status: "motion", hasFace: Boolean(rawTargetLm), hasPose: Boolean(poseLandmarks) };
    }

    // =========================================================================
    // MODO 1: CAMBIAR CARA EN EL VIDEO (Face Swap 478 puntos + Expresiones + Escena)
    // =========================================================================
    if (this.smoothedTargetLandmarks) {
      const targetCanvasPts = this.smoothedTargetLandmarks.map(mapNormToCanvas);
      const targetColor = computeFaceColorStats(
        dstCtx,
        this.smoothedTargetLandmarks.map((p) => ({
          x: (vidTransform.offsetX + p.x * vidTransform.drawW) / width,
          y: (vidTransform.offsetY + p.y * vidTransform.drawH) / height,
        })),
        width,
        height
      );
      const harmonizedSource = this.updateColorMatchedSource(targetColor, 0.62);

      // 1. Deformar cada triángulo de la cara de @image1 a la cara del actor en @video1
      this.workCtx.clearRect(0, 0, width, height);
      for (const [i0, i1, i2] of sp.meshTriangles) {
        const s0 = { x: sp.landmarks[i0].x * sp.width, y: sp.landmarks[i0].y * sp.height };
        const s1 = { x: sp.landmarks[i1].x * sp.width, y: sp.landmarks[i1].y * sp.height };
        const s2 = { x: sp.landmarks[i2].x * sp.width, y: sp.landmarks[i2].y * sp.height };
        const d0 = targetCanvasPts[i0];
        const d1 = targetCanvasPts[i1];
        const d2 = targetCanvasPts[i2];
        if (!d0 || !d1 || !d2) continue;
        warpTriangle(this.workCtx, harmonizedSource, s0, s1, s2, d0, d1, d2);
      }

      // 2. Crear máscara difuminada (Feathered Face Oval) para fusión invisible con piel y cabello
      const noseCenter = targetCanvasPts[1] || targetCanvasPts[4];
      this.maskCtx.clearRect(0, 0, width, height);
      this.maskCtx.save();
      this.maskCtx.filter = "blur(9px)";
      this.maskCtx.fillStyle = "#ffffff";
      this.maskCtx.beginPath();
      FACE_OVAL_INDICES.forEach((lmIdx, idx) => {
        const pt = targetCanvasPts[lmIdx];
        // Contraer ligeramente (5%) hacia el centro nasal para respetar la línea de cabello del actor
        const shrink = idx < 10 || idx > 28 ? 0.92 : 0.96;
        const x = noseCenter.x + (pt.x - noseCenter.x) * shrink;
        const y = noseCenter.y + (pt.y - noseCenter.y) * shrink;
        if (idx === 0) this.maskCtx.moveTo(x, y);
        else this.maskCtx.lineTo(x, y);
      });
      this.maskCtx.closePath();
      this.maskCtx.fill();
      this.maskCtx.restore();

      // 3. Aplicar máscara al rostro deformado y componer sobre el actor del video
      this.workCtx.save();
      this.workCtx.globalCompositeOperation = "destination-in";
      this.workCtx.drawImage(this.maskCanvas, 0, 0);
      this.workCtx.restore();

      dstCtx.drawImage(this.workCanvas, 0, 0);
      dstCtx.restore();
      return { status: "faceswap-active", hasFace: true };
    }

    // Si el fotograma actual del video no tiene ningún rostro humano (ej. plano de un objeto o carta),
    // integramos al personaje recortado (sin fondo) animado en la escena para que nunca quede vacío
    this.renderFullCharacterMotionTransfer(
      dstCtx,
      width,
      height,
      null,
      poseLandmarks,
      mapNormToCanvas,
      timeSec
    );
    dstCtx.restore();
    return { status: "scene-character-fallback", hasFace: false };
  }

  /**
   * Modo LivePortrait: Anima la foto del usuario (@image1) deformando sus 478 puntos faciales
   * (boca, ojos, cejas, rotación 3D de cabeza) según los gestos del video (@video1).
   */
  renderLivePortraitMode(dstCtx, width, height, targetLm, timeSec) {
    const sp = this.sourceProfile;
    // Fondo estilizado desenfocado de la propia foto
    dstCtx.save();
    dstCtx.filter = "blur(22px) brightness(0.45)";
    this.drawCover(dstCtx, sp.rawCanvas, sp.width, sp.height, width, height);
    dstCtx.restore();

    // Calcular transformación base de la foto centrada en el lienzo
    const fit = this.getContainTransform(sp.width, sp.height, width, height);
    dstCtx.drawImage(sp.cutoutCanvas, fit.offsetX, fit.offsetY, fit.drawW, fit.drawH);

    // Calcular deformación delta de los 478 puntos desde el video (o animación natural si el frame no tiene cara)
    const srcNose = sp.landmarks[1];
    const animatedPts = sp.landmarks.map((sPt, idx) => {
      let dx = 0;
      let dy = 0;
      if (targetLm && targetLm[idx] && targetLm[1]) {
        const tNose = targetLm[1];
        const tPt = targetLm[idx];
        // Transferir desplazamiento relativo (expresión + rotación de cabeza)
        dx = (tPt.x - tNose.x) - (sPt.x - srcNose.x);
        dy = (tPt.y - tNose.y) - (sPt.y - srcNose.y);
      } else {
        // Respiración y articulación viva guiada por el tiempo del video
        const headTurn = Math.sin(timeSec * 2.2) * 0.018;
        const nod = Math.cos(timeSec * 1.7) * 0.012;
        dx = headTurn * (1 - Math.abs(sPt.y - srcNose.y));
        dy = nod;
        // Movimiento de labios al hablar
        if (idx === 14 || idx === 17 || idx === 87 || idx === 317) {
          dy += Math.abs(Math.sin(timeSec * 6.5)) * 0.022;
        }
      }
      return {
        x: fit.offsetX + (sPt.x + dx * 0.85) * fit.drawW,
        y: fit.offsetY + (sPt.y + dy * 0.85) * fit.drawH,
      };
    });

    this.workCtx.clearRect(0, 0, width, height);
    for (const [i0, i1, i2] of sp.meshTriangles) {
      const s0 = { x: sp.landmarks[i0].x * sp.width, y: sp.landmarks[i0].y * sp.height };
      const s1 = { x: sp.landmarks[i1].x * sp.width, y: sp.landmarks[i1].y * sp.height };
      const s2 = { x: sp.landmarks[i2].x * sp.width, y: sp.landmarks[i2].y * sp.height };
      warpTriangle(
        this.workCtx,
        sp.rawCanvas,
        s0,
        s1,
        s2,
        animatedPts[i0],
        animatedPts[i1],
        animatedPts[i2]
      );
    }

    // Fundir la malla facial animada sobre el retrato
    this.maskCtx.clearRect(0, 0, width, height);
    this.maskCtx.save();
    this.maskCtx.filter = "blur(10px)";
    this.maskCtx.fillStyle = "#fff";
    this.maskCtx.beginPath();
    FACE_OVAL_INDICES.forEach((lmIdx, i) => {
      const p = animatedPts[lmIdx];
      if (i === 0) this.maskCtx.moveTo(p.x, p.y);
      else this.maskCtx.lineTo(p.x, p.y);
    });
    this.maskCtx.closePath();
    this.maskCtx.fill();
    this.maskCtx.restore();

    this.workCtx.save();
    this.workCtx.globalCompositeOperation = "destination-in";
    this.workCtx.drawImage(this.maskCanvas, 0, 0);
    this.workCtx.restore();

    dstCtx.drawImage(this.workCanvas, 0, 0);
  }

  /**
   * Modo Motion Transfer: El personaje recortado de @image1 (sin fondo cuadrado) se deforma
   * siguiendo la pose corporal (hombros, torso, inclinación de cabeza) y las expresiones faciales del video.
   */
  renderFullCharacterMotionTransfer(dstCtx, width, height, targetFaceLm, poseLm, mapNormToCanvas, timeSec) {
    const sp = this.sourceProfile;

    // 1. Determinar posición, escala e inclinación del actor en el video
    let anchorX = width * 0.5 + Math.sin(timeSec * 1.8) * (width * 0.04);
    let anchorY = height * 0.56 + Math.cos(timeSec * 2.4) * (height * 0.02);
    let bodyAngle = Math.sin(timeSec * 1.5) * 0.04;
    let charHeight = height * 0.78;

    if (poseLm && poseLm[11] && poseLm[12]) {
      const lSh = mapNormToCanvas(poseLm[11]);
      const rSh = mapNormToCanvas(poseLm[12]);
      const nose = poseLm[0] ? mapNormToCanvas(poseLm[0]) : { x: (lSh.x + rSh.x) / 2, y: lSh.y - height * 0.15 };
      const shMidX = (lSh.x + rSh.x) / 2;
      const shMidY = (lSh.y + rSh.y) / 2;
      const shWidth = Math.hypot(rSh.x - lSh.x, rSh.y - lSh.y);

      anchorX = (shMidX + nose.x) / 2;
      anchorY = shMidY + height * 0.1;
      bodyAngle = Math.atan2(lSh.y - rSh.y, lSh.x - rSh.x) * 0.45;
      charHeight = Math.max(height * 0.52, Math.min(height * 0.95, shWidth * 2.6));

      // Ocultar suavemente la silueta del actor original detrás del nuevo personaje (Inpainting ambiental)
      dstCtx.save();
      dstCtx.beginPath();
      dstCtx.ellipse(anchorX, anchorY, shWidth * 0.95, charHeight * 0.55, bodyAngle, 0, Math.PI * 2);
      dstCtx.clip();
      dstCtx.filter = "blur(26px)";
      dstCtx.drawImage(dstCtx.canvas, 0, 0);
      dstCtx.restore();
    } else if (targetFaceLm && targetFaceLm[1]) {
      const nose = mapNormToCanvas(targetFaceLm[1]);
      anchorX = nose.x;
      anchorY = nose.y + height * 0.22;
    }

    const aspect = sp.width / sp.height;
    const drawH = charHeight;
    const drawW = drawH * aspect;

    // 2. Dibujar el personaje recortado (sin fondo cuadrado) con la inclinación y movimiento del cuerpo
    dstCtx.save();
    dstCtx.translate(anchorX, anchorY);
    dstCtx.rotate(bodyAngle);
    dstCtx.drawImage(sp.cutoutCanvas, -drawW / 2, -drawH / 2, drawW, drawH);

    // 3. Animar además los 478 puntos de la cara del personaje con las expresiones del video
    const srcNose = sp.landmarks[1];
    this.workCtx.clearRect(0, 0, width, height);
    const localFacePts = sp.landmarks.map((sPt, idx) => {
      let dx = 0;
      let dy = 0;
      if (targetFaceLm && targetFaceLm[idx] && targetFaceLm[1]) {
        dx = (targetFaceLm[idx].x - targetFaceLm[1].x) - (sPt.x - srcNose.x);
        dy = (targetFaceLm[idx].y - targetFaceLm[1].y) - (sPt.y - srcNose.y);
      } else if (idx === 14 || idx === 17) {
        dy = Math.abs(Math.sin(timeSec * 6.0)) * 0.018;
      }
      return {
        x: (sPt.x + dx * 0.8) * drawW,
        y: (sPt.y + dy * 0.8) * drawH,
      };
    });

    for (const [i0, i1, i2] of sp.meshTriangles) {
      const s0 = { x: sp.landmarks[i0].x * sp.width, y: sp.landmarks[i0].y * sp.height };
      const s1 = { x: sp.landmarks[i1].x * sp.width, y: sp.landmarks[i1].y * sp.height };
      const s2 = { x: sp.landmarks[i2].x * sp.width, y: sp.landmarks[i2].y * sp.height };
      warpTriangle(
        this.workCtx,
        sp.cutoutCanvas,
        s0,
        s1,
        s2,
        localFacePts[i0],
        localFacePts[i1],
        localFacePts[i2]
      );
    }

    this.maskCtx.clearRect(0, 0, width, height);
    this.maskCtx.save();
    this.maskCtx.filter = "blur(7px)";
    this.maskCtx.fillStyle = "#fff";
    this.maskCtx.beginPath();
    FACE_OVAL_INDICES.forEach((lmIdx, i) => {
      const p = localFacePts[lmIdx];
      if (i === 0) this.maskCtx.moveTo(p.x, p.y);
      else this.maskCtx.lineTo(p.x, p.y);
    });
    this.maskCtx.closePath();
    this.maskCtx.fill();
    this.maskCtx.restore();

    this.workCtx.save();
    this.workCtx.globalCompositeOperation = "destination-in";
    this.workCtx.drawImage(this.maskCanvas, 0, 0);
    this.workCtx.restore();

    dstCtx.drawImage(this.workCtx, 0, 0, drawW, drawH, -drawW / 2, -drawH / 2, drawW, drawH);
    dstCtx.restore();
  }

  drawCover(ctx2d, media, mediaW, mediaH, boxW, boxH) {
    const mw = mediaW || 640;
    const mh = mediaH || 360;
    const scale = Math.max(boxW / mw, boxH / mh);
    const drawW = mw * scale;
    const drawH = mh * scale;
    const offsetX = (boxW - drawW) / 2;
    const offsetY = (boxH - drawH) / 2;
    ctx2d.drawImage(media, offsetX, offsetY, drawW, drawH);
    return { scale, drawW, drawH, offsetX, offsetY };
  }

  getContainTransform(mediaW, mediaH, boxW, boxH) {
    const scale = Math.min(boxW / mediaW, boxH / mediaH) * 0.94;
    const drawW = mediaW * scale;
    const drawH = mediaH * scale;
    const offsetX = (boxW - drawW) / 2;
    const offsetY = (boxH - drawH) / 2;
    return { scale, drawW, drawH, offsetX, offsetY };
  }
}
