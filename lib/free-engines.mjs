// Motores GRATUITOS reales (GPU de Hugging Face Spaces) para replicar Genjutsu:
//  - "faceswap": intercambia la cara del video por la de tu foto, conservando movimientos y escena.
//  - "motion":   Wan 2.2 Animate — tu personaje copia los movimientos del video (Motion Transfer real).
//  - "liveportrait": anima el rostro de tu foto con las expresiones/movimientos de cabeza del video.
// Cada modo tiene una cadena de respaldo: si un Space está caído o sin cupo, prueba el siguiente.
import fs from "node:fs";
import path from "node:path";
import { Client, handle_file } from "@gradio/client";

const ENV_HF_TOKEN = process.env.HF_TOKEN || undefined;

const MIME = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".mov": "video/quicktime",
};

function fileBlob(p) {
  const type = MIME[path.extname(p).toLowerCase()] || "application/octet-stream";
  return new Blob([fs.readFileSync(p)], { type });
}

function videoInput(p) {
  return { video: handle_file(fileBlob(p)), subtitles: null };
}

const ENGINES = {
  faceswap: [
    {
      space: "tonyassi/video-face-swap",
      endpoint: "/generate",
      label: "Video Face Swap (tonyassi)",
      args: ({ imagePath, videoPath }) => ({
        input_image: handle_file(fileBlob(imagePath)),
        input_video: videoInput(videoPath),
        gender: "all",
      }),
    },
    {
      space: "MarkoVidrih/video-face-swap",
      endpoint: "/predict",
      label: "Video Face Swap (MarkoVidrih)",
      args: ({ imagePath, videoPath }) => [handle_file(fileBlob(imagePath)), videoInput(videoPath)],
    },
    {
      space: "Bobeabo/video-face-swap",
      endpoint: "/predict",
      label: "Video Face Swap (Bobeabo)",
      args: ({ imagePath, videoPath }) => [handle_file(fileBlob(imagePath)), videoInput(videoPath)],
    },
  ],
  motion: [
    {
      space: "hugging-apps/wan2-2-animate-2-14b",
      endpoint: "/animate",
      label: "Wan 2.2 Animate 14B",
      args: ({ imagePath, videoPath, prompt, aspectRatio }) => ({
        image: handle_file(fileBlob(imagePath)),
        driving_video: videoInput(videoPath),
        prompt: prompt || "a person",
        max_seconds: 3,
        height: aspectRatio === "16:9" ? 384 : 480,
        width: aspectRatio === "16:9" ? 640 : 384,
        num_inference_steps: 6,
        guidance_scale: 1,
        sample_shift: 5,
        seed: 0,
      }),
    },
  ],
  liveportrait: [
    {
      space: "KlingTeam/LivePortrait",
      endpoint: "/gpu_wrapped_execute_video",
      label: "LivePortrait (KlingTeam)",
      args: ({ imagePath, videoPath }) => [handle_file(fileBlob(imagePath)), videoInput(videoPath), true, true, true],
    },
    {
      space: "Han-123/LivePortrait",
      endpoint: "/gpu_wrapped_execute_video",
      label: "LivePortrait (Han-123)",
      args: ({ imagePath, videoPath }) => [handle_file(fileBlob(imagePath)), videoInput(videoPath), true, true, true],
    },
  ],
};

// Si el motor principal de un modo falla, también se prueban estos modos como último recurso.
const FALLBACK_MODES = {
  motion: ["faceswap"],
  faceswap: ["liveportrait"],
  liveportrait: ["faceswap"],
};

function findVideoUrl(data) {
  const stack = [data];
  while (stack.length) {
    const v = stack.shift();
    if (!v) continue;
    if (typeof v === "string" && /^https?:\/\//.test(v)) return v;
    if (typeof v === "object") {
      if (typeof v.url === "string" && /\.(mp4|webm|mov)(\?|$)|\/file=/.test(v.url)) return v.url;
      if (v.video) stack.push(v.video);
      if (Array.isArray(v)) stack.push(...v);
      else stack.push(...Object.values(v));
    }
  }
  return null;
}

function withTimeout(promise, ms, msg) {
  return Promise.race([
    promise,
    new Promise((_, rej) => setTimeout(() => rej(new Error(msg)), ms)),
  ]);
}

async function runOne(engine, input, onLog, HF_TOKEN) {
  onLog(`Conectando con ${engine.label}...`);
  const app = await withTimeout(
    Client.connect(engine.space, { events: ["data", "status"], ...(HF_TOKEN ? { hf_token: HF_TOKEN } : {}) }),
    45000,
    "No se pudo conectar (Space dormido o caído)",
  );

  const job = app.submit(engine.endpoint, engine.args(input));
  let finalData = null;
  const deadline = Date.now() + 15 * 60 * 1000;

  for await (const msg of job) {
    if (Date.now() > deadline) throw new Error("Tiempo de espera agotado (15 min)");
    if (msg.type === "status") {
      if (msg.stage === "error") throw new Error(msg.message || "El Space devolvió un error");
      if (msg.queue && typeof msg.position === "number") {
        onLog(`${engine.label}: en cola, posición ${msg.position + 1}`);
      } else if (msg.stage === "pending") {
        onLog(`${engine.label}: procesando con IA...`);
      } else if (msg.progress_data?.length) {
        const p = msg.progress_data[0];
        if (p.index != null && p.length) onLog(`${engine.label}: ${p.index}/${p.length} ${p.unit || ""}`);
      }
    }
    if (msg.type === "data") {
      finalData = msg.data;
      break;
    }
  }

  const url = findVideoUrl(finalData);
  if (!url) throw new Error("El Space no devolvió un video");
  onLog(`${engine.label}: descargando resultado...`);
  const res = await fetch(url, HF_TOKEN ? { headers: { Authorization: `Bearer ${HF_TOKEN}` } } : {});
  if (!res.ok) throw new Error(`No se pudo descargar el video (${res.status})`);
  return Buffer.from(await res.arrayBuffer());
}

export async function runFreeEngine({ mode = "faceswap", imagePath, videoPath, prompt, aspectRatio, hfToken, onLog = () => {} }) {
  const HF_TOKEN = (hfToken || ENV_HF_TOKEN || "").trim() || undefined;
  if (!imagePath || !fs.existsSync(imagePath)) throw new Error("Falta la imagen del personaje/cara.");
  if (!videoPath || !fs.existsSync(videoPath)) throw new Error("Falta el video base.");

  const modes = [mode, ...(FALLBACK_MODES[mode] || [])];
  const errors = [];
  for (const m of modes) {
    for (const engine of ENGINES[m] || []) {
      try {
        const buffer = await runOne(engine, { imagePath, videoPath, prompt, aspectRatio }, onLog, HF_TOKEN);
        return { buffer, engine: engine.label, mode: m };
      } catch (err) {
        const msg = String(err?.message || err);
        errors.push(`${engine.label}: ${msg}`);
        onLog(`${engine.label} no disponible (${msg}). Probando otro motor gratuito...`);
      }
    }
  }
  const quota = errors.some((e) => /quota|GPU|exceeded/i.test(e));
  throw new Error(
    (quota
      ? "Se agotó el cupo gratuito de GPU por ahora. Espera unos minutos o agrega un HF_TOKEN gratuito para más cupo.\n"
      : "Todos los motores gratuitos fallaron.\n") + errors.join("\n"),
  );
}
