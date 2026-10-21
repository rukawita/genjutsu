import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI } from "@google/genai";
import { runFreeEngine } from "./lib/free-engines.mjs";

// Trabajos de generación gratuita en curso (IA real en GPU de Hugging Face)
const jobs = new Map();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3456;

const UPLOADS_DIR = path.join(__dirname, "uploads");
const OUTPUTS_DIR = path.join(__dirname, "outputs");
fs.mkdirSync(UPLOADS_DIR, { recursive: true });
fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".task": "application/octet-stream",
  ".tflite": "application/octet-stream",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".mov": "video/quicktime",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf8");
        resolve(raw ? JSON.parse(raw) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}

async function waitForFileActive(ai, fileName, maxWaitMs = 180000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    const info = await ai.files.get({ name: fileName });
    const state = info.state || info.status;
    if (!state || state === "ACTIVE") {
      return info;
    }
    if (state === "FAILED") {
      throw new Error(`La subida del archivo ${fileName} falló en Files API.`);
    }
    await new Promise((r) => setTimeout(r, 2500));
  }
  throw new Error(`Tiempo de espera agotado activando ${fileName} en Files API.`);
}

async function uploadLocalAssetToGemini(ai, localPath, mimeType, displayName) {
  const uploaded = await ai.files.upload({
    file: localPath,
    config: {
      mimeType,
      displayName: displayName || path.basename(localPath),
    },
  });
  const active = await waitForFileActive(ai, uploaded.name);
  const cleanUri = (active.uri || uploaded.uri || "").split("?")[0];
  return {
    name: active.name || uploaded.name,
    uri: cleanUri,
    mime_type: active.mimeType || uploaded.mimeType || mimeType,
  };
}

/**
 * Traduce el estado de Genjutsu (Motion Transfer / Objects Swap + @image1 / @video1 + scene_mode)
 * al formato exacto de Interactions API de Gemini Omni 1.1 Flash (`gemini-omni-1.1-flash`).
 */
function buildOmniFlashPromptAndParts({
  mode,
  sceneMode,
  rawPrompt,
  uploadedVideo,
  uploadedImages,
  uploadedLocation,
}) {
  const inputParts = [];
  const imageRefTags = [];

  // 1. Adjuntar imágenes de referencia (@image1, @image2...)
  uploadedImages.forEach((img, idx) => {
    inputParts.push({
      type: "image",
      uri: img.uri,
      mime_type: img.mime_type,
    });
    imageRefTags.push(`<IMAGE_REF_${idx}>@Image${idx + 1}`);
  });

  // 2. Si hay imagen de fondo manual (scene_mode === "location"), adjuntarla como última imagen de referencia
  let locationRefIdx = null;
  if (sceneMode === "location" && uploadedLocation) {
    locationRefIdx = uploadedImages.length;
    inputParts.push({
      type: "image",
      uri: uploadedLocation.uri,
      mime_type: uploadedLocation.mime_type,
    });
    imageRefTags.push(`<IMAGE_REF_${locationRefIdx}>@Image${locationRefIdx + 1}`);
  }

  // 3. Adjuntar el video (como Source <VIDEO_0> en Objects Swap o Reference <VIDEO_REF_0> en Motion Transfer)
  if (uploadedVideo) {
    inputParts.push({
      type: "video",
      uri: uploadedVideo.uri,
      mime_type: uploadedVideo.mime_type,
    });
  }

  // 4. Traducir menciones @image1, @video1, @location en el prompt del usuario
  let translatedPrompt = (rawPrompt || "").trim();
  uploadedImages.forEach((_, idx) => {
    const regex = new RegExp(`@image${idx + 1}\\b`, "gi");
    translatedPrompt = translatedPrompt.replace(regex, `<IMAGE_REF_${idx}>`);
  });
  if (locationRefIdx !== null) {
    translatedPrompt = translatedPrompt.replace(
      /@location\b/gi,
      `<IMAGE_REF_${locationRefIdx}>`,
    );
  }

  let prefixDeclarations = "";
  let suffixGuidance = "";

  if (mode === "replace-objects") {
    // Modo Objects Swap -> Edición directa sobre <VIDEO_0>
    translatedPrompt = translatedPrompt.replace(/@video1\b/gi, "<VIDEO_0>");
    const sourcesBlock = uploadedVideo ? "[# Sources <VIDEO_0>@Video1]" : "";
    const refsBlock =
      imageRefTags.length > 0 ? `[# References ${imageRefTags.join(" ")}]` : "";
    prefixDeclarations = [sourcesBlock, refsBlock].filter(Boolean).join(" ");
    if (!translatedPrompt) {
      translatedPrompt =
        imageRefTags.length > 0
          ? "Replace the main subject in <VIDEO_0> with <IMAGE_REF_0>."
          : "Enhance and restyle the video.";
    }
    if (!/keep everything else the same/i.test(translatedPrompt)) {
      suffixGuidance = "Keep everything else the same.";
    }
  } else {
    // Modo Motion Transfer -> Transferencia de movimiento desde <VIDEO_REF_0> hacia <IMAGE_REF_0>
    translatedPrompt = translatedPrompt.replace(/@video1\b/gi, "<VIDEO_REF_0>");
    const refItems = [...imageRefTags];
    if (uploadedVideo) {
      refItems.push("<VIDEO_REF_0>@Video1");
    }
    prefixDeclarations =
      refItems.length > 0 ? `[# References ${refItems.join(" ")}]` : "";

    if (!translatedPrompt) {
      translatedPrompt =
        uploadedImages.length > 0 && uploadedVideo
          ? "The character in <IMAGE_REF_0> performs the exact motion and camera movement from <VIDEO_REF_0> in a single continuous unbroken shot."
          : "A cinematic scene in a single continuous shot.";
    }

    // Reglas de Background Control (scene_mode extraído por REA de 2feeef17-f0uOQAEmXoMSrOm4.js)
    if (sceneMode === "video" && uploadedVideo) {
      suffixGuidance =
        "Preserve the environment and background setting from <VIDEO_REF_0>. Use the given image(s) as character references and the video for motion.";
    } else if (sceneMode === "character" && uploadedImages.length > 0) {
      suffixGuidance =
        "Use the background and environment shown in <IMAGE_REF_0> while transferring the motion from <VIDEO_REF_0>.";
    } else if (sceneMode === "location" && locationRefIdx !== null) {
      suffixGuidance = `Use <IMAGE_REF_${locationRefIdx}> as the background environment location, <IMAGE_REF_0> as the subject, and transfer the motion from <VIDEO_REF_0>.`;
    } else {
      suffixGuidance =
        "Use the given image(s) as references for video generation and transfer the motion and camera work from the reference video.";
    }
  }

  const finalPromptText = [prefixDeclarations, translatedPrompt, suffixGuidance]
    .filter(Boolean)
    .join(" ")
    .trim();

  inputParts.push({
    type: "text",
    text: finalPromptText,
  });

  return { finalPromptText, inputParts };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (url.pathname === "/api/health" && req.method === "GET") {
    return sendJson(res, 200, {
      ok: true,
      service: "open-genjutsu-studio",
      hasEnvApiKey: Boolean(process.env.GEMINI_API_KEY),
    });
  }

  // Subida local de archivos (Video / Imagen) para preparar la sesión
  if (url.pathname === "/api/upload-local" && req.method === "POST") {
    try {
      const body = await readJsonBody(req);
      const { filename, mimeType, base64Data } = body;
      if (!filename || !base64Data) {
        return sendJson(res, 400, { ok: false, error: "Faltan datos del archivo." });
      }
      const safeName = `${Date.now()}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
      const filePath = path.join(UPLOADS_DIR, safeName);
      const buffer = Buffer.from(base64Data, "base64");
      fs.writeFileSync(filePath, buffer);
      return sendJson(res, 200, {
        ok: true,
        localPath: filePath,
        publicUrl: `/uploads/${safeName}`,
        filename: safeName,
        mimeType: mimeType || "application/octet-stream",
        bytes: buffer.byteLength,
      });
    } catch (err) {
      return sendJson(res, 500, { ok: false, error: String(err.message || err) });
    }
  }

  // Generación GRATUITA con IA real (Face Swap / Motion Transfer / LivePortrait)
  if (url.pathname === "/api/generate-free" && req.method === "POST") {
    try {
      const body = await readJsonBody(req);
      const { mode = "faceswap", imagePath, videoPath, prompt = "", aspectRatio = "9:16", hfToken = "" } = body;
      const inUploads = (p) => p && path.resolve(p).startsWith(UPLOADS_DIR) && fs.existsSync(p);
      if (!inUploads(imagePath)) return sendJson(res, 400, { ok: false, error: "Sube primero la foto del personaje (Paso 2)." });
      if (!inUploads(videoPath)) return sendJson(res, 400, { ok: false, error: "Sube primero el video base (Paso 1)." });

      const id = `job_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      const job = { id, status: "running", logs: [], message: "Iniciando...", videoUrl: null, error: null, engine: null };
      jobs.set(id, job);

      runFreeEngine({
        mode,
        imagePath,
        videoPath,
        prompt: prompt.replace(/@\w+/g, "").trim(),
        aspectRatio,
        hfToken,
        onLog: (m) => {
          job.message = m;
          job.logs.push(m);
          console.log(`[${id}] ${m}`);
        },
      })
        .then(({ buffer, engine }) => {
          const outFilename = `genjutsu_${mode}_${Date.now()}.mp4`;
          fs.writeFileSync(path.join(OUTPUTS_DIR, outFilename), buffer);
          Object.assign(job, { status: "done", engine, videoUrl: `/outputs/${outFilename}`, message: `Listo con ${engine}` });
        })
        .catch((err) => {
          Object.assign(job, { status: "error", error: String(err.message || err) });
        });

      return sendJson(res, 200, { ok: true, jobId: id });
    } catch (err) {
      return sendJson(res, 500, { ok: false, error: String(err.message || err) });
    }
  }

  if (url.pathname.startsWith("/api/jobs/") && req.method === "GET") {
    const job = jobs.get(url.pathname.split("/").pop());
    if (!job) return sendJson(res, 404, { ok: false, error: "Trabajo no encontrado." });
    return sendJson(res, 200, { ok: true, ...job, logs: undefined });
  }

  // Guardar en disco (outputs/) los videos generados gratis por el motor integrado
  if (url.pathname === "/api/save-rendered-video" && req.method === "POST") {
    try {
      const body = await readJsonBody(req);
      const { base64Data, ext = "webm" } = body;
      if (!base64Data) {
        return sendJson(res, 400, { ok: false, error: "No se recibieron datos de video." });
      }
      const cleanExt = ext === "mp4" ? "mp4" : "webm";
      const outFilename = `genjutsu_free_${Date.now()}.${cleanExt}`;
      const outPath = path.join(OUTPUTS_DIR, outFilename);
      const buffer = Buffer.from(base64Data, "base64");
      fs.writeFileSync(outPath, buffer);
      return sendJson(res, 200, {
        ok: true,
        videoUrl: `/outputs/${outFilename}`,
        outputPath: outPath,
        filename: outFilename,
        bytes: buffer.byteLength,
      });
    } catch (err) {
      return sendJson(res, 500, { ok: false, error: String(err.message || err) });
    }
  }

  // Generación real con Gemini Omni 1.1 Flash (Interactions API) o Veo 3.1
  if (url.pathname === "/api/generate-gemini" && req.method === "POST") {
    try {
      const body = await readJsonBody(req);
      const apiKey = (body.apiKey || process.env.GEMINI_API_KEY || "").trim();
      if (!apiKey) {
        return sendJson(res, 400, {
          ok: false,
          error:
            "Ingresa tu GEMINI_API_KEY de Google AI Studio en el panel lateral (o defínela en la variable de entorno GEMINI_API_KEY).",
        });
      }

      const {
        engine = "gemini-omni-1.1-flash",
        mode = "motion-control",
        sceneMode = "off",
        prompt = "",
        resolution = "720p",
        aspectRatio = "16:9",
        duration = null,
        sourceVideo = null,
        images = [],
        locationMedia = null,
        previousInteractionId = null,
      } = body;

      const normalizedAspect =
        aspectRatio === "9:16" ? "9:16" : "16:9";
      const normalizedResolution = ["360p", "720p", "1080p", "4k"].includes(
        resolution,
      )
        ? resolution
        : "720p";

      const ai = new GoogleGenAI({
        vertexai: false,
        apiKey,
        httpOptions: { timeout: 600000 },
      });

      // OPCIÓN A: Gemini Omni 1.1 Flash via Interactions API (El motor multimodal exacto de Higgsfield)
      if (engine === "gemini-omni-1.1-flash") {
        let uploadedVideo = null;
        if (sourceVideo?.localPath && fs.existsSync(sourceVideo.localPath)) {
          uploadedVideo = await uploadLocalAssetToGemini(
            ai,
            sourceVideo.localPath,
            sourceVideo.mimeType || "video/mp4",
            sourceVideo.name,
          );
        }

        const uploadedImages = [];
        for (const img of images) {
          if (img?.localPath && fs.existsSync(img.localPath)) {
            const up = await uploadLocalAssetToGemini(
              ai,
              img.localPath,
              img.mimeType || "image/png",
              img.name,
            );
            uploadedImages.push(up);
          }
        }

        let uploadedLocation = null;
        if (
          sceneMode === "location" &&
          locationMedia?.localPath &&
          fs.existsSync(locationMedia.localPath)
        ) {
          uploadedLocation = await uploadLocalAssetToGemini(
            ai,
            locationMedia.localPath,
            locationMedia.mimeType || "image/png",
            locationMedia.name,
          );
        }

        const { finalPromptText, inputParts } = buildOmniFlashPromptAndParts({
          mode,
          sceneMode,
          rawPrompt: prompt,
          uploadedVideo,
          uploadedImages,
          uploadedLocation,
        });

        const videoConfig = {
          type: "video",
          delivery: "uri",
          aspect_ratio: normalizedAspect,
          resolution: normalizedResolution,
          ...(duration ? { duration: `${duration}s` } : {}),
        };

        const createParams = {
          model: "gemini-omni-1.1-flash",
          input: inputParts,
          response_format: videoConfig,
          ...(previousInteractionId
            ? { previous_interaction_id: previousInteractionId }
            : {}),
        };

        const interaction = await ai.interactions.create(createParams);
        const outputVideo = interaction.output_video;
        if (!outputVideo || !outputVideo.uri) {
          return sendJson(res, 502, {
            ok: false,
            error:
              "La respuesta de Gemini Omni 1.1 Flash no devolvió un URI de video. Nota: Si subiste un video de entrada, verifica que tu región admita edición de video (no disponible en EEE/Reino Unido/Suiza).",
            interactionId: interaction.id || null,
          });
        }

        const outFilename = `genjutsu_omni_${Date.now()}.mp4`;
        const outPath = path.join(OUTPUTS_DIR, outFilename);
        await ai.files.download({
          file: outputVideo.uri || outputVideo,
          downloadPath: outPath,
        });

        return sendJson(res, 200, {
          ok: true,
          engine: "gemini-omni-1.1-flash",
          interactionId: interaction.id || null,
          compiledPrompt: finalPromptText,
          videoUrl: `/outputs/${outFilename}`,
          outputPath: outPath,
        });
      }

      // OPCIÓN B: Veo 3.1 (generateVideos API)
      const veoModel =
        engine === "veo-3.1-generate-preview"
          ? "veo-3.1-generate-preview"
          : "veo-3.1-fast-generate-preview";

      const sourcePayload = {
        prompt:
          prompt.trim() ||
          "Cinematic high-definition video with realistic motion and lighting",
      };

      if (images[0]?.localPath && fs.existsSync(images[0].localPath)) {
        const imgBytes = fs.readFileSync(images[0].localPath).toString("base64");
        sourcePayload.image = {
          imageBytes: imgBytes,
          mimeType: images[0].mimeType || "image/png",
        };
      }

      let operation = await ai.models.generateVideos({
        model: veoModel,
        source: sourcePayload,
        config: {
          numberOfVideos: 1,
          aspectRatio: normalizedAspect,
        },
      });

      while (!operation.done) {
        await new Promise((r) => setTimeout(r, 8000));
        operation = await ai.operations.get({ operation });
      }

      const generatedVideos = operation.response?.generatedVideos;
      if (!generatedVideos || generatedVideos.length === 0) {
        return sendJson(res, 502, {
          ok: false,
          error: "Veo no devolvió videos generados en la operación.",
        });
      }

      const outFilename = `genjutsu_veo_${Date.now()}.mp4`;
      const outPath = path.join(OUTPUTS_DIR, outFilename);
      await ai.files.download({
        file: generatedVideos[0].video || generatedVideos[0],
        downloadPath: outPath,
      });

      return sendJson(res, 200, {
        ok: true,
        engine: veoModel,
        compiledPrompt: sourcePayload.prompt,
        videoUrl: `/outputs/${outFilename}`,
        outputPath: outPath,
      });
    } catch (err) {
      return sendJson(res, 500, {
        ok: false,
        error: String(err.message || err),
      });
    }
  }

  // Archivos estáticos (index.html, uploads/, outputs/)
  let filePath = path.join(
    __dirname,
    url.pathname === "/" ? "index.html" : decodeURIComponent(url.pathname),
  );
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
    });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404);
    res.end("Not found");
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Open Genjutsu Studio running at http://127.0.0.1:${PORT}`);
});
