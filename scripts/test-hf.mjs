// Prueba rápida de los motores gratuitos en Hugging Face Spaces.
// Uso: node scripts/test-hf.mjs faceswap|motion|liveportrait
import fs from "node:fs";
import path from "node:path";
import { runFreeEngine } from "../lib/free-engines.mjs";

const mode = process.argv[2] || "faceswap";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, "$1")), "..");
const image = path.join(root, "samples", "s9.jpg");
const video = path.join(root, "samples", "d3.mp4");

const t0 = Date.now();
const result = await runFreeEngine({
  mode,
  imagePath: image,
  videoPath: video,
  prompt: "a person",
  onLog: (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, m),
});
const out = path.join(root, "outputs", `test_${mode}_${Date.now()}.mp4`);
fs.writeFileSync(out, result.buffer);
console.log("OK", result.engine, out, result.buffer.byteLength);
process.exit(0);
