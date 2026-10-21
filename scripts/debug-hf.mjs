import fs from "node:fs";
import { Client, handle_file } from "@gradio/client";
const blob = (p, t) => new Blob([fs.readFileSync(p)], { type: t });
const space = process.argv[2] || "KlingTeam/LivePortrait";
const app = await Client.connect(space, { events: ["data", "status"] });
let args;
if (space.includes("LivePortrait")) {
  args = [handle_file(blob("samples/s9.jpg", "image/jpeg")), { video: handle_file(blob("samples/d3.mp4", "video/mp4")), subtitles: null }, true, true, true];
  var ep = "/gpu_wrapped_execute_video";
} else {
  args = { image: handle_file(blob("samples/s9.jpg", "image/jpeg")), driving_video: { video: handle_file(blob("samples/d3.mp4", "video/mp4")), subtitles: null }, prompt: "a person", max_seconds: 2, height: 480, width: 384, num_inference_steps: 6, guidance_scale: 1, sample_shift: 5, seed: 0 };
  var ep = "/animate";
}
const job = app.submit(ep, args);
for await (const m of job) {
  console.log(JSON.stringify(m).slice(0, 600));
  if (m.type === "data") break;
}
process.exit(0);

