// Extrae el <script type="module"> de index.html y valida su sintaxis.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";

const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const m = html.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!m) throw new Error("No se encontró el script");
const tmp = path.join(os.tmpdir(), "gj_check.mjs");
fs.writeFileSync(tmp, m[1]);
execFileSync(process.execPath, ["--check", tmp], { stdio: "inherit" });

// Verifica que los IDs usados con getElementById existan en el HTML
const ids = [...m[1].matchAll(/getElementById\("([^"]+)"\)/g)].map((x) => x[1]);
const missing = ids.filter((id) => !html.includes(`id="${id}"`));
console.log(missing.length ? `IDs faltantes: ${missing.join(", ")}` : "JS OK · todos los IDs existen");
