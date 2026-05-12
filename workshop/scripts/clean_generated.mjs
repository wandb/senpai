#!/usr/bin/env node
import { rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));

const generatedPaths = [
  "autoresearch-workshop.pdf",
  "autoresearch-workshop-pages",
  "autoresearch-workshop-pages2",
  "pdf-inspection",
  "review",
  "dist",
  ".slidev",
  ".vite",
];

for (const relativePath of generatedPaths) {
  const path = join(root, relativePath);
  await rm(path, { recursive: true, force: true });
  console.log(`removed ${relativePath}`);
}

console.log("Generated workshop outputs cleaned. Source files and package-lock.json were preserved.");
