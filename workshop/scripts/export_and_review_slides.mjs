#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-chromium";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const slidesPath = join(root, "slides.md");
const pdfPath = join(root, "autoresearch-workshop.pdf");
const reviewDir = join(root, "review", "autoresearch-workshop");
const screenshotsDir = join(reviewDir, "slides");
const slidevBin = join(root, "node_modules", ".bin", "slidev");
const port = Number(process.env.SLIDEV_REVIEW_PORT || 3130);
const baseUrl = `http://127.0.0.1:${port}`;

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: options.capture ? ["ignore", "pipe", "pipe"] : "inherit",
      env: { ...process.env, ...options.env },
    });
    let stdout = "";
    let stderr = "";
    if (options.capture) {
      child.stdout.on("data", chunk => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", chunk => {
        stderr += chunk.toString();
      });
    }
    child.on("error", reject);
    child.on("close", code => {
      if (code === 0) {
        resolvePromise({ stdout, stderr });
      } else {
        reject(new Error(`${command} ${args.join(" ")} exited ${code}\n${stdout}\n${stderr}`));
      }
    });
  });
}

function startServer() {
  const logPath = join(reviewDir, "slidev-server.log");
  const out = createWriteStream(logPath, { flags: "w" });
  const child = spawn(slidevBin, ["slides.md", "--port", String(port)], {
    cwd: root,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.pipe(out);
  child.stderr.pipe(out);
  return { child, logPath };
}

async function waitForServer(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise(resolvePromise => setTimeout(resolvePromise, 500));
  }
  throw new Error(`Slidev server did not become ready: ${lastError}`);
}

async function countPdfPages(path) {
  const data = await readFile(path);
  const text = data.toString("latin1");
  const matches = text.match(/\/Type\s*\/Page\b/g);
  return matches?.length ?? 0;
}

async function countSlidesFromMarkdown(path) {
  const text = await readFile(path, "utf8");
  const lines = text.split(/\r?\n/);
  let inFence = false;
  let frontmatterDelimiters = 0;
  let slideSeparators = 0;
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    if (/^---\s*$/.test(line)) {
      if (frontmatterDelimiters < 2) {
        frontmatterDelimiters += 1;
      } else {
        slideSeparators += 1;
      }
    }
  }
  return slideSeparators + 1;
}

function isIgnorableConsole(message) {
  return (
    message.includes("WakeLock")
    || message.includes("Wake Lock permission request denied")
    || message.includes("requesting page is not visible")
  );
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function main() {
  await mkdir(screenshotsDir, { recursive: true });

  console.log("Exporting Slidev PDF...");
  await run(slidevBin, ["export", "slides.md", "--output", pdfPath]);

  const pdfPageCount = await countPdfPages(pdfPath);
  const markdownSlideCount = await countSlidesFromMarkdown(slidesPath);
  const pageCount = pdfPageCount || markdownSlideCount;
  console.log(`PDF exported: ${pdfPath}`);
  console.log(`Pages from PDF metadata scan: ${pdfPageCount || "unavailable"}`);
  console.log(`Slides from markdown scan: ${markdownSlideCount}`);

  const server = startServer();
  let browser;
  const findings = [];
  const slides = [];

  try {
    await waitForServer();
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

    page.on("console", msg => {
      const text = msg.text();
      if (["error", "warning"].includes(msg.type()) && !isIgnorableConsole(text)) {
        findings.push({ kind: "console", type: msg.type(), text });
      }
    });
    page.on("pageerror", error => {
      const text = String(error);
      if (!isIgnorableConsole(text)) {
        findings.push({ kind: "pageerror", text });
      }
    });

    for (let index = 1; index <= pageCount; index += 1) {
      const url = `${baseUrl}/${index}`;
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(250);

      const text = (await page.locator("body").innerText()).trim();
      const title = (await page.title()).trim();
      const screenshotName = `slide-${String(index).padStart(3, "0")}.png`;
      const screenshotPath = join(screenshotsDir, screenshotName);
      await page.screenshot({ path: screenshotPath, fullPage: false });

      const flags = [];
      if (text.length < 12) flags.push("very little extracted visible text");
      if (/failed to fetch/i.test(text)) flags.push("contains failed-to-fetch text");

      slides.push({
        index,
        url,
        title,
        textLength: text.length,
        textPreview: text.slice(0, 240),
        screenshot: `slides/${screenshotName}`,
        flags,
      });
    }
  } finally {
    if (browser) await browser.close();
    server.child.kill("SIGTERM");
  }

  const flagged = slides.filter(slide => slide.flags.length > 0);
  const report = {
    pdf: basename(pdfPath),
    pageCount,
    reviewDir,
    screenshotsDir,
    serverLog: server.logPath,
    pdfPageCount,
    markdownSlideCount,
    findings,
    flaggedSlides: flagged,
    slides,
  };

  await writeFile(join(reviewDir, "review-report.json"), JSON.stringify(report, null, 2) + "\n");

  const md = [
    "# Slide Export Review",
    "",
    `- PDF: \`${pdfPath}\``,
    `- Pages reviewed: ${pageCount}`,
    `- PDF page count heuristic: ${pdfPageCount || "unavailable"}`,
    `- Markdown slide count: ${markdownSlideCount}`,
    `- Screenshots: \`${screenshotsDir}\``,
    `- Server log: \`${server.logPath}\``,
    `- Console/page findings: ${findings.length}`,
    `- Flagged slides: ${flagged.length}`,
    "",
    "## Findings",
    "",
    findings.length
      ? findings.map(item => `- ${item.kind}${item.type ? `/${item.type}` : ""}: ${item.text}`).join("\n")
      : "- No non-ignorable console or page errors captured.",
    "",
    "## Flagged Slides",
    "",
    flagged.length
      ? flagged.map(slide => `- Slide ${slide.index}: ${slide.flags.join(", ")} (${slide.screenshot})`).join("\n")
      : "- No slides flagged by the lightweight checks.",
    "",
    "## Slide Text Preview",
    "",
    ...slides.map(slide => `### Slide ${slide.index}\n\n${slide.textPreview || "(no text extracted)"}\n`),
  ].join("\n");
  await writeFile(join(reviewDir, "review-report.md"), md);

  const html = [
    "<!doctype html>",
    "<meta charset='utf-8'>",
    "<title>Slide Export Contact Sheet</title>",
    "<style>body{font-family:system-ui;margin:24px;background:#111;color:#eee} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}.card{background:#1c1c1c;padding:12px;border-radius:10px}.card.flag{outline:3px solid #f59e0b} img{width:100%;border-radius:6px;background:white}.meta{font-size:13px;color:#bbb} code{color:#93c5fd}</style>",
    "<h1>Slide Export Contact Sheet</h1>",
    `<p>PDF: <code>${escapeHtml(pdfPath)}</code></p>`,
    "<div class='grid'>",
    ...slides.map(slide => [
      `<div class='card ${slide.flags.length ? "flag" : ""}'>`,
      `<h2>Slide ${slide.index}</h2>`,
      `<img src='${escapeHtml(slide.screenshot)}'>`,
      `<p class='meta'>Text length: ${slide.textLength}${slide.flags.length ? ` · Flags: ${escapeHtml(slide.flags.join(", "))}` : ""}</p>`,
      `<p>${escapeHtml(slide.textPreview)}</p>`,
      "</div>",
    ].join("\n")),
    "</div>",
  ].join("\n");
  await writeFile(join(reviewDir, "contact-sheet.html"), html);

  console.log(`Review report: ${join(reviewDir, "review-report.md")}`);
  console.log(`Contact sheet: ${join(reviewDir, "contact-sheet.html")}`);

  if (findings.length || flagged.length) {
    console.log("Review completed with findings. Inspect the report/contact sheet.");
  } else {
    console.log("Review completed without automated findings.");
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
