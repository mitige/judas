import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, _electron as electron } from "playwright";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = path.join(root, "output", "playwright");
fs.mkdirSync(outDir, { recursive: true });

const started = [];
const checks = [];

function logCheck(name) {
  checks.push(name);
  console.log(`[ok] ${name}`);
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(fn, label, timeoutMs = 30000) {
  const startedAt = Date.now();
  let lastError;
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`${label} timed out${lastError ? `: ${lastError.message}` : ""}`);
}

async function json(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

async function httpOk(url) {
  const res = await fetch(url, { signal: AbortSignal.timeout(1000) });
  return res.ok;
}

async function portOpen(port) {
  try {
    await fetch(`http://127.0.0.1:${port}`, { signal: AbortSignal.timeout(600) });
    return true;
  } catch {
    return false;
  }
}

function run(command, args, cwd = root) {
  const child = spawn(command, args, {
    cwd,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.on("data", (b) => process.stdout.write(`[${path.basename(cwd)}] ${b}`));
  child.stderr.on("data", (b) => process.stderr.write(`[${path.basename(cwd)}] ${b}`));
  return child;
}

async function expectedServer(port, marker) {
  if (!marker) return true;
  try {
    const res = await fetch(`http://127.0.0.1:${port}`, {
      signal: AbortSignal.timeout(1500),
    });
    return (await res.text()).includes(marker);
  } catch {
    return false;
  }
}

async function ensureServer(name, port, command, args, cwd, readyUrl, marker) {
  if (await portOpen(port)) {
    if (!(await expectedServer(port, marker))) {
      throw new Error(
        `le port :${port} est occupé par un AUTRE serveur que ${name} ` +
        `(marqueur "${marker}" absent) — sur cette machine c'est souvent le ` +
        `frontend Orchidée. Libérer le port puis relancer.`
      );
    }
    console.log(`[reuse] ${name} on :${port}`);
    return;
  }
  console.log(`[start] ${name}`);
  const child = run(command, args, cwd);
  started.push(child);
  await waitFor(async () => {
    try {
      return await httpOk(readyUrl);
    } catch {
      return false;
    }
  }, `${name} ready`, 60000);
}

function killStarted() {
  for (const child of started.reverse()) {
    if (!child.pid || child.exitCode != null) continue;
    try {
      execFileSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
        stdio: "ignore",
      });
    } catch {
      try {
        child.kill();
      } catch {}
    }
  }
}

function field(page, label) {
  return page.locator(".field", { hasText: label }).locator("input, select").first();
}

async function expectNoClientErrors(errors, scope) {
  const serious = errors.filter((entry) =>
    !/favicon|Failed to load resource/i.test(entry)
  );
  if (serious.length) {
    throw new Error(`${scope} client errors:\n${serious.join("\n")}`);
  }
}

async function selectOptionContaining(select, needle) {
  const value = await waitFor(async () => select.evaluate((el, text) => {
    const opt = [...el.options].find((o) => o.textContent.includes(text) || o.value.includes(text));
    return opt?.value || null;
  }, needle), `option containing ${needle}`, 10000);
  if (!value) throw new Error(`no option containing ${needle}`);
  await select.selectOption(value);
}

async function smokeControlApp(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`${msg.type()}: ${msg.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));

  await page.goto("http://127.0.0.1:5173", { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "Dashboard" }).waitFor();
  await page.getByText("daemon online").waitFor({ timeout: 5000 });
  await page.getByText(/gpu|no gpu/i).first().waitFor();
  logCheck("app dashboard renders with daemon status");

  await page.getByRole("button", { name: "Training" }).click();
  await page.getByRole("heading", { name: "Training" }).waitFor();
  await field(page, "name").fill("ui_train_smoke");
  await field(page, "envs").fill("2");
  await field(page, "rollout").fill("8");
  await field(page, "iterations").fill("100000");
  await field(page, "minibatch").fill("16");
  await field(page, "target hits").fill("3");
  await field(page, "max ticks").fill("60");
  await page.getByRole("button", { name: /start/i }).click();
  await waitFor(async () => (await json("http://127.0.0.1:8765/status")).training.running,
    "training running", 20000);
  logCheck("training start button launches supervised process");
  await page.getByRole("button", { name: /stop/i }).click();
  await waitFor(async () => !(await json("http://127.0.0.1:8765/status")).training.running,
    "training stopped", 30000);
  logCheck("training stop button terminates supervised process");

  await page.getByRole("button", { name: "Models" }).click();
  await page.getByRole("heading", { name: "Models" }).waitFor();
  await page.getByText("ui_smoke").first().waitFor();
  const exportButton = page.locator(
    'xpath=//span[contains(@class,"k") and normalize-space()="ui_smoke"]' +
    '/ancestor::div[contains(@class,"kv")]/following-sibling::table[1]' +
    '//button[contains(normalize-space(.),"export")]'
  ).first();
  await exportButton.click();
  await waitFor(async () => {
    const models = await json("http://127.0.0.1:8765/models");
    return models.exported.some((m) => m.path.includes("ui_smoke-ckpt_000001"));
  }, "model exported from UI", 30000);
  logCheck("models export button creates TorchScript export");

  await page.getByRole("button", { name: "Live" }).click();
  await page.getByRole("heading", { name: "Live" }).waitFor();
  const liveSelect = page.locator("select").first();
  await selectOptionContaining(liveSelect, "ui-smoke");
  await page.getByRole("button", { name: /^load$/i }).click();
  await waitFor(async () => {
    const st = await json("http://127.0.0.1:8765/status");
    return st.live.model && st.live.model.includes("ui-smoke");
  }, "live model loaded", 30000);
  logCheck("live load button loads exported model");
  await page.getByRole("button", { name: /^arm$/i }).click();
  await waitFor(async () => (await json("http://127.0.0.1:8765/status")).live.enabled,
    "live armed", 10000);
  logCheck("live arm button enables live inference");
  await page.getByRole("button", { name: /^kill$/i }).click();
  await waitFor(async () => !(await json("http://127.0.0.1:8765/status")).live.enabled,
    "live killed", 10000);
  logCheck("live kill button disables live inference");

  await expectNoClientErrors(errors, "control app");
  await page.close();
}

async function smokeViz(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`${msg.type()}: ${msg.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));

  await page.goto("http://127.0.0.1:5174", { waitUntil: "networkidle" });
  await page.getByText(/connected|daemon offline/i).first().waitFor();
  await page.getByText("connected").waitFor({ timeout: 5000 });
  const selects = page.locator("select");
  await selectOptionContaining(selects.nth(0), "ui-smoke");
  await selectOptionContaining(selects.nth(1), "ui-smoke");
  await page.getByRole("button", { name: /^load$/i }).click();
  await waitFor(async () => (await json("http://127.0.0.1:8765/arena/status")).ready,
    "arena ready", 30000);
  logCheck("arena load button prepares model-vs-model match");
  await page.getByRole("button", { name: /play/i }).click();
  await waitFor(async () => (await json("http://127.0.0.1:8765/arena/status")).running,
    "arena running", 10000);
  await page.waitForTimeout(1500);
  await page.getByText(/tick\s+\d+/i).first().waitFor();
  const canvasOk = await page.locator("canvas").evaluateAll((canvases) =>
    canvases.some((canvas) => {
      const ctx = canvas.getContext("webgl2") || canvas.getContext("webgl");
      return Boolean(ctx) && canvas.width > 0 && canvas.height > 0;
    })
  );
  if (!canvasOk) throw new Error("arena WebGL canvas is not initialized");
  logCheck("arena play button streams ticks and renders WebGL canvas");
  await page.getByRole("button", { name: /pause/i }).click();
  await waitFor(async () => !(await json("http://127.0.0.1:8765/arena/status")).running,
    "arena paused", 10000);
  logCheck("arena pause button stops simulation loop");

  await expectNoClientErrors(errors, "arena app");
  await page.close();
}

async function smokeElectronShell(appDir, readyText, port) {
  const executablePath = path.join(root, appDir, "node_modules", "electron", "dist", "electron.exe");
  if (!fs.existsSync(executablePath)) {
    throw new Error(`Electron executable missing for ${appDir}: ${executablePath}`);
  }
  const electronApp = await electron.launch({
    executablePath,
    args: [path.join(root, appDir)],
    env: { ...process.env, NODE_ENV: "development" },
  });
  try {
    const page = await electronApp.firstWindow();
    await page.waitForLoadState("domcontentloaded");
    await waitFor(() => page.url().includes(`:${port}`), `${appDir} dev URL`, 10000);
    await page.getByText(readyText).first().waitFor({ timeout: 10000 });
    logCheck(`${appDir} Electron shell loads dev UI`);
  } finally {
    await electronApp.close();
  }
}

async function main() {
  execFileSync("cmd.exe", ["/d", "/s", "/c", "call scripts\\env.bat && python tools\\ui_smoke_seed.py"], {
    cwd: root,
    stdio: "inherit",
  });

  await ensureServer(
    "daemon",
    8765,
    "cmd.exe",
    ["/d", "/s", "/c", "scripts\\daemon.bat"],
    root,
    "http://127.0.0.1:8765/status",
    null                                  // /status validé par readyUrl
  );
  await ensureServer(
    "app vite",
    5173,
    "cmd.exe",
    ["/d", "/s", "/c", "npm --prefix app run dev:web -- --host 127.0.0.1"],
    root,
    "http://127.0.0.1:5173",
    "<title>Judas</title>"
  );
  await ensureServer(
    "viz vite",
    5174,
    "cmd.exe",
    ["/d", "/s", "/c", "npm --prefix viz run dev:web -- --host 127.0.0.1"],
    root,
    "http://127.0.0.1:5174",
    "<title>Judas — Arène</title>"
  );

  const browser = await chromium.launch({ headless: true });
  try {
    await smokeElectronShell("app", "Dashboard", 5173);
    await smokeElectronShell("viz", "fighters", 5174);
    await smokeControlApp(browser);
    await smokeViz(browser);
  } catch (error) {
    for (const page of browser.contexts().flatMap((ctx) => ctx.pages())) {
      await page.screenshot({
        path: path.join(outDir, `failure-${Date.now()}.png`),
        fullPage: true,
      }).catch(() => {});
    }
    throw error;
  } finally {
    await browser.close();
  }

  console.log(`\nUI smoke passed (${checks.length} checks).`);
}

main()
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  })
  .finally(killStarted);
