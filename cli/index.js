#!/usr/bin/env node
/**
 * cli/index.js — DeepRisk OSS Supply Chain Security CLI
 *
 * HACKATHON DEMO FEATURES:
 *   • Rich terminal output with risk bars, CVE IDs, safer alternatives
 *   • <500ms response (Redis/local cache hit = <5ms)
 *   • Offline mode: uses cached results with age warning
 *   • Exit code 2 on HIGH risk (integrates with CI/CD pipelines)
 *   • `deeprisk demo` runs a pre-scripted hackathon scenario
 *
 * FIXES (v1.0.1):
 *   • Windows-safe HTTP timeout (clearTimeout + abort controller pattern)
 *   • ANSI colour auto-detection (works on cmd, PowerShell, Windows Terminal)
 *   • fetchBatch no longer double-hangs on offline servers
 *   • All async paths guaranteed to resolve or print an error
 */
"use strict";

const { spawnSync } = require("child_process");
const readline      = require("readline");
const fs            = require("fs");
const path          = require("path");
const os            = require("os");
const http          = require("http");
const https         = require("https");
const { URL }       = require("url");

// ─── Config ───────────────────────────────────────────────────────────────────
const CFG = {
  apiUrl  : process.env.DEEPRISK_API_URL  || "http://localhost:8000",
  timeout : parseInt(process.env.DEEPRISK_TIMEOUT || "4500"),
  cacheDir: path.join(os.homedir(), ".deeprisk", "cache"),
  cacheTtl: 7 * 24 * 3600 * 1000,
  noBlock : !!process.env.DEEPRISK_NO_BLOCK,
  version : "1.0.1",
};

const cfgFile = path.join(os.homedir(), ".deeprisk", "config.json");
try {
  if (fs.existsSync(cfgFile))
    Object.assign(CFG, JSON.parse(fs.readFileSync(cfgFile, "utf8")));
} catch (_) {}

// ─── Colours (Windows-safe) ───────────────────────────────────────────────────
// Enable VT sequences on Windows 10+ (no-op on other platforms)
if (process.platform === "win32") {
  try {
    // Attempt to enable ANSI on Windows via a benign escape write
    process.stdout.write("\x1b[0m");
  } catch (_) {}
}

const USE_COLOR =
  process.stdout.isTTY &&
  process.env.NO_COLOR === undefined &&
  (
    process.platform !== "win32" ||       // Unix always on
    !!process.env.WT_SESSION  ||          // Windows Terminal
    !!process.env.TERM_PROGRAM ||         // VSCode terminal
    process.env.TERM === "xterm-256color" ||
    !!process.env.ConEmuPID               // ConEmu / cmder
  );

const C = USE_COLOR
  ? {
      r   : "\x1b[31m", y: "\x1b[33m", g: "\x1b[32m", c: "\x1b[36m",
      grey: "\x1b[90m", bold: "\x1b[1m", dim: "\x1b[2m", E: "\x1b[0m",
    }
  : { r:"", y:"", g:"", c:"", grey:"", bold:"", dim:"", E:"" };

const col = (lbl, txt) =>
  ({ HIGH: C.r, MEDIUM: C.y, LOW: C.g, UNKNOWN: C.grey }[lbl] || "") + txt + C.E;

// ─── Cache ────────────────────────────────────────────────────────────────────
const cacheKey  = n => path.join(CFG.cacheDir, n.replace(/[@/\\:*?"<>|]/g, "_") + ".json");
const ensureDir = d => { try { fs.mkdirSync(d, { recursive: true }); } catch (_) {} };

function readCache(name) {
  ensureDir(CFG.cacheDir);
  const f = cacheKey(name);
  if (!fs.existsSync(f)) return null;
  try {
    const { ts, data } = JSON.parse(fs.readFileSync(f, "utf8"));
    const age = Date.now() - ts;
    if (age > CFG.cacheTtl) { try { fs.unlinkSync(f); } catch (_) {} return null; }
    data._cached = true;
    data._age_h  = (age / 3_600_000).toFixed(1);
    return data;
  } catch (_) { return null; }
}

function writeCache(name, data) {
  ensureDir(CFG.cacheDir);
  try {
    fs.writeFileSync(cacheKey(name), JSON.stringify({ ts: Date.now(), data }));
  } catch (_) {}
}

// ─── HTTP helper (Windows-safe timeout) ───────────────────────────────────────
function post(urlStr, body) {
  return new Promise((resolve, reject) => {
    const u   = new URL(urlStr);
    const pay = JSON.stringify(body);
    const lib = u.protocol === "https:" ? https : http;

    let settled  = false;
    let timer    = null;

    /** Call exactly once */
    const settle = (ok, val) => {
      if (settled) return;
      settled = true;
      if (timer) { clearTimeout(timer); timer = null; }
      ok ? resolve(val) : reject(val);
    };

    const req = lib.request(
      {
        hostname: u.hostname,
        port    : u.port || (u.protocol === "https:" ? 443 : 80),
        path    : u.pathname,
        method  : "POST",
        headers : {
          "Content-Type"  : "application/json",
          "Content-Length": Buffer.byteLength(pay),
          "User-Agent"    : `deeprisk-cli/${CFG.version}`,
        },
      },
      (res) => {
        let raw = "";
        res.on("data", chunk => (raw += chunk));
        res.on("end", () => {
          try { settle(true, JSON.parse(raw)); }
          catch (e) {
            settle(false, new Error("JSON parse error: " + raw.slice(0, 80)));
          }
        });
        res.on("error", e => settle(false, e));
      }
    );

    // ── Windows-safe timeout: use clearTimeout, not req.setTimeout alone ──
    timer = setTimeout(() => {
      try { req.destroy(new Error(`Timeout after ${CFG.timeout}ms`)); } catch (_) {}
      settle(false, new Error(`Request timed out after ${CFG.timeout}ms`));
    }, CFG.timeout);

    req.on("error",   e  => settle(false, e));
    req.on("timeout", () => {
      try { req.destroy(); } catch (_) {}
      settle(false, new Error(`Socket timeout after ${CFG.timeout}ms`));
    });

    req.setTimeout(CFG.timeout);
    req.write(pay);
    req.end();
  });
}

// ─── Offline result factory ───────────────────────────────────────────────────
const offlineResult = (name, msg) => ({
  package   : name,
  risk_label: "UNKNOWN",
  risk_score: null,
  risk_prob : null,
  risk_tier : "Server unreachable",
  in_dataset: false,
  warnings  : [`API error: ${msg}`],
  explanation: "Cannot assess risk — server offline or unreachable.",
  _offline  : true,
});

// ─── Fetch helpers ─────────────────────────────────────────────────────────────
async function fetchOne(name) {
  const cached = readCache(name);
  if (cached) return cached;

  try {
    const d = await post(`${CFG.apiUrl}/predict`, { package: name, with_neighbors: true });
    writeCache(name, d);
    return d;
  } catch (e) {
    return offlineResult(name, e.message);
  }
}

async function fetchBatch(names) {
  if (!names.length) return [];

  const fromCache = [];
  const needApi   = [];

  for (const n of names) {
    const c = readCache(n);
    if (c) fromCache.push([n, c]);
    else   needApi.push(n);
  }

  const map = new Map(fromCache);

  if (needApi.length) {
    let batchOk = false;
    try {
      const resp = await post(`${CFG.apiUrl}/predict/batch`, {
        packages      : needApi,
        with_neighbors: false,
      });
      const list = Array.isArray(resp) ? resp : (resp.results || []);
      for (const r of list) {
        writeCache(r.package, r);
        map.set(r.package, r);
      }
      batchOk = true;
    } catch (e) {
      // Batch endpoint failed — fall back to individual fetches.
      // fetchOne itself catches all errors and returns an offline stub,
      // so this loop will ALWAYS complete without throwing.
      for (const n of needApi) {
        map.set(n, await fetchOne(n));
      }
    }

    // Safety net: if batch succeeded but a package was missing from response
    if (batchOk) {
      for (const n of needApi) {
        if (!map.has(n)) map.set(n, offlineResult(n, "Not found in batch response"));
      }
    }
  }

  return names.map(n => map.get(n) || offlineResult(n, "Unknown fetch failure"));
}

// ─── Display ──────────────────────────────────────────────────────────────────
function bar(score, w = 24) {
  const filled = Math.max(0, Math.min(w, Math.round((score / 100) * w)));
  return "█".repeat(filled) + "░".repeat(w - filled);
}

function printCard(d, showAlts = true) {
  const {
    package: pkg, risk_score, risk_prob, risk_label, risk_tier,
    top_neighbors = [], osv, explanation, warnings = [], alternatives = [],
    _cached, _age_h, _offline,
  } = d;

  const icons = { HIGH: "⚠ ", MEDIUM: "⚡", LOW: "✅", UNKNOWN: "❓" };
  const icon  = icons[risk_label] || "❓";

  console.log("\n" + "─".repeat(62));
  console.log(`${icon}  ${C.bold}${pkg}${C.E}`);
  console.log("─".repeat(62));

  if (risk_score === null || !d.in_dataset) {
    console.log(col("UNKNOWN", `  Status  : ${risk_tier || "Unknown"}`));
    if (osv && osv.queried) {
      const cveLine = osv.cve_ids && osv.cve_ids.length
        ? `  CVEs: ${osv.cve_ids.slice(0, 3).join(", ")}`
        : "";
      console.log(`  OSV.dev : ${osv.vuln_count} vuln(s)${cveLine}`);
    }
  } else {
    console.log(col(risk_label,
      `  Score    : ${risk_score}/100  [${bar(risk_score)}]  ${risk_label}`));
    console.log(`  Tier     : ${risk_tier}`);
    if (risk_prob !== null && risk_prob !== undefined) {
      const thr = d.threshold != null ? (d.threshold * 100).toFixed(0) : "?";
      const tmp = d.temperature != null ? d.temperature.toFixed(2) : "?";
      console.log(`  Prob     : ${(risk_prob * 100).toFixed(1)}%  (T=${tmp}, thr=${thr}%)`);
    }

    if (osv && osv.queried && osv.vuln_count > 0) {
      const cveLine = (osv.cve_ids || []).slice(0, 3).join(", ");
      console.log(`  ${C.r}OSV      : ${osv.vuln_count} known vuln(s)  ${cveLine}${C.E}`);
    }

    if (explanation)
      console.log(`\n  ${C.c}Why:${C.E} ${explanation}`);

    if (top_neighbors.length) {
      console.log(`\n  ${C.c}Top graph neighbours (attention):${C.E}`);
      top_neighbors.slice(0, 3).forEach(n => {
        const sym = n.risk_prob >= (d.threshold || 0.5) ? "⚠" : "○";
        const prob = n.risk_prob != null ? (n.risk_prob * 100).toFixed(1) : "?";
        const attn = n.attention_weight != null ? n.attention_weight.toFixed(3) : "?";
        console.log(`    ${sym} ${n.package.padEnd(28)} risk=${prob}%  attn=${attn}`);
      });
    }
  }

  if (showAlts && alternatives.length) {
    console.log(`\n  ${C.g}💡 Safer alternatives:${C.E}`);
    alternatives.forEach(a => console.log(`     • ${a}`));
  }
  if (warnings.length)
    warnings.forEach(w => console.log(`  ${C.y}⚠  ${w}${C.E}`));
  if (_cached)
    console.log(`  ${C.grey}(cached ${_age_h}h ago)${C.E}`);
  if (_offline)
    console.log(`  ${C.grey}(offline — server unreachable)${C.E}`);

  console.log("─".repeat(62));
}

function printBatchSummary(results) {
  const H = results.filter(r => r.risk_label === "HIGH").length;
  const M = results.filter(r => r.risk_label === "MEDIUM").length;
  const L = results.filter(r => r.risk_label === "LOW").length;
  const U = results.filter(r => r.risk_label === "UNKNOWN").length;

  console.log("\n" + "═".repeat(62));
  console.log(`${C.bold}BATCH SCAN SUMMARY${C.E}  (${results.length} packages)`);
  console.log("═".repeat(62));
  if (H) console.log(col("HIGH",    `  HIGH risk     : ${H}`));
  if (M) console.log(col("MEDIUM",  `  MEDIUM risk   : ${M}`));
  if (L) console.log(col("LOW",     `  LOW risk      : ${L}`));
  if (U) console.log(col("UNKNOWN", `  UNKNOWN       : ${U}`));

  if (H) {
    console.log(`\n  ${C.r}HIGH risk packages:${C.E}`);
    results
      .filter(r => r.risk_label === "HIGH")
      .forEach(r => {
        const pct = r.risk_prob != null ? ` (${(r.risk_prob * 100).toFixed(1)}%)` : "";
        console.log(`    ⚠  ${r.package}${pct}`);
      });
  }
  console.log("═".repeat(62));
}

function ask(q) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise(r => rl.question(q, a => { rl.close(); r(a); }));
}

// ─── Commands ─────────────────────────────────────────────────────────────────

const { spawn } = require("child_process");

async function cmdInstall(args) {
  const pkgs  = args.filter(a => !a.startsWith("-"));
  const flags = args.filter(a => a.startsWith("-"));

  // 👉 If no package → normal npm install
  if (!pkgs.length) {
    spawn("npm", ["install", ...flags], {
      stdio: "inherit",
      shell: true
    });
    return;
  }

  console.log(`\n${C.c}🔍 DeepRisk scanning ${pkgs.length} package(s)…${C.E}`);
  const t0 = Date.now();

  const results = await fetchBatch(pkgs);

  console.log(`${C.grey}  Scan completed in ${Date.now() - t0}ms${C.E}`);

  let blocked = false;

  for (const r of results) {
    printCard(r, true);

    if (r.risk_label === "HIGH" && !CFG.noBlock) {
      const ans = await ask(
        `\n${C.y}⚠  ${r.package} is HIGH risk. Install anyway? (y/N) ${C.E}`
      );

      if (ans.trim().toLowerCase() !== "y") {
        console.log(`${C.r}✗ Installation of ${r.package} aborted.${C.E}`);
        blocked = true;
      }
    }
  }

  // 🚫 If blocked → stop
  if (blocked) {
    console.log(`\n${C.r}❌ Install cancelled.${C.E}`);
    process.exit(1);
  }

  // ✅ Run npm install (ONLY ONCE)
  console.log(`\n📦 Running npm install...\n`);

  const install = spawn("npm", ["install", ...pkgs, ...flags], {
    stdio: "inherit",
    shell: true   // 🔥 required for Windows
  });

  install.on("close", (code) => {
    if (code === 0) {
      console.log(`\n${C.g}✅ Installation completed successfully${C.E}`);
    } else {
      console.error(`\n${C.r}❌ npm install failed (code ${code})${C.E}`);
    }
    process.exit(code);
  });
}

async function cmdScan(args) {
  // Support piped input (e.g. cat packages.txt | deeprisk scan)
  if (!args.length && !process.stdin.isTTY) {
    const rl    = readline.createInterface({ input: process.stdin });
    const lines = [];
    for await (const l of rl) {
      const t = l.trim();
      if (t && !t.startsWith("#")) lines.push(t);
    }
    args = lines;
  }

  if (!args.length) {
    console.error("Usage: deeprisk scan <pkg> [pkg2...]\n       echo 'lodash' | deeprisk scan");
    process.exit(1);
  }

  console.log(`\n${C.c}🔍 DeepRisk scanning ${args.length} package(s)…${C.E}`);
  const t0      = Date.now();
  const results = await fetchBatch(args);
  console.log(`${C.grey}Scan completed in ${Date.now() - t0}ms${C.E}`);

  results.forEach(r => printCard(r, true));
  if (args.length > 1) printBatchSummary(results);

  const hasHigh = results.some(r => r.risk_label === "HIGH");
  process.exit(hasHigh ? 2 : 0);
}



async function cmdHealth() {
  console.log(`Checking ${CFG.apiUrl} …`);
  try {
    const d = await post(`${CFG.apiUrl}/health`, {});
    console.log(`${C.g}✓ API server reachable${C.E}  url=${CFG.apiUrl}`);
    console.log(`  model_ready : ${d.model_ready}`);
    console.log(`  startup_ms  : ${d.startup_ms}ms`);
    if (d.startup_error) console.log(`  ${C.r}error: ${d.startup_error}${C.E}`);
  } catch (e) {
    console.log(`${C.r}✗ Server unreachable${C.E}  url=${CFG.apiUrl}`);
    console.log(`  Reason : ${e.message}`);
    console.log(`\nStart server: cd backend && python main.py`);
    process.exit(1);
  }
}

async function cmdConfig(args) {
  if (!args.length) { console.log(JSON.stringify(CFG, null, 2)); return; }
  const [key, val] = args;
  if (!val) { console.log(`${key} = ${CFG[key]}`); return; }
  ensureDir(path.dirname(cfgFile));
  let saved = {};
  try { saved = JSON.parse(fs.readFileSync(cfgFile, "utf8")); } catch (_) {}
  saved[key] = isNaN(val) ? val : Number(val);
  fs.writeFileSync(cfgFile, JSON.stringify(saved, null, 2));
  console.log(`${C.g}✓ Set ${key} = ${val}${C.E}`);
}

async function cmdCache(args) {
  const sub = args[0] || "info";
  ensureDir(CFG.cacheDir);
  if (sub === "clear") {
    const files = fs.readdirSync(CFG.cacheDir).filter(f => f.endsWith(".json"));
    files.forEach(f => { try { fs.unlinkSync(path.join(CFG.cacheDir, f)); } catch (_) {} });
    console.log(`${C.g}✓ Cleared ${files.length} cache entries${C.E}`);
  } else {
    const n = fs.readdirSync(CFG.cacheDir).filter(f => f.endsWith(".json")).length;
    console.log(`Cache dir : ${CFG.cacheDir}`);
    console.log(`Entries   : ${n}`);
    console.log(`TTL       : 7 days`);
  }
}

function help() {
  console.log(`
${C.bold}DeepRisk CLI v${CFG.version}${C.E} — npm Supply Chain Security Scanner

${C.c}Commands:${C.E}
  deeprisk install <pkg> [flags]    Scan then install (blocks HIGH risk)
  deeprisk scan    <pkg> [pkg2...]  Scan only (exit 2 on HIGH risk)
  deeprisk health                   Check API server connectivity
  deeprisk config  [key] [value]    Show / set config
  deeprisk cache   clear|info       Manage local cache

${C.c}Examples:${C.E}
  deeprisk install colors             
  deeprisk install chalk              
  deeprisk scan lodash express axios
  deeprisk config 
  DEEPRISK_NO_BLOCK=1 deeprisk install colors   # warn only, never block

${C.c}Pipe mode:${C.E}
  cat packages.txt | deeprisk scan

${C.c}Exit codes:${C.E}  0 = safe   1 = install aborted   2 = HIGH risk found (scan only)
`);
}

// ─── Safe console wrapper (guards against Windows encoding crashes) ───────────
const say = (...args) => {
  try { console.log(...args); }
  catch (_) {
    // Strip ANSI and retry with plain ASCII if encoding blows up
    const plain = args.join(" ").replace(/\x1b\[[0-9;]*m/g, "")
                                 .replace(/[^\x00-\x7F]/g, "?");
    process.stdout.write(plain + "\n");
  }
};

// Patch console.log globally so all existing calls are safe
const _origLog = console.log.bind(console);
console.log = (...a) => {
  try { _origLog(...a); }
  catch (_) {
    const plain = a.join(" ").replace(/\x1b\[[0-9;]*m/g, "")
                              .replace(/[^\x00-\x7F]/g, "?");
    process.stdout.write(plain + "\n");
  }
};

// ─── Entry ────────────────────────────────────────────────────────────────────
async function main() {
  // Write immediately so we know the script IS running
  process.stdout.write(`DeepRisk CLI v${CFG.version} starting...\n`);

  const [,, cmd, ...rest] = process.argv;

  switch (cmd) {
    case "install": await cmdInstall(rest); break;
    case "scan":    await cmdScan(rest);    break;
    case "demo":    await cmdDemo();        break;
    case "health":  await cmdHealth();      break;
    case "config":  await cmdConfig(rest);  break;
    case "cache":   await cmdCache(rest);   break;
    case "--help":
    case "-h":
    case "help":    help(); break;
    case "--version":
    case "-v":      console.log(`deeprisk v${CFG.version}`); break;
    default:
      if (!cmd) help();
      else { console.error(`${C.r}Unknown command: ${cmd}${C.E}\n`); help(); process.exit(1); }
  }
}

main().catch(e => {
  console.error(`\n${C.r}Fatal error: ${e.message}${C.E}`);
  if (process.env.DEBUG) console.error(e.stack);
  process.exit(1);
});
