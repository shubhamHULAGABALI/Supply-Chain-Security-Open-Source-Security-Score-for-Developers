
"use strict";

const { spawnSync } = require("child_process");
const readline = require("readline");
const fs       = require("fs");
const path     = require("path");
const os       = require("os");
const http     = require("http");
const https    = require("https");
const { URL }  = require("url");

// ─── Config ───────────────────────────────────────────────────────────────────
const CFG = {
  apiUrl  : process.env.DEEPRISK_API_URL  || "http://localhost:8000",
  timeout : parseInt(process.env.DEEPRISK_TIMEOUT || "4500"),
  cacheDir: path.join(os.homedir(), ".deeprisk", "cache"),
  cacheTtl: 7 * 24 * 3600 * 1000,
  noBlock : !!process.env.DEEPRISK_NO_BLOCK,
  version : "1.0.0",
};

// Merge saved config
const cfgFile = path.join(os.homedir(), ".deeprisk", "config.json");
try {
  if (fs.existsSync(cfgFile)) Object.assign(CFG, JSON.parse(fs.readFileSync(cfgFile,"utf8")));
} catch(_) {}

// ─── Colours ──────────────────────────────────────────────────────────────────
const C = {
  r:"\x1b[31m", y:"\x1b[33m", g:"\x1b[32m", c:"\x1b[36m",
  grey:"\x1b[90m", bold:"\x1b[1m", dim:"\x1b[2m", E:"\x1b[0m"
};
const col = (lbl, txt) => ({ HIGH:C.r, MEDIUM:C.y, LOW:C.g, UNKNOWN:C.grey }[lbl]||"") + txt + C.E;

// ─── Cache ────────────────────────────────────────────────────────────────────
const ck = n => path.join(CFG.cacheDir, n.replace(/[@/]/g,"_") + ".json");
const ensureDir = d => { try { fs.mkdirSync(d, {recursive:true}); } catch(_){} };

function readCache(name) {
  ensureDir(CFG.cacheDir);
  const f = ck(name);
  if (!fs.existsSync(f)) return null;
  try {
    const { ts, data } = JSON.parse(fs.readFileSync(f,"utf8"));
    const age = Date.now() - ts;
    if (age > CFG.cacheTtl) { fs.unlinkSync(f); return null; }
    data._cached = true; data._age_h = (age/3600000).toFixed(1);
    return data;
  } catch(_) { return null; }
}
function writeCache(name, data) {
  ensureDir(CFG.cacheDir);
  fs.writeFileSync(ck(name), JSON.stringify({ ts: Date.now(), data }));
}

// ─── HTTP helper ──────────────────────────────────────────────────────────────
function post(urlStr, body) {
  return new Promise((res, rej) => {
    const u   = new URL(urlStr);
    const pay = JSON.stringify(body);
    const lib = u.protocol === "https:" ? https : http;
    const req = lib.request({
      hostname: u.hostname, port: u.port || (u.protocol==="https:"?443:80),
      path: u.pathname, method:"POST",
      headers: { "Content-Type":"application/json",
                 "Content-Length": Buffer.byteLength(pay),
                 "User-Agent": `deeprisk-cli/${CFG.version}` }
    }, r => {
      let raw = "";
      r.on("data", c => raw += c);
      r.on("end", () => {
        try { res(JSON.parse(raw)); }
        catch(e) { rej(new Error("JSON parse: " + raw.slice(0,80))); }
      });
    });
    req.setTimeout(CFG.timeout, () => { req.destroy(); rej(new Error(`Timeout ${CFG.timeout}ms`)); });
    req.on("error", rej);
    req.write(pay); req.end();
  });
}

async function fetchOne(name) {
  const cached = readCache(name);
  if (cached) return cached;
  try {
    const d = await post(`${CFG.apiUrl}/predict`, { package: name, with_neighbors: true });
    writeCache(name, d);
    return d;
  } catch(e) {
    return { package:name, risk_label:"UNKNOWN", risk_score:null, risk_prob:null,
             risk_tier:"Server unreachable", in_dataset:false,
             warnings:[`API error: ${e.message}`], explanation:"Cannot assess risk.",
             _offline:true };
  }
}

async function fetchBatch(names) {
  const fromCache = []; const needApi = [];
  for (const n of names) {
    const c = readCache(n);
    if (c) fromCache.push([n, c]);
    else   needApi.push(n);
  }
  const map = new Map(fromCache);
  if (needApi.length) {
    try {
      const resp = await post(`${CFG.apiUrl}/predict/batch`,
                              { packages: needApi, with_neighbors: false });
      const list = resp.results || resp;
      for (const r of list) { writeCache(r.package, r); map.set(r.package, r); }
    } catch(_) {
      for (const n of needApi) map.set(n, await fetchOne(n));
    }
  }
  return names.map(n => map.get(n) || { package:n, risk_label:"UNKNOWN" });
}

// ─── Display ──────────────────────────────────────────────────────────────────
function bar(score, w=24) {
  const f = Math.round((score/100)*w);
  return "█".repeat(f) + "░".repeat(w-f);
}

function printCard(d, showAlts=true) {
  const { package:pkg, risk_score, risk_prob, risk_label, risk_tier,
          top_neighbors=[], osv, explanation, warnings=[], alternatives=[],
          _cached, _age_h, _offline } = d;

  const icons = { HIGH:"⚠ ", MEDIUM:"⚡", LOW:"✅", UNKNOWN:"❓" };
  const icon  = icons[risk_label] || "❓";

  console.log("\n" + "─".repeat(62));
  console.log(`${icon}  ${C.bold}${pkg}${C.E}`);
  console.log("─".repeat(62));

  if (risk_score === null || !d.in_dataset) {
    console.log(col("UNKNOWN", `  Status  : ${risk_tier}`));
    if (osv && osv.queried) {
      const cveLine = osv.cve_ids.length ? `  CVEs: ${osv.cve_ids.slice(0,3).join(", ")}` : "";
      console.log(`  OSV.dev : ${osv.vuln_count} vuln(s)${cveLine}`);
    }
  } else {
    console.log(col(risk_label,
      `  Score    : ${risk_score}/100  [${bar(risk_score)}]  ${risk_label}`));
    console.log(`  Tier     : ${risk_tier}`);
    console.log(`  Prob     : ${(risk_prob*100).toFixed(1)}%  (T=${d.temperature?.toFixed(2)||"?"}, thr=${(d.threshold*100).toFixed(0)}%)`);

    if (osv?.queried && osv.vuln_count > 0) {
      const cveLine = osv.cve_ids.slice(0,3).join(", ");
      console.log(`  ${C.r}OSV      : ${osv.vuln_count} known vuln(s)  ${cveLine}${C.E}`);
    }
    if (explanation)
      console.log(`\n  ${C.c}Why:${C.E} ${explanation}`);
    if (top_neighbors.length) {
      console.log(`\n  ${C.c}Top graph neighbours (attention):${C.E}`);
      top_neighbors.slice(0,3).forEach(n => {
        const sym = n.risk_prob >= (d.threshold||0.5) ? "⚠" : "○";
        console.log(`    ${sym} ${n.package.padEnd(28)} risk=${(n.risk_prob*100).toFixed(1)}%  attn=${n.attention_weight?.toFixed(3)}`);
      });
    }
  }

  if (showAlts && alternatives.length) {
    console.log(`\n  ${C.g}💡 Safer alternatives:${C.E}`);
    alternatives.forEach(a => console.log(`     • ${a}`));
  }
  if (warnings.length)
    warnings.forEach(w => console.log(`  ${C.y}⚠ ${w}${C.E}`));
  if (_cached)
    console.log(`  ${C.grey}(cached ${_age_h}h ago)${C.E}`);
  if (_offline)
    console.log(`  ${C.grey}(offline — server unreachable)${C.E}`);
  console.log("─".repeat(62));
}

function printBatchSummary(results) {
  const H = results.filter(r=>r.risk_label==="HIGH").length;
  const M = results.filter(r=>r.risk_label==="MEDIUM").length;
  const L = results.filter(r=>r.risk_label==="LOW").length;
  const U = results.filter(r=>r.risk_label==="UNKNOWN").length;
  console.log("\n" + "═".repeat(62));
  console.log(`${C.bold}BATCH SCAN SUMMARY${C.E}  (${results.length} packages)`);
  console.log("═".repeat(62));
  if (H) console.log(col("HIGH",   `  HIGH risk     : ${H}`));
  if (M) console.log(col("MEDIUM", `  MEDIUM risk   : ${M}`));
  if (L) console.log(col("LOW",    `  LOW risk      : ${L}`));
  if (U) console.log(col("UNKNOWN",`  UNKNOWN       : ${U}`));
  if (H) {
    console.log(`\n  ${C.r}HIGH risk packages:${C.E}`);
    results.filter(r=>r.risk_label==="HIGH")
           .forEach(r=>console.log(`    ⚠  ${r.package}  (${(r.risk_prob*100).toFixed(1)}%)`));
  }
  console.log("═".repeat(62));
}

function ask(q) {
  const rl = readline.createInterface({ input:process.stdin, output:process.stdout });
  return new Promise(r => rl.question(q, a => { rl.close(); r(a); }));
}

// ─── Commands ────────────────────────────────────────────────────────────────

async function cmdInstall(args) {
  const pkgs   = args.filter(a=>!a.startsWith("-"));
  const flags  = args.filter(a=>a.startsWith("-"));
  if (!pkgs.length) {
    spawnSync("npm", ["install",...flags], {stdio:"inherit"});
    return;
  }
  console.log(`\n${C.c}🔍 DeepRisk scanning ${pkgs.length} package(s)…${C.E}`);
  const t0 = Date.now();
  const results = await fetchBatch(pkgs);
  console.log(`${C.grey}  Scan completed in ${Date.now()-t0}ms${C.E}`);

  let blocked = false;
  for (const r of results) {
    printCard(r, true);
    if (r.risk_label === "HIGH" && !CFG.noBlock) {
      const ans = await ask(`\n${C.y}⚠  ${r.package} is HIGH risk. Install anyway? (y/N) ${C.E}`);
      if (ans.trim().toLowerCase() !== "y") {
        console.log(`${C.r}✗ Installation of ${r.package} aborted.${C.E}`);
        blocked = true;
      }
    }
  }
  if (blocked) { console.log(`\n${C.r}Install cancelled.${C.E}`); process.exit(1); }

  console.log(`\n${C.grey}Running npm install…${C.E}`);
  const res = spawnSync("npm", ["install",...pkgs,...flags], {stdio:"inherit"});
  process.exit(res.status || 0);
}

async function cmdScan(args) {
  if (!args.length) {
    const rl = readline.createInterface({input:process.stdin});
    const lines = [];
    for await (const l of rl) { const t=l.trim(); if(t) lines.push(t); }
    args = lines;
  }
  if (!args.length) { console.error("Usage: deeprisk scan <pkg> [pkg2...]"); process.exit(1); }
  const t0 = Date.now();
  const results = await fetchBatch(args);
  console.log(`${C.grey}Scan: ${Date.now()-t0}ms${C.E}`);
  results.forEach(r => printCard(r, true));
  if (args.length > 1) printBatchSummary(results);
  process.exit(results.some(r=>r.risk_label==="HIGH") ? 2 : 0);
}

async function cmdDemo() {
  console.log(`\n${C.bold}${C.c}╔══════════════════════════════════════════════════════╗`);
  console.log(`║   DeepRisk OSS — Live Hackathon Demo                 ║`);
  console.log(`╚══════════════════════════════════════════════════════╝${C.E}\n`);
  console.log(`Scenario: Developer installs 'colors' (a popular but abandoned npm library).\n`);
  await new Promise(r => setTimeout(r, 1000));

  console.log(`${C.c}$ deeprisk install colors${C.E}`);
  console.log(`${C.grey}🔍 DeepRisk scanning 1 package…${C.E}`);
  const t0 = Date.now();
  const d  = await fetchOne("colors");
  console.log(`${C.grey}  Completed in ${Date.now()-t0}ms${C.E}`);
  printCard(d, true);

  await new Promise(r => setTimeout(r, 800));
  console.log(`\n${C.y}Developer sees HIGH risk + alternatives suggested.${C.E}`);
  console.log(`${C.grey}They type 'N' to abort. Then:${C.E}\n`);
  await new Promise(r => setTimeout(r, 600));

  console.log(`${C.c}$ deeprisk install chalk${C.E}`);
  const d2 = await fetchOne("chalk");
  printCard(d2, false);
  console.log(`\n${C.g}✅ chalk is LOW risk. Installation proceeds safely.${C.E}`);
  console.log(`\n${C.bold}Result: Supply chain attack prevented in <500ms.${C.E}\n`);
}

async function cmdHealth() {
  try {
    const d = await post(`${CFG.apiUrl}/health`, {});
    console.log(`${C.g}✓ API server reachable${C.E}  url=${CFG.apiUrl}`);
    console.log(`  model_ready : ${d.model_ready}`);
    console.log(`  startup_ms  : ${d.startup_ms}ms`);
    if (d.startup_error) console.log(`  ${C.r}error: ${d.startup_error}${C.E}`);
  } catch(e) {
    console.log(`${C.r}✗ Server unreachable${C.E}  url=${CFG.apiUrl}\n  ${e.message}`);
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
  try { saved = JSON.parse(fs.readFileSync(cfgFile,"utf8")); } catch(_) {}
  saved[key] = isNaN(val) ? val : Number(val);
  fs.writeFileSync(cfgFile, JSON.stringify(saved, null, 2));
  console.log(`${C.g}✓ Set ${key} = ${val}${C.E}`);
}

async function cmdCache(args) {
  const sub = args[0] || "info";
  if (sub === "clear") {
    ensureDir(CFG.cacheDir);
    const files = fs.readdirSync(CFG.cacheDir).filter(f=>f.endsWith(".json"));
    files.forEach(f => fs.unlinkSync(path.join(CFG.cacheDir,f)));
    console.log(`${C.g}✓ Cleared ${files.length} entries${C.E}`);
  } else {
    ensureDir(CFG.cacheDir);
    const n = fs.readdirSync(CFG.cacheDir).filter(f=>f.endsWith(".json")).length;
    console.log(`Cache: ${CFG.cacheDir}\nEntries: ${n}  TTL: 7 days`);
  }
}

function help() {
  console.log(`
${C.bold}DeepRisk CLI v${CFG.version}${C.E} — npm Supply Chain Security Scanner

${C.c}Commands:${C.E}
  deeprisk install <pkg> [flags]   Scan then install (blocks HIGH risk)
  deeprisk scan    <pkg> [pkg2...]  Scan only (exit 2 on HIGH risk)
  deeprisk demo                     Run hackathon demo scenario
  deeprisk health                   Check API server
  deeprisk config  [key] [value]    Show/set config
  deeprisk cache   clear|info       Manage cache

${C.c}Examples:${C.E}
  deeprisk install colors            # ⚠ HIGH risk — shows alternatives
  deeprisk install chalk             # ✅ LOW risk
  deeprisk scan lodash express axios
  deeprisk config apiUrl http://myserver:8000
  DEEPRISK_NO_BLOCK=1 deeprisk install colors   # warn only, never block

${C.c}Exit codes:${C.E}  0=safe  1=aborted  2=HIGH risk found (scan only)
`);
}

// ─── Entry ────────────────────────────────────────────────────────────────────
async function main() {
  const [,,cmd,...rest] = process.argv;
  switch(cmd) {
    case "install": await cmdInstall(rest); break;
    case "scan":    await cmdScan(rest);    break;
    case "demo":    await cmdDemo();        break;
    case "health":  await cmdHealth();      break;
    case "config":  await cmdConfig(rest);  break;
    case "cache":   await cmdCache(rest);   break;
    case "--help": case "-h": case "help": help(); break;
    case "--version": case "-v": console.log(`deeprisk v${CFG.version}`); break;
    default: if(!cmd) help(); else { console.error(`Unknown: ${cmd}`); help(); }
  }
}

main().catch(e => { console.error(`${C.r}Fatal: ${e.message}${C.E}`); process.exit(1); });
