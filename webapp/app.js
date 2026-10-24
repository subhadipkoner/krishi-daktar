// Krishi Daktar: runs the model fully in the browser with TensorFlow.js (no server).
const THRESHOLD = 0.70;          // below this we say "I am not sure"
const IMG_SIZE = 224;            // must match training
const MODEL_URL = "model/model.json";

const SEND_ID = new URLSearchParams(location.search).get("send");   // set when a phone scans the QR
const IS_TOUCH = matchMedia("(pointer: coarse)").matches;          // phone/tablet
let lang = "bn", model = null, labels = [], info = null, lastResult = null;   // lastResult = {top, online}
let aiReady = false;                                                              // is /api/diagnose deployed + configured?
let aiOn = true;                                                                  // online AI check allowed?
try { aiOn = localStorage.getItem("aiOn") !== "0"; } catch (e) {}
const $ = (id) => document.getElementById(id);
const t = (k) => I18N[lang][k];
try { lang = localStorage.getItem("lang") || "bn"; } catch (e) {}

// ---------- language ----------
function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
  $("langBtn").textContent = t("switchTo");
  $("chatInput").placeholder = t("chatPh");
  if (info) $("disclaimer").textContent = info.disclaimer[lang];
  if (lastResult) render();
  if (typeof uiRefresh === "function") uiRefresh();
}
$("langBtn").onclick = () => {
  lang = lang === "bn" ? "en" : "bn";
  try { localStorage.setItem("lang", lang); } catch (e) {}
  applyLang();
};

// ---------- load model + data ----------
function setStatus(msg, isErr) {
  $("status").textContent = msg || "";
  $("status").className = "status" + (isErr ? " err" : "");
}

async function init() {
  applyLang();
  if (SEND_ID) return startSender();
  initUI();                       // phone opened via QR: no model needed
  setStatus(t("loadingModel"));
  try {
    [labels, info] = await Promise.all([
      fetch("data/labels.json").then((r) => r.json()),
      fetch("data/diseases.json").then((r) => r.json()),
    ]);
    applyLang();
    onDataReady();
    model = await tf.loadGraphModel(MODEL_URL);
    tf.tidy(() => model.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])).dataSync()); // warm-up
    setStatus("");
  } catch (e) {
    console.error(e);
    setStatus(t("modelError"), true);
  }
  probeAI();
}

// The online AI only exists when the serverless function is deployed AND has its API key.
// An empty request gets 400 from a configured function (503 = no key, 404/501 = not deployed).
async function probeAI() {
  if (!navigator.onLine) return;
  try {
    const r = await fetch("/api/diagnose", { method: "POST", body: "{}", signal: AbortSignal.timeout(8000) });
    aiReady = r.status === 400;
  } catch (e) { aiReady = false; }
  $("aiRow").hidden = !aiReady;
  showChat();                              // show the switch only when it can work
}

// ---------- prediction ----------
async function classify(file) {
  if (!model) return setStatus(t("modelError"), true);
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    $("preview").src = url; $("previewWrap").hidden = false; $("scanline").hidden = false;
    $("result").hidden = true; go("vScan"); qualityCheck(img);
    setStatus(t("analyzing"));
    await new Promise((r) => setTimeout(r, 30));          // let the UI repaint
    // Test-time augmentation: average 4 views (original, mirrored, two centre zooms).
    // Real phone photos vary more than lab photos; averaging makes the answer steadier.
    const probs = tf.tidy(() => {
      const base = tf.browser.fromPixels(img).toFloat();           // pixels 0..255
      const zoom = (f) => {
        const [h, w] = base.shape, ch = Math.round(h * f), cw = Math.round(w * f);
        return base.slice([Math.floor((h - ch) / 2), Math.floor((w - cw) / 2), 0], [ch, cw, 3]);
      };
      const views = [base, tf.reverse(base, [1]), zoom(0.85), zoom(0.7)]
        .map((v) => tf.image.resizeBilinear(v, [IMG_SIZE, IMG_SIZE]));   // same squash as training
      return model.predict(tf.stack(views)).mean(0).dataSync();         // normalisation is inside the model
    });
    const top = Array.from(probs).map((p, i) => ({ p, id: labels[i].id }))
      .sort((a, b) => b.p - a.p).slice(0, 2);
    const wantOnline = aiOn && aiReady && navigator.onLine;
    const mine = (lastResult = { top, online: wantOnline ? "pending" : "off" });
    resetChat();
    $("scanline").hidden = true; setStatus("");
    render();
    const thumb = makeThumb(img);
    if (wantOnline) {                                        // second opinion from the online AI
      const res = await askOnline(img);
      if (lastResult === mine) { mine.online = res; render(); }   // ignore if a newer photo arrived
    }
    if (lastResult === mine) saveScan(mine, thumb);
  };
  img.onerror = () => {
    $("previewWrap").hidden = true; $("scanline").hidden = true; $("result").hidden = true; lastResult = null; showChat();
    setStatus(t("imageError"), true);
  };
  img.src = url;
}

// ---------- online AI second opinion (hybrid mode) ----------
// Offline: only the on-device model (potato, tomato, rice). Online: the photo is also sent to our
// own serverless function (/api/diagnose) which asks a vision AI. It knows many more crops and
// diseases. Both answers are combined in decide(). Any failure just keeps the on-device answer.
function askOnline(img) {
  const c = document.createElement("canvas");
  const s = Math.min(1, 768 / Math.max(img.width, img.height));
  c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  const image = c.toDataURL("image/jpeg", 0.8).split(",")[1];
  return fetch("/api/diagnose", { method: "POST", body: JSON.stringify({ image, weather: weatherLine() }),
                                  headers: { "content-type": "application/json" }, signal: AbortSignal.timeout(30000) })
    .then(async (r) => {
      if (r.status === 503) return "off";                       // no key configured
      if (r.ok) return r.json();
      const j = await r.json().catch(() => ({}));
      return r.status === 504 || [429, 503].includes(j.status) ? "busy" : "error";   // busy / quota used up
    })
    .catch(() => "error");
}

const OUR_CROPS = ["potato", "tomato", "rice", "sugarcane"];
function cardOfId(id, p) {                                   // curated Bangla/English text for our 16 classes
  const d = info.diseases[id];
  return { crop: info.crops[id.split("_")[0]][lang], name: d.name, cause: d.cause[lang], tips: d.tips[lang],
           healthy: id.endsWith("_healthy"), p };
}
function cardOfOnline(o) {                                   // text written by the online AI
  const pick = (bn, en) => (lang === "bn" ? bn || en : en || bn);
  const part = pick(o.part_bn, o.part_en);
  return { crop: pick(o.crop_bn, o.crop_en) + (part ? " · " + part : ""),
           name: { bn: o.healthy ? "সুস্থ" + (o.part_bn ? " " + o.part_bn : "") : o.disease_bn || o.disease_en,
                  en: o.healthy ? "Healthy" + (o.part_en ? " " + o.part_en.toLowerCase() : "") : o.disease_en || o.disease_bn },
           noteKey: "healthyNoteAny",
           cause: pick(o.cause_bn, o.cause_en), tips: (lang === "bn" && o.tips_bn.length ? o.tips_bn : o.tips_en),
           healthy: o.healthy, p: o.confidence };
}

// Combine the two answers. Returns {card|null, badge, warn, showDevice}
function decideCore({ top, online }) {
  const dev = top[0];
  const o = online && typeof online === "object" ? online : null;
  const devCard = cardOfId(dev.id, dev.p);
  if (!o) {
    const badge = { pending: "badgePending", off: "badgeOffline", error: "badgeError", busy: "badgeBusy" }[online] || "badgeOffline";
    return { card: devCard, badge, showDevice: true };
  }
  if (!o.is_plant) return { card: null, badge: "badgeNotLeaf", showDevice: false };
  const cropKey = (o.crop_en || "").toLowerCase();
  const ourCrop = o.known_class_id || OUR_CROPS.some((c) => cropKey.includes(c));
  if (o.unsure) return { card: devCard, badge: "badgeAiUnsure", showDevice: true };
  if (o.known_class_id === dev.id) {                         // both agree
    return { card: cardOfId(dev.id, Math.max(dev.p, o.confidence)), badge: "badgeAgree", aiCard: true, showDevice: true };
  }
  if (o.known_class_id) {                                    // both know the class but disagree
    const aiWins = o.confidence > dev.p;
    return { card: aiWins ? cardOfId(o.known_class_id, o.confidence) : devCard, badge: "badgeDisagree", warnKey: "disagreeWarn", aiCard: aiWins,
             other: aiWins ? devCard : cardOfId(o.known_class_id, o.confidence), showDevice: true };
  }
  // the AI sees something our model cannot name (other crop, rice blast, healthy rice, new disease)
  return { card: cardOfOnline(o), badge: ourCrop ? "badgeOnlineOnly" : "badgeOtherCrop", warnKey: ourCrop ? null : "otherCropWarn", aiCard: true,
           showDevice: false };
}

function decide(r) {                                        // + extra fields from the online AI (severity, prevention ...)
  const v = decideCore(r);
  const o = r.online && typeof r.online === "object" && r.online.is_plant ? r.online : null;
  if (v.card && o && v.aiCard) {          // extras describe the AI's disease, so show them only on the AI's card
    const pick = (bn, en) => (lang === "bn" ? bn || en : en || bn);
    v.card = { ...v.card, extra: { severity: o.severity || "", spread: pick(o.spread_bn, o.spread_en),
      prevention: (lang === "bn" && o.prevention_bn && o.prevention_bn.length ? o.prevention_bn : o.prevention_en) || [],
      check: pick(o.check_bn, o.check_en) } };
  }
  return v;
}

// ---------- show result ----------
const pct = (p) => Math.round(p * 100);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
function render() {
  if (!lastResult) return;
  const v = decide(lastResult);
  const bar = (p) => `<div class="conf"><span>${t("confidence")}</span><div class="bar"><i style="width:${pct(p)}%"></i></div><b>${pct(p)}%</b></div>`;
  const badge = `<div class="badge">${t(v.badge)}</div>`;
  const warn = v.warnKey ? `<div class="officer">${t(v.warnKey)}</div>` : "";
  const other = v.other ? `<p class="en">${t("otherAnswer")}: ${esc(v.other.name[lang])} (${pct(v.other.p)}%)</p>` : "";
  const top2 = v.showDevice ? `<h3>${t("top2")}</h3><div class="top2">` + lastResult.top.map((x) =>
    `<div><span>${esc(info.diseases[x.id].name[lang])}</span><b>${pct(x.p)}%</b></div>`).join("") + `</div>` : "";

  let html;
  const c = v.card;
  if (!c || c.p < THRESHOLD) {
    html = `<div class="card unsure">${badge}<h2>⚠️ ${t("unsureTitle")}</h2>
      <p>${t(c ? "unsureText" : "notLeafText")}</p>${c ? bar(c.p) : ""}
      <div class="officer">${t("officer")}</div><p class="en">${t("retake")}</p>${top2}</div>`;
  } else {
    const otherName = lang === "bn" ? c.name.en : c.name.bn;                 // show both languages
    html = `<div class="card ${c.healthy ? "healthy" : ""}">${badge}
      <span class="crop-pill">${t("crop")}: ${esc(c.crop)}</span>${sevChip(c)}
      <h2>${c.healthy ? "✅ " : ""}${esc(c.name[lang])}</h2><div class="en">${esc(otherName)}</div>
      ${bar(c.p)}${warn}${other}
      ${c.healthy ? `<p>${t(c.noteKey || "healthyNote")}</p>` : ""}
      ${c.cause ? `<h3>${t("cause")}</h3><p>${esc(c.cause)}</p>` : ""}
      ${c.tips.length ? `<h3>${t("tips")}</h3><ul>${c.tips.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${extrasHtml(c)}${actionsHtml()}
      ${top2}</div>`;
  }
  $("result").innerHTML = html;
  $("result").hidden = false;
  showChat();
  wireResultButtons();
}

// ---------- online AI switch ----------
$("aiToggle").checked = aiOn;
$("aiToggle").onchange = (e) => {
  aiOn = e.target.checked;
  try { localStorage.setItem("aiOn", aiOn ? "1" : "0"); } catch (err) {}
};

// ---------- inputs ----------
["camInput", "galInput"].forEach((id) =>
  $(id).addEventListener("change", (e) => {
    if (e.target.files[0]) classify(e.target.files[0]);
    e.target.value = "";                                    // allow picking the same photo again
  }));


// ---------- computer: "Take photo" shows a QR code, the phone sends the photo ----------
// The photo goes phone -> ntfy.sh (free public relay, plain HTTPS so it works on any network)
// -> this computer. The topic name is a random unguessable id and ntfy deletes the photo after
// about 3 hours. The disease detection itself still runs only in this browser.
const RELAY = "https://ntfy.sh/";
let qrSource = null;
function closeQR() {
  $("qrModal").hidden = true;
  if (qrSource) { qrSource.close(); qrSource = null; }
  if (qrPoll) { clearInterval(qrPoll); qrPoll = null; }
}
let qrPoll = null;
function openQR() {
  $("qrModal").hidden = false;
  $("qrBox").innerHTML = "";
  if (typeof qrcode === "undefined" || !navigator.onLine) {
    $("qrStatus").textContent = t("qrOffline"); return;
  }
  const topic = "kd-" + crypto.randomUUID();                       // unguessable, one per popup
  const url = location.origin + location.pathname + "?send=" + topic;
  const qr = qrcode(0, "M"); qr.addData(url); qr.make();
  $("qrBox").innerHTML = qr.createSvgTag({ cellSize: 6, margin: 2, scalable: true });
  $("qrStatus").textContent = t("qrWaiting");
  let handled = false;
  const receive = async (m) => {
    if (handled || !m || m.event !== "message" || !m.attachment) return;
    handled = true;
    $("qrStatus").textContent = t("qrReceiving");
    try {
      const blob = await (await fetch(m.attachment.url)).blob();
      closeQR();
      classify(new File([blob], "phone.jpg", { type: blob.type || "image/jpeg" }));
    } catch (err) { handled = false; $("qrStatus").textContent = t("qrWaiting"); }
  };
  // two ways to listen, so one flaky connection does not lose the photo:
  qrSource = new EventSource(RELAY + topic + "/sse?since=10m");     // live stream
  qrSource.onmessage = (e) => receive(JSON.parse(e.data));
  qrPoll = setInterval(async () => {                                 // backup: ask every 3 s
    try {
      const txt = await (await fetch(RELAY + topic + "/json?poll=1&since=10m")).text();
      txt.split("\n").filter(Boolean).forEach((l) => receive(JSON.parse(l)));
    } catch (err) { /* try again next time */ }
  }, 3000);
}
$("camLabel").addEventListener("click", (e) => {
  if (!IS_TOUCH) { e.preventDefault(); openQR(); }          // on a computer show the QR code instead
});
$("qrClose").onclick = closeQR;
$("qrFile").onclick = () => { closeQR(); $("galInput").click(); };

// ---------- phone: opened from the QR code, takes a photo and sends it ----------
function startSender() {
  $("app").hidden = true; $("splash").hidden = true;
  $("sender").hidden = false;
  $("sendInput").addEventListener("change", async (e) => {
    const file = e.target.files[0]; e.target.value = "";
    if (file) sendPhoto(file);
  });
}
function shrink(file, max = 1024) {                        // smaller photo = faster upload on mobile data
  return new Promise((resolve, reject) => {
    const img = new Image(), url = URL.createObjectURL(file);
    img.onload = () => {
      const s = Math.min(1, max / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      c.toBlob((b) => resolve(b), "image/jpeg", 0.85);
    };
    img.onerror = reject; img.src = url;
  });
}
async function sendPhoto(file) {
  const say = (k, err) => { $("sendStatus").textContent = t(k); $("sendStatus").className = "status" + (err ? " err" : ""); };
  say("sending");
  let blob;
  try { blob = await shrink(file); } catch (err) { return say("imageError", true); }
  for (let attempt = 1; attempt <= 3; attempt++) {                  // mobile data can drop: retry
    try {
      const res = await fetch(RELAY + SEND_ID, { method: "PUT", body: blob, headers: { Filename: "leaf.jpg" },
                                                 signal: AbortSignal.timeout(30000) });
      if (res.ok) return say("sent");
    } catch (err) { /* retry */ }
    await new Promise((r) => setTimeout(r, 1500 * attempt));
  }
  say("sendFail", true);
}


// ======================= APP UI (home, weather, crops, history, chat, settings) =======================
const CROPS = {
  potato:    { em: "🥔", bn: "আলু",   en: "Potato",    season: { bn: "রবি মরসুম: কার্তিক-অগ্রহায়ণে (নভেম্বর-ডিসেম্বর) লাগান", en: "Rabi season: plant Nov to Dec" } },
  tomato:    { em: "🍅", bn: "টমেটো", en: "Tomato",    season: { bn: "শীতকালীন ফসল: আশ্বিন-পৌষে (অক্টোবর-জানুয়ারি) চারা লাগান", en: "Winter crop: transplant Oct to Jan" } },
  rice:      { em: "🌾", bn: "ধান",   en: "Rice",      season: { bn: "আউশ, আমন ও বোরো: তিন মরসুমে চাষ হয়", en: "Grown in Aus, Aman and Boro seasons" } },
  sugarcane: { em: "🎋", bn: "আখ",    en: "Sugarcane", season: { bn: "ফাল্গুন-চৈত্রে (ফেব্রুয়ারি-মার্চ) সেট লাগানো হয়", en: "Setts are planted in Feb to Mar" } },
};
const DAY_TIPS = [
  { bn: "সকালে খেত ঘুরে দেখুন, নতুন দাগ বা পোকা আগে ধরা পড়ে।", en: "Walk the field in the morning; new spots and pests show up early." },
  { bn: "আক্রান্ত পাতা তুলে দূরে পুড়িয়ে ফেলুন, সুস্থ গাছের পাশে ফেলবেন না।", en: "Remove sick leaves and burn them away from healthy plants." },
  { bn: "গোড়ায় সেচ দিন, পাতা ভিজিয়ে নয়। ভেজা পাতায় রোগ বেশি ছড়ায়।", en: "Water at the base, not on the leaves; wet leaves spread disease." },
  { bn: "বেশি ইউরিয়া দিলে রোগ বাড়ে। সুষম সার দিন।", en: "Too much urea increases disease. Use balanced fertilizer." },
  { bn: "ফসলের পালা বদল করলে মাটির রোগ কমে।", en: "Rotating crops reduces soil-borne disease." },
  { bn: "ওষুধ দেওয়ার আগে কৃষি অফিসারের পরামর্শ নিন।", en: "Ask the agriculture officer before using any medicine." },
];
let W = null, curCrop = "potato", voiceOn = true, deferredInstall = null;
try { voiceOn = localStorage.getItem("voiceOn") !== "0"; } catch (e) {}
const ago = (ts) => new Date(ts).toLocaleDateString(lang === "bn" ? "bn-IN" : "en-IN", { day: "numeric", month: "short" });

function go(id) {
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== id));
  document.querySelectorAll(".tabbar button").forEach((b) => b.classList.toggle("on", b.dataset.go === id));
  window.scrollTo(0, 0);
  if (id === "vChat") showChat();
  if (id === "vFarm") renderFarm();
  if (id === "vHome") renderHome();
}
function initUI() {
  document.querySelectorAll("[data-go]").forEach((b) => (b.onclick = () => go(b.dataset.go)));
  $("sheetClose").onclick = () => ($("sheet").hidden = true);
  $("sheet").onclick = (e) => { if (e.target.id === "sheet") $("sheet").hidden = true; };
  const net = () => { $("netDot").classList.toggle("off", !navigator.onLine); showChat(); };
  addEventListener("online", () => { net(); probeAI(); loadWeather(); }); addEventListener("offline", net); net();
  $("voiceToggle").checked = voiceOn;
  $("voiceToggle").onchange = (e) => { voiceOn = e.target.checked; try { localStorage.setItem("voiceOn", voiceOn ? "1" : "0"); } catch (x) {} if (lastResult) render(); };
  $("clearHist").onclick = () => { try { localStorage.removeItem("kd_hist"); } catch (e) {} renderHome(); renderFarm(); };
  addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferredInstall = e; $("installBtn").hidden = false; });
  $("installBtn").onclick = async () => { if (deferredInstall) { deferredInstall.prompt(); deferredInstall = null; $("installBtn").hidden = true; } };
  setTimeout(() => $("splash").classList.add("hide"), 900);
  loadWeather();
}
function onDataReady() { buildCrops(); renderHome(); }
function uiRefresh() { if (!info) return; renderHome(); buildCrops(); renderFarm(); renderWeather(); showChat(); }

// ----- weather + disease risk (Open-Meteo, no key; only when online) -----
const WMO = (c) => (c === 0 ? "☀️" : c <= 2 ? "🌤️" : c === 3 ? "☁️" : c <= 48 ? "🌫️" : c <= 57 ? "🌦️" : c <= 67 ? "🌧️" : c <= 77 ? "❄️" : c <= 82 ? "🌧️" : "⛈️");
async function loadWeather() {
  if (!navigator.onLine) return renderWeather();
  let lat = 22.57, lon = 88.36;                                         // Kolkata until we know better
  try { const s = JSON.parse(localStorage.getItem("kd_loc") || "null"); if (s) { lat = s.lat; lon = s.lon; } } catch (e) {}
  const pos = await new Promise((res) => navigator.geolocation ? navigator.geolocation.getCurrentPosition((p) => res(p.coords), () => res(null), { timeout: 6000, maximumAge: 3600000 }) : res(null));
  if (pos) { lat = pos.latitude; lon = pos.longitude; try { localStorage.setItem("kd_loc", JSON.stringify({ lat, lon })); } catch (e) {} }
  try {
    const j = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=precipitation_probability_max&timezone=auto&forecast_days=1`, { signal: AbortSignal.timeout(10000) })).json();
    W = { t: Math.round(j.current.temperature_2m), h: j.current.relative_humidity_2m, wind: Math.round(j.current.wind_speed_10m), code: j.current.weather_code, pop: (j.daily.precipitation_probability_max || [0])[0] || 0, place: "" };
    renderWeather();
    const g = await (await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=${lang === "bn" ? "bn" : "en"}`, { signal: AbortSignal.timeout(6000) })).json();
    W.place = g.city || g.locality || g.principalSubdivision || ""; renderWeather();
  } catch (e) { renderWeather(); }
}
function weatherLine() { return W ? `${W.place ? W.place + ", " : ""}${W.t} C, humidity ${W.h}%, rain chance ${W.pop}%, wind ${W.wind} km/h` : ""; }
function diseaseRisks(w) {
  const r = [], cool = w.t >= 10 && w.t <= 24, warm = w.t > 24 && w.t <= 33;
  if (w.h >= 85 && cool) r.push({ lvl: "high", key: "riskLate" }); else if (w.h >= 75 && w.t <= 28) r.push({ lvl: "mid", key: "riskBlightMid" });
  if (w.h >= 80 && warm) r.push({ lvl: "mid", key: "riskRice" });
  if (w.h >= 80 && w.t >= 22) r.push({ lvl: "mid", key: "riskCane" });
  if (w.h < 45 && w.t >= 30) r.push({ lvl: "mid", key: "riskMites" });
  if (w.pop >= 60) r.push({ lvl: "info", key: "riskRain" });
  if (w.wind >= 25) r.push({ lvl: "info", key: "riskWind" });
  return r.length ? r : [{ lvl: "low", key: "riskLow" }];
}
function renderWeather() {
  if (W) {
    $("wTemp").textContent = W.t; $("wIcon").textContent = WMO(W.code); $("wPlace").textContent = W.place || "📍";
    $("wMeta").textContent = `${t("humidity")} ${W.h}% · ${t("rainChance")} ${W.pop}% · ${t("wind")} ${W.wind} km/h`;
  } else { $("wMeta").textContent = t("weatherOff"); }
  $("riskBox").innerHTML = W ? diseaseRisks(W).map((x) => `<div class="risk"><span class="dot ${x.lvl}"></span><span>${t(x.key)}</span></div>`).join("") + `<p class="note">${t("riskNote")}</p>`
                             : `<p class="note">${t("weatherOff")}</p>`;
}

// ----- home / crops -----
function renderHome() {
  $("cropCards").innerHTML = Object.entries(CROPS).map(([k, c]) => `<div class="crop" data-crop="${k}"><div class="em">${c.em}</div><b>${c[lang]}</b><small>${cropDiseases(k).length} ${t("diseasesOf")}</small></div>`).join("");
  document.querySelectorAll("#cropCards .crop").forEach((el) => (el.onclick = () => { curCrop = el.dataset.crop; buildCrops(); go("vCrops"); }));
  const h = hist();
  $("recentBox").innerHTML = h.length ? h.slice(0, 3).map(histItem).join("") : `<div class="card"><p class="note">${t("noScans")}</p></div>`;
  wireHist($("recentBox"));
  const tip = DAY_TIPS[new Date().getDate() % DAY_TIPS.length]; $("dayTip").textContent = tip[lang];
}
const cropDiseases = (k) => (info ? Object.keys(info.diseases).filter((id) => id.startsWith(k + "_") && !id.endsWith("_healthy")) : []);
function buildCrops() {
  if (!info) return;
  $("cropTabs").innerHTML = Object.entries(CROPS).map(([k, c]) => `<button class="tab ${k === curCrop ? "on" : ""}" data-crop="${k}">${c.em} ${c[lang]}</button>`).join("");
  document.querySelectorAll("#cropTabs .tab").forEach((b) => (b.onclick = () => { curCrop = b.dataset.crop; buildCrops(); }));
  const c = CROPS[curCrop];
  $("cropBody").innerHTML = `<div class="card"><h2>${c.em} ${c[lang]}</h2><p>${t("season")}: ${c.season[lang]}</p></div><h3 class="sec">${t("diseasesOf")}</h3>` +
    cropDiseases(curCrop).map((id) => { const d = info.diseases[id]; return `<div class="hitem" data-d="${id}"><div style="font-size:1.8rem">🦠</div><div><b>${esc(d.name[lang])}</b><small>${esc(lang === "bn" ? d.name.en : d.name.bn)}</small></div></div>`; }).join("");
  document.querySelectorAll("#cropBody .hitem").forEach((el) => (el.onclick = () => openDisease(el.dataset.d)));
}
function openDisease(id) {
  const d = info.diseases[id];
  openSheet(`<h2>${esc(d.name[lang])}</h2><p class="en">${esc(lang === "bn" ? d.name.en : d.name.bn)}</p><h3>${t("cause")}</h3><p>${esc(d.cause[lang])}</p><h3>${t("tips")}</h3><ul>${d.tips[lang].map((x) => `<li>${esc(x)}</li>`).join("")}</ul><div class="rowbtns"><button id="askAbout">${t("askAI")}</button></div>`);
  $("askAbout").onclick = () => { $("sheet").hidden = true; go("vChat"); $("chatInput").value = d.name[lang] + ": "; $("chatInput").focus(); };
}
function openSheet(html) { $("sheetBody").innerHTML = html; $("sheet").hidden = false; }

// ----- scan history (stored on this device only) -----
const hist = () => { try { return JSON.parse(localStorage.getItem("kd_hist") || "[]"); } catch (e) { return []; } };
function makeThumb(img) { const c = document.createElement("canvas"); c.width = c.height = 96; const s = Math.max(96 / img.width, 96 / img.height); c.getContext("2d").drawImage(img, (96 - img.width * s) / 2, (96 - img.height * s) / 2, img.width * s, img.height * s); return c.toDataURL("image/jpeg", 0.6); }
function saveScan(r, thumb) {
  const c = decide(r).card; if (!c || c.p < THRESHOLD) return;
  const arr = hist(); arr.unshift({ t: Date.now(), thumb, crop: c.crop, name: c.name, p: c.p, healthy: !!c.healthy, cause: c.cause, tips: c.tips });
  try { localStorage.setItem("kd_hist", JSON.stringify(arr.slice(0, 25))); } catch (e) {}
  renderHome(); renderFarm();
}
const histItem = (h, i) => `<div class="hitem" data-h="${i}"><img src="${h.thumb || ""}" alt=""><div><b>${esc(h.name[lang] || h.name.en)}</b><small>${esc(h.crop)} · ${pct(h.p)}% · ${ago(h.t)}</small></div></div>`;
function wireHist(box) {
  box.querySelectorAll(".hitem").forEach((el) => (el.onclick = () => {
    const h = hist()[+el.dataset.h]; if (!h) return;
    openSheet(`<img src="${h.thumb}" alt="" style="width:96px;height:96px;border-radius:16px"><h2>${esc(h.name[lang] || h.name.en)}</h2><p class="en">${esc(h.crop)} · ${pct(h.p)}% · ${ago(h.t)}</p>${h.cause ? `<h3>${t("cause")}</h3><p>${esc(h.cause)}</p>` : ""}${h.tips && h.tips.length ? `<h3>${t("tips")}</h3><ul>${h.tips.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}`);
  }));
}
function renderFarm() {
  const h = hist(), crops = new Set(h.map((x) => x.crop.split(" ")[0])).size, sick = h.filter((x) => !x.healthy).length;
  $("farmStats").innerHTML = `<div><b>${h.length}</b><small>${t("scansN")}</small></div><div><b>${crops}</b><small>${t("cropsN")}</small></div><div><b>${sick}</b><small>${t("sickN")}</small></div>`;
  $("historyBox").innerHTML = h.length ? h.map(histItem).join("") : `<div class="card"><p class="note">${t("noScans")}</p></div>`;
  wireHist($("historyBox"));
}

// ----- result extras: severity, prevention, listen, share, photo quality -----
function sevChip(c) { const s = c.extra && c.extra.severity; return s ? `<span class="sev ${s}">${t("sevLabel")}: ${t("sev" + s[0].toUpperCase() + s.slice(1))}</span>` : ""; }
function extrasHtml(c) {
  const x = c.extra; if (!x) return "";
  return (x.spread ? `<h3>${t("spreadLabel")}</h3><p>${esc(x.spread)}</p>` : "") +
         (x.prevention.length ? `<h3>${t("preventLabel")}</h3><ul>${x.prevention.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : "") +
         (x.check ? `<h3>${t("checkLabel")}</h3><p>${esc(x.check)}</p>` : "");
}
function actionsHtml() { return `<div class="rowbtns">${voiceOn && "speechSynthesis" in window ? `<button id="btnListen">${t("listen")}</button>` : ""}<button id="btnShare">${t("share")}</button><button id="btnAsk">${t("askAI")}</button></div>`; }
function resultText() { const c = lastResult && decide(lastResult).card; if (!c) return ""; return `${c.crop}. ${c.name[lang]}. ${t("confidence")} ${pct(c.p)}%. ${(c.tips || []).slice(0, 3).join(" ")}`; }
function wireResultButtons() {
  const L = $("btnListen"), S = $("btnShare"), A = $("btnAsk");
  if (L) L.onclick = () => { speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(resultText()); u.lang = lang === "bn" ? "bn-IN" : "en-IN"; speechSynthesis.speak(u); };
  if (S) S.onclick = async () => { const txt = "Krishi Daktar: " + resultText(); try { if (navigator.share) await navigator.share({ text: txt }); else { await navigator.clipboard.writeText(txt); S.textContent = t("copied"); } } catch (e) {} };
  if (A) A.onclick = () => go("vChat");
}
function qualityCheck(img) {
  const c = document.createElement("canvas"), s = 128 / Math.max(img.width, img.height);
  c.width = Math.max(8, Math.round(img.width * s)); c.height = Math.max(8, Math.round(img.height * s));
  const x = c.getContext("2d"); x.drawImage(img, 0, 0, c.width, c.height);
  const d = x.getImageData(0, 0, c.width, c.height).data, n = c.width * c.height, g = new Float32Array(n); let sum = 0;
  for (let i = 0; i < n; i++) { g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]; sum += g[i]; }
  const mean = sum / n; let lv = 0, cnt = 0;
  for (let yy = 1; yy < c.height - 1; yy++) for (let xx = 1; xx < c.width - 1; xx++) { const i = yy * c.width + xx, l = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - c.width] - g[i + c.width]; lv += l * l; cnt++; }
  lv /= cnt;
  const w = [lv < 45 && "qBlur", mean < 50 && "qDark", mean > 225 && "qBright"].filter(Boolean);
  $("quality").innerHTML = w.length ? w.map((k) => `<span class="chip warn">${t(k)}</span>`).join("") : `<span class="chip">${t("qOk")}</span>`;
}

// ----- AI chat (online only): follow-up questions with the scan and the weather as context -----
const CHAT_MAX = 8;                                        // keeps the free AI quota safe
let chatLog = [], chatCount = 0;
function chatContext() { if (!lastResult) return null; const c = decide(lastResult).card; return c ? { crop: c.crop, disease: c.name.en + " / " + c.name.bn } : null; }
function showChat() {
  const on = navigator.onLine && aiReady;
  $("chatState").textContent = t(on ? "chatOn" : "chatOff"); $("chatState").className = "pill" + (on ? "" : " off");
  $("chatOffline").hidden = on;
  const c = chatContext(); $("chatCtx").textContent = c ? t("chatCtx") + ": " + c.crop + " · " + c.disease : t("chatCtxNone");
}
function resetChat() { showChat(); }
function addMsg(who, text) {
  const d = document.createElement("div");
  d.className = "msg " + who; d.textContent = text;
  $("chatLog").appendChild(d); d.scrollIntoView({ block: "nearest" });
  return d;
}
async function ask(q) {
  q = (q || "").trim();
  if (!q) return;
  if (chatCount >= CHAT_MAX) { addMsg("bot", t("chatLimit")); return; }
  chatCount++;
  addMsg("me", q); $("chatInput").value = "";
  const wait = addMsg("bot", t("chatThinking"));
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: q, lang, context: chatContext(), weather: weatherLine(), history: chatLog.slice(-6) }),
      signal: AbortSignal.timeout(20000) });
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.answer) {
      wait.textContent = j.answer;
      chatLog.push({ role: "user", text: q }, { role: "model", text: j.answer });
    } else {
      wait.textContent = t(r.status === 504 || [429, 503].includes(j.status) ? "chatBusy" : "chatError");
    }
  } catch (e) { wait.textContent = t("chatError"); }
}
$("chatForm").onsubmit = (e) => { e.preventDefault(); ask($("chatInput").value); };
document.querySelectorAll(".chip[data-q]").forEach((b) => (b.onclick = () => ask(t(b.dataset.q))));
// voice input (Bangla / English) where the browser supports it
const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Speech) {
  $("micBtn").hidden = false;
  $("micBtn").onclick = () => {
    const rec = new Speech();
    rec.lang = lang === "bn" ? "bn-IN" : "en-IN";
    rec.onresult = (e) => { $("chatInput").value = e.results[0][0].transcript; };
    rec.start();
  };
}

// ---------- PWA: offline support ----------
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});

init();                                                   // start everything once all functions and variables exist
