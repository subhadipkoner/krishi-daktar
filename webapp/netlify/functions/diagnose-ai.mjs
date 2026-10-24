// Online "second opinion" for Krishi Daktar (Netlify Function, served at /api/diagnose).
// The browser sends a small JPEG; this function asks a vision AI (Google Gemini) to identify the crop and
// its disease (ANY crop, not only the 3 the on-device model knows) and returns plain JSON with
// simple Bangla + English text. The API key lives ONLY in the Netlify environment variable
// GEMINI_API_KEY (never in the page). Without a key it answers 503 and the app simply stays
// in offline (on-device) mode.

// ids of the on-device model; the AI maps to one of them when the leaf is one of those crops.
const KNOWN = [
  "potato_healthy", "potato_early_blight", "potato_late_blight",
  "tomato_healthy", "tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight", "tomato_leaf_mold",
  "tomato_septoria_leaf_spot", "tomato_spider_mites", "tomato_target_spot", "tomato_mosaic_virus",
  "tomato_yellow_leaf_curl_virus", "rice_bacterial_leaf_blight", "rice_brown_spot", "rice_leaf_smut",
  "sugarcane_healthy", "sugarcane_mosaic", "sugarcane_red_rot", "sugarcane_rust", "sugarcane_yellow_leaf",
];

const SYSTEM = `You are an expert plant pathologist helping small farmers in West Bengal, India.
Identify the crop in the photo and the most likely disease or pest problem (or say it is healthy).
The photo may show a leaf, a fruit or vegetable, a stem, a flower, a root or the whole plant; say which part in part_en / part_bn.
is_plant is false ONLY when the photo shows no plant, fruit, vegetable or crop part at all.
You may answer for ANY crop (rice, potato, tomato, brinjal, chilli, jute, mango, banana, wheat, mustard, vegetables ...).
If the leaf is potato, tomato, rice or sugarcane and matches one of these ids, set known_class_id to it, else null:
${KNOWN.join(", ")}
(for rice leaf blast or healthy rice there is no id: use null.)
Rules: be honest and calibrated; if the photo is unclear, not a plant, or you cannot tell, set unsure=true.
Write cause_bn and tips_bn in very simple, spoken Bangla for farmers (not literary). Keep every cause under 30 words and give exactly 3 tips of under 14 words each; cause_en/tips_en the same in English.
Also judge how serious it is (severity low, medium or high) and say what to check next. Keep prevention to 2 short tips and give ONE short question that would help confirm the diagnosis.
If local weather is given, use it in the advice.
Never give pesticide names or doses; say "use an approved medicine on the advice of the agriculture officer (Krishi Bibhag)" when a spray is needed.
Reply with ONLY this JSON, no other text:
{"unsure": false, "is_plant": true, "part_en": "", "part_bn": "", "crop_en": "", "crop_bn": "", "disease_en": "", "disease_bn": "", "healthy": false,
 "known_class_id": null, "confidence": 0.0, "cause_en": "", "cause_bn": "", "tips_en": ["", "", ""], "tips_bn": ["", "", ""],
 "severity": "low", "spread_bn": "", "spread_en": "", "prevention_bn": ["", ""], "prevention_en": ["", ""], "check_bn": "", "check_en": ""}`;

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });

const str = (v, n = 300) => String(v ?? "").slice(0, n);
const list = (v) => (Array.isArray(v) ? v.slice(0, 5).map((x) => str(x, 200)) : []);

const VERSION = "2026-10-19-h";
const BUDGET_MS = 8500;                                          // Netlify cuts a function off at about 10 s

// thinking switched off / minimal: much faster, and enough for leaf photos
const thinkingFor = (model) =>
  /2\.5-flash/.test(model) ? [{ thinkingBudget: 0 }, null] : /gemini-3|latest/.test(model) ? [{ thinkingLevel: "minimal" }, null] : [null];

export default async (req) => {
  const key = Netlify.env.get("GEMINI_API_KEY");
  if (req.method === "GET") return json({ ok: true, version: VERSION, configured: !!key });   // health check, no AI call
  if (req.method !== "POST") return json({ error: "post_only" }, 405);
  if (!key) return json({ error: "not_configured" }, 503);

  const origin = req.headers.get("origin");                      // only our own site may call this
  if (origin && origin !== new URL(req.url).origin) return json({ error: "forbidden" }, 403);

  let body;
  try { body = await req.json(); } catch { return json({ error: "bad_json" }, 400); }
  const image = body && body.image;
  const weather = typeof (body && body.weather) === "string" ? body.weather.slice(0, 160) : "";
  if (typeof image !== "string" || image.length < 100 || image.length > 1_400_000) return json({ error: "bad_image" }, 400);

  // Google retires model names often: try ours in order, 404 = retired -> next one.
  // (Local `netlify dev` only: ?model=... lets us time a single model.)
  const forced = Netlify.env.get("NETLIFY_DEV") === "true" ? new URL(req.url).searchParams.get("model") : null;
  const models = forced ? [forced]
    : [...new Set([Netlify.env.get("GEMINI_MODEL"), "gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-3.8-flash"].filter(Boolean))];
  const started = Date.now();
  let res, used;
  outer:
  for (const model of models) {
    for (const thinkingConfig of thinkingFor(model)) {
      const left = BUDGET_MS - (Date.now() - started);
      if (left < 1500) break outer;
      const generationConfig = { temperature: 0.2, maxOutputTokens: 1500, responseMimeType: "application/json" };
      if (thinkingConfig) generationConfig.thinkingConfig = thinkingConfig;
      try {
        res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: "POST",
          headers: { "x-goog-api-key": key, "content-type": "application/json" },
          signal: AbortSignal.timeout(left > 6500 ? 5000 : left),     // a slow model must not eat the whole budget
          body: JSON.stringify({
            systemInstruction: { parts: [{ text: SYSTEM }] },
            contents: [{ role: "user", parts: [
              { inlineData: { mimeType: "image/jpeg", data: image } },
              { text: "Identify the crop and its disease from this photo." + (weather ? ` Local weather now: ${weather}.` : "") }] }],
            generationConfig,
          }),
        });
      } catch (e) { res = undefined; continue outer; }               // too slow / unreachable -> next model
      used = model;
      if ([404, 429, 500, 502, 503, 504].includes(res.status)) continue outer;   // retired, busy or broken -> next model
      if (res.status === 400 && thinkingConfig) continue;         // thinking option not supported -> retry plain
      break outer;
    }
  }
  if (!res) return json({ error: "upstream_timeout", version: VERSION, ms: Date.now() - started }, 504);
  if (!res.ok) {                                                  // pass Google's reason on (it never contains our key)
    let message = "";
    try { message = (await res.json()).error?.message || ""; } catch { /* ignore */ }
    return json({ error: "upstream_error", status: res.status, model: used, version: VERSION, message: str(message, 200) }, 502);
  }

  try {
    const data = await res.json();
    const text = ((data.candidates || [])[0]?.content?.parts || []).map((p) => p.text || "").join("");
    const o = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
    return json({
      unsure: !!o.unsure, is_plant: o.is_plant !== false && o.is_leaf !== false, part_en: str(o.part_en, 30), part_bn: str(o.part_bn, 30), healthy: !!o.healthy,
      known_class_id: KNOWN.includes(o.known_class_id) ? o.known_class_id : null,
      confidence: Math.max(0, Math.min(1, Number(o.confidence) || 0)),
      crop_en: str(o.crop_en, 60), crop_bn: str(o.crop_bn, 60),
      disease_en: str(o.disease_en, 100), disease_bn: str(o.disease_bn, 100),
      cause_en: str(o.cause_en), cause_bn: str(o.cause_bn),
      tips_en: list(o.tips_en), tips_bn: list(o.tips_bn),
      severity: ["low", "medium", "high"].includes(o.severity) ? o.severity : "",
      spread_en: str(o.spread_en, 160), spread_bn: str(o.spread_bn, 160),
      prevention_en: list(o.prevention_en).slice(0, 3), prevention_bn: list(o.prevention_bn).slice(0, 3),
      check_en: str(o.check_en, 200), check_bn: str(o.check_bn, 200), model: used, version: VERSION, ms: Date.now() - started,
    });
  } catch { return json({ error: "bad_answer", model: used, version: VERSION }, 502); }
};

export const config = { path: "/api/diagnose" };
