// AI chat for Krishi Daktar (Netlify Function, served at /api/chat).
// The farmer asks a follow-up question about the diagnosis; Gemini answers in short, simple Bangla
// (or English). Key only in the Netlify env var GEMINI_API_KEY. Online only.

const VERSION = "2026-10-19-chat2";
const BUDGET_MS = 8500;                                          // Netlify cuts a function off at about 10 s

const system = (lang, ctx, weather) => `You are "Krishi Daktar", a friendly crop-doctor assistant for small farmers in West Bengal, India.
${weather ? `Local weather now: ${weather}. Use it when it matters.` : ""}
${ctx ? `The farmer's last photo was diagnosed as: crop = ${ctx.crop}; problem = ${ctx.disease}. Use this as context.` : "No photo diagnosis yet."}
Answer ${lang === "bn" ? "in very simple, spoken Bangla (not literary)" : "in simple English"}.
Keep every answer short: at most 5 short lines. Practical, field-level advice (cultural, organic and preventive steps first).
Never give pesticide brand names or doses; when a spray is needed say to use an approved medicine on the advice of the local agriculture officer (Krishi Bibhag).
If you are not sure, or the question is about human/animal health, money or law, say so and refer to the agriculture officer or the right expert.
Only discuss farming, crops, soil, water, weather and pests; politely decline other topics.`;

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
const str = (v, n) => String(v ?? "").slice(0, n);

export default async (req) => {
  const key = Netlify.env.get("GEMINI_API_KEY");
  if (req.method === "GET") return json({ ok: true, version: VERSION, configured: !!key });
  if (req.method !== "POST") return json({ error: "post_only" }, 405);
  if (!key) return json({ error: "not_configured" }, 503);
  const origin = req.headers.get("origin");                      // only our own site may call this
  if (origin && origin !== new URL(req.url).origin) return json({ error: "forbidden" }, 403);

  let b;
  try { b = await req.json(); } catch { return json({ error: "bad_json" }, 400); }
  const question = str(b && b.question, 400).trim();
  if (!question) return json({ error: "empty" }, 400);
  const lang = b.lang === "en" ? "en" : "bn";
  const ctx = b.context && b.context.crop ? { crop: str(b.context.crop, 80), disease: str(b.context.disease, 160) } : null;
  const weather = str(b.weather, 160);
  const history = (Array.isArray(b.history) ? b.history : []).slice(-6)
    .map((h) => ({ role: h.role === "model" ? "model" : "user", parts: [{ text: str(h.text, 600) }] }));

  const models = [...new Set([Netlify.env.get("GEMINI_MODEL"), "gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-3.8-flash"].filter(Boolean))];
  const started = Date.now();
  let res, used;
  outer:
  for (const model of models) {
    for (const thinkingConfig of [{ thinkingLevel: "minimal" }, null]) {
      const left = BUDGET_MS - (Date.now() - started);
      if (left < 1500) break outer;
      const generationConfig = { temperature: 0.4, maxOutputTokens: 600 };
      if (thinkingConfig) generationConfig.thinkingConfig = thinkingConfig;
      try {
        res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: "POST",
          headers: { "x-goog-api-key": key, "content-type": "application/json" },
          signal: AbortSignal.timeout(left > 6500 ? 5000 : left),
          body: JSON.stringify({
            systemInstruction: { parts: [{ text: system(lang, ctx, weather) }] },
            contents: [...history, { role: "user", parts: [{ text: question }] }],
            generationConfig,
          }),
        });
      } catch { res = undefined; continue outer; }               // too slow / unreachable -> next model
      used = model;
      if ([404, 429, 500, 502, 503, 504].includes(res.status)) continue outer;   // retired, busy or broken
      if (res.status === 400 && thinkingConfig) continue;         // thinking option not supported -> retry plain
      break outer;
    }
  }
  if (!res) return json({ error: "upstream_timeout", status: 504, version: VERSION }, 504);
  if (!res.ok) return json({ error: "upstream_error", status: res.status, model: used, version: VERSION }, 502);
  try {
    const data = await res.json();
    const answer = ((data.candidates || [])[0]?.content?.parts || []).map((p) => p.text || "").join("").trim();
    if (!answer) return json({ error: "empty_answer", model: used }, 502);
    return json({ answer: str(answer, 1500), model: used, version: VERSION });
  } catch { return json({ error: "bad_answer", model: used }, 502); }
};

export const config = { path: "/api/chat" };
