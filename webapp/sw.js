// Service worker: caches the app, TF.js and the model so it works offline after the first load.
const CACHE = "krishi-v11";
const TFJS = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
const QR_LIB = "https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.js";
const CORE = ["./", "index.html", "style.css", "i18n.js", "app.js", "manifest.webmanifest",
  "data/labels.json", QR_LIB, "data/diseases.json", "icons/icon-192.png", "icons/icon-512.png", TFJS];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await Promise.all(CORE.map((u) => cache.add(new Request(u, { cache: "reload" })).catch(() => {})));
    try {                                                   // precache model.json + weight shards
      const res = await fetch("model/model.json", { cache: "reload" });
      if (res.ok) {
        await cache.put("model/model.json", res.clone());
        const manifest = (await res.json()).weightsManifest;
        const shards = manifest.flatMap((g) => g.paths);
        await Promise.all(shards.map((p) => cache.add(new Request("model/" + p, { cache: "reload" })).catch(() => {})));
      }
    } catch (err) { /* model not deployed yet */ }
    self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// App files (html/js/css/json): network first, so users always get the newest version, cache when offline.
// Model files and CDN libraries never change: cache first. The photo relay is never touched.
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.hostname === "ntfy.sh" || url.hostname.endsWith("open-meteo.com") || url.hostname.endsWith("bigdatacloud.net") || url.pathname.startsWith("/api/")) return;
  const heavy = url.origin !== location.origin || url.pathname.startsWith("/model/");
  const save = (res) => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return res;
  };
  if (heavy) {
    e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request).then(save)));
  } else {
    e.respondWith(fetch(e.request).then(save).catch(() => caches.match(e.request, { ignoreSearch: true })));
  }
});
