/*
 * sw.js - service worker for CardioScope.
 *
 * What it does
 *   - Saves the app shell (page, manifest, icons) so the app opens instantly
 *     and still opens with no connection.
 *   - Keeps the last /api/metrics response so the performance section can be
 *     shown offline.
 *   - Never caches /api/predict: a prediction always comes from the live model.
 *
 * Bump CACHE_VERSION whenever you change index.html or this file, so phones
 * pick up the new version.
 */
const CACHE_VERSION = 'v1';
const CACHE = `cardioscope-${CACHE_VERSION}`;
const SHELL = [
  './',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) =>
      // Add files one by one so a single missing file cannot block the install.
      Promise.all(SHELL.map((url) => cache.add(url).catch(() => {})))
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('cardioscope-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;                       // predictions (POST) go straight to the network
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;            // leave fonts and other sites alone

  // Model results: try the network first, fall back to the last copy.
  if (url.pathname.endsWith('/api/metrics')) {
    event.respondWith(networkFirst(request));
    return;
  }
  // Everything else under /api (health, docs) is always live.
  if (url.pathname.includes('/api/')) return;

  // The page itself: network first so updates show up, cache when offline.
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request, './'));
    return;
  }
  // Icons, manifest and other static files: serve the saved copy, refresh quietly.
  event.respondWith(staleWhileRevalidate(request));
});

async function networkFirst(request, fallbackKey) {
  const cache = await caches.open(CACHE);
  try {
    const fresh = await fetch(request);
    if (fresh.ok) cache.put(fallbackKey || request, fresh.clone());
    return fresh;
  } catch (err) {
    const saved = await cache.match(fallbackKey || request);
    if (saved) return saved;
    throw err;
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE);
  const saved = await cache.match(request);
  const refresh = fetch(request)
    .then((res) => { if (res.ok) cache.put(request, res.clone()); return res; })
    .catch(() => saved);
  return saved || refresh;
}
