/**
 * Loads the self-hosted OpenCV.js build and resolves once the WebAssembly
 * runtime is initialized (i.e. `cv.Mat` is callable).
 *
 * Why self-hosted instead of the docs.opencv.org CDN:
 *   - OpenCV prunes old pinned versions from its CDN. The previously used
 *     `https://docs.opencv.org/4.10.0/opencv.js` now returns HTTP 404, which
 *     left the app stuck on "Loading OpenCV.js WASM..." forever.
 *   - The CDN is also slow or blocked in some regions.
 * The build ships in `public/opencv/opencv.js` alongside the TF.js models, so
 * it is served from the app's own origin and respects the Vite base path.
 */

declare global {
  interface Window {
    // OpenCV.js attaches its Emscripten module here once the script runs.
    cv?: any;
  }
}

const SCRIPT_ID = "opencv-js";
const OPENCV_SRC = `${import.meta.env.BASE_URL}opencv/opencv.js`;
const INIT_TIMEOUT_MS = 60_000;
const POLL_INTERVAL_MS = 100;

let loadPromise: Promise<void> | null = null;

/** True once the OpenCV WASM runtime has finished initializing. */
function isRuntimeReady(): boolean {
  return typeof window.cv !== "undefined" && typeof window.cv.Mat === "function";
}

/**
 * Load OpenCV.js once and cache the resulting promise. Subsequent calls return
 * the same promise. A failed load clears the cache so a retry can start fresh.
 */
export function loadOpenCV(): Promise<void> {
  if (loadPromise) return loadPromise;

  const attempt = new Promise<void>((resolve, reject) => {
    if (isRuntimeReady()) {
      resolve();
      return;
    }

    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearInterval(poll);
      clearTimeout(timer);
      if (error) reject(error);
      else resolve();
    };

    // Poll as a fallback in case onRuntimeInitialized fired before we wired it.
    const poll = setInterval(() => {
      if (isRuntimeReady()) finish();
    }, POLL_INTERVAL_MS);

    const timer = setTimeout(
      () =>
        finish(
          new Error(
            "OpenCV.js failed to initialize within 60 seconds. Check that " +
              "public/opencv/opencv.js is being served correctly.",
          ),
        ),
      INIT_TIMEOUT_MS,
    );

    // Hook the Emscripten runtime-ready callback (the canonical signal).
    const wireRuntimeCallback = () => {
      if (isRuntimeReady()) {
        finish();
        return;
      }
      const cv = window.cv;
      if (cv && typeof cv === "object") {
        const previous = cv.onRuntimeInitialized;
        cv.onRuntimeInitialized = () => {
          if (typeof previous === "function") previous();
          finish();
        };
      }
    };

    const existing = document.getElementById(
      SCRIPT_ID,
    ) as HTMLScriptElement | null;
    if (existing) {
      wireRuntimeCallback();
      return;
    }

    const script = document.createElement("script");
    script.id = SCRIPT_ID;
    script.async = true;
    script.src = OPENCV_SRC;
    script.onload = wireRuntimeCallback;
    script.onerror = () =>
      finish(new Error(`Failed to load OpenCV.js from ${OPENCV_SRC}`));
    document.body.appendChild(script);
  });

  // Cache the in-flight/successful promise; on failure, clear the cache so a
  // later call can retry the load from scratch.
  loadPromise = attempt.catch((err) => {
    loadPromise = null;
    throw err;
  });

  return loadPromise;
}
