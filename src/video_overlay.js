// ====== CONFIG ======
const VIDEO_URL = "/GS_Forest/Red_Pine_1/Red_Pine_shorter_flight_path1.mp4";
const JSON_URL  = "/GS_Forest/Red_Pine_1/measurements_DJI_multipath_path1_tracking.json";
const FPS = 30;                 // your video's fps
const OVERLAY_ALPHA = 0.5;      // overall overlay opacity over the video
const SHOW_MASKS = true;
const SHOW_BOXES = true;
const SHOW_KEYLINES = true;

// ====== UTILS ======
function colorForId(id) {
  const h = ((id ?? 0) * 47) % 360;
  return `hsl(${h} 90% 55%)`;
}

// COCO-style RLE (counts alternating 0-run,1-run; column-major)
function decodeRLEtoMask(rle) {
  const [h, w] = rle.size; // [rows, cols]
  const total = h * w;
  const counts = rle.counts;
  const flat = new Uint8Array(total); // 0/1 per pixel (column-major)
  let v = 0, idx = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i] >>> 0;
    if (idx + c > total) break;
    if (v === 1) flat.fill(1, idx, idx + c);
    idx += c;
    v ^= 1;
  }
  return { data: flat, h, w };
}

// Turn mask into an offscreen canvas with tinted RGBA where mask==1
function rleMaskToCanvas(rle, tintCss = "red", alpha = 1.0) {
  const { data, h, w } = decodeRLEtoMask(rle);

  // parse tint to RGB
  const tmp = document.createElement("canvas");
  tmp.width = 1; tmp.height = 1;
  const tctx = tmp.getContext("2d");
  tctx.fillStyle = tintCss;
  tctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = tctx.getImageData(0, 0, 1, 1).data;

  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d", { willReadFrequently: true });
  const img = octx.createImageData(w, h);
  const A = Math.round(alpha * 255);

  // column-major -> row-major while writing RGBA
  let k = 0;
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const on = data[k++];
      if (on) {
        const idx = (y * w + x) * 4;
        img.data[idx + 0] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = A;
      }
    }
  }
  octx.putImageData(img, 0, 0);
  return off;
}

function denormBox(bn, W, H) {
  const [x, y, w, h] = bn;
  return [x * W, y * H, w * W, h * H];
}

function drawCircle(ctx, x, y, r = 3) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

function pxLabel(ctx, text, x, y) {
  ctx.font = "12px system-ui, sans-serif";
  ctx.textBaseline = "top";
  const m = ctx.measureText(text);
  const pad = 3;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x - pad, y - pad, m.width + pad * 2, 14 + pad * 2);
  ctx.fillStyle = "white";
  ctx.fillText(text, x, y);
}

// Find the nearest available frame index for a given time
function nearestFrameIndex(sortedIdx, guess) {
  // binary search for closest
  let lo = 0, hi = sortedIdx.length - 1;
  if (guess <= sortedIdx[0]) return sortedIdx[0];
  if (guess >= sortedIdx[hi]) return sortedIdx[hi];
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const val = sortedIdx[mid];
    if (val === guess) return val;
    if (val < guess) lo = mid + 1; else hi = mid - 1;
  }
  // lo is first greater than guess, hi is lower than guess
  const a = sortedIdx[hi], b = sortedIdx[lo];
  return (Math.abs(a - guess) <= Math.abs(b - guess)) ? a : b;
}

// ====== MAIN ======
(async function () {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;

  // Prepare hidden <video>
  const video = document.createElement("video");
  video.src = VIDEO_URL;
  video.playsInline = true;
  video.muted = true;                 // helps with autoplay after click
  video.preload = "auto";
  video.crossOrigin = "anonymous";    // safe default if served CORS-enabled

  // Load tracking JSON
  const resp = await fetch(JSON_URL);
  const tracking = await resp.json();
  const frames = tracking.frames || [];
  if (!frames.length) {
    console.warn("No frames in tracking JSON.");
    return;
  }
  const minFrame = Math.min(...frames.map(f => f.frame_index));
  const frameMap = new Map(frames.map(f => [f.frame_index, f]));
  const sortedFrameIdx = frames.map(f => f.frame_index).sort((a,b)=>a-b);

  // Cache decoded mask canvases: key "frameIndex#detIdx"
  const maskCache = new Map();

  // Size canvas to video (or meta) with HiDPI awareness
  function sizeTo(w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // When metadata is ready, size the canvas
  video.addEventListener("loadedmetadata", () => {
    const W = video.videoWidth  || (tracking.meta && tracking.meta.width)  || 640;
    const H = video.videoHeight || (tracking.meta && tracking.meta.height) || 384;
    sizeTo(W, H);
  });

  // Fallback size immediately (before metadata), then adjust later
  {
    const W = (tracking.meta && tracking.meta.width)  || 640;
    const H = (tracking.meta && tracking.meta.height) || 384;
    sizeTo(W, H);
  }

  // Toggle play/pause on canvas click
  canvas.addEventListener("click", async () => {
    if (video.paused) {
      try { await video.play(); } catch (e) { /* ignore */ }
    } else {
      video.pause();
    }
  });

  // Core draw (called for each frame)
  function drawOverlay() {
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;

    // Draw current video frame
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(video, 0, 0, W, H);

    // Compute (nearest) tracking frame for current time
    const guess = Math.round(video.currentTime * FPS) + minFrame;
    const fidx = nearestFrameIndex(sortedFrameIdx, guess);
    const fr = frameMap.get(fidx);
    if (!fr || !Array.isArray(fr.detections)) return;

    // Overlay
    ctx.save();
    ctx.globalAlpha = OVERLAY_ALPHA;

    for (let di = 0; di < fr.detections.length; di++) {
      const det = fr.detections[di];
      const id = det.id ?? det.raw_id ?? -1;
      const tint = colorForId(id);

      // Mask
      if (SHOW_MASKS && det.mask_rle) {
        const key = fidx + "#" + di;
        let mcan = maskCache.get(key);
        if (!mcan) {
          // bake mask with alpha=1.0; we control overall opacity via globalAlpha
          mcan = rleMaskToCanvas(det.mask_rle, tint, 1.0);
          maskCache.set(key, mcan);
        }
        ctx.drawImage(mcan, 0, 0, W, H);
      }

      // BBox + label
      if (SHOW_BOXES && det.bbox_norm) {
        const [x, y, w, h] = denormBox(det.bbox_norm, W, H);
        ctx.strokeStyle = tint;
        ctx.lineWidth = Math.max(1, Math.min(W, H) * 0.0025);
        ctx.strokeRect(x, y, w, h);

        let label = `ID ${id}`;
        if (typeof det.dbh_m === "number") {
          const inches = det.dbh_m * 100 * 0.393701;
          label += ` • DBH ${inches.toFixed(1)}"`;
        }
        pxLabel(ctx, label, x + 2, y + 2);
      }

      // DBH line + keypoints
      if (SHOW_KEYLINES && det.dbh_left_pt && det.dbh_right_pt) {
        const [lx, ly] = det.dbh_left_pt;
        const [rx, ry] = det.dbh_right_pt;
        ctx.strokeStyle = tint;
        ctx.lineWidth = Math.max(1, Math.min(W, H) * 0.003);
        ctx.beginPath();
        ctx.moveTo(lx, ly);
        ctx.lineTo(rx, ry);
        ctx.stroke();

        ctx.fillStyle = tint;
        drawCircle(ctx, lx, ly, 3.5);
        drawCircle(ctx, rx, ry, 3.5);
      }
    }

    ctx.restore();
  }

  // Render loop using requestVideoFrameCallback when available
  function startRVFC() {
    const step = () => {
      drawOverlay();
      if (!video.paused && !video.ended) {
        video.requestVideoFrameCallback(step);
      }
    };
    video.requestVideoFrameCallback(step);
  }

  // RAF fallback (if rVFC not supported)
  let rafId = null;
  function rafLoop() {
    drawOverlay();
    rafId = requestAnimationFrame(rafLoop);
  }
  function startRAF() {
    if (rafId == null) rafLoop();
  }
  function stopRAF() {
    if (rafId != null) cancelAnimationFrame(rafId);
    rafId = null;
  }

  // Wire up play/pause to loops
  video.addEventListener("play", () => {
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype || "requestVideoFrameCallback" in video) {
      startRVFC();
    } else {
      startRAF();
    }
  });
  video.addEventListener("pause", stopRAF);
  video.addEventListener("ended", stopRAF);

  // Kick off: user will click canvas to play
  // (Optional) Autoplay after first user gesture elsewhere could call video.play()
})();