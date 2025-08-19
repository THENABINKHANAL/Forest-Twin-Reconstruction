const JSON_URL  = "/GS_Forest/Red_Pine_1/measurements_DJI_multipath_path1_tracking.json";

// Load tracking JSON
const resp = await fetch(JSON_URL);
const tracking = await resp.json();

const frames = tracking.frames || [];
if (!frames.length) {
  console.warn("No frames in tracking JSON.");
}

export const frameMap = new Map(frames.map(f => [f.frame_index, f]));

// Decode COCO-style RLE to binary mask
export function decodeRLEtoMask(rle) {
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

// Denormalize bbox to pixel coords
function denormBox(bn, W, H) {
  const [x, y, w, h] = bn;
  return [x * W, y * H, w * W, h * H];
}

// Cache decoded masks
const decodedMaskCache = new Map(); // key: "frameIndex#detIdx"

// Get mask id for given frame & normalized coords (0..1)
export function getMaskIdAtPoint(frameNum, xNorm, yNorm) {
  const fr = frameMap.get(frameNum);
  if (!fr || !Array.isArray(fr.detections)) return null;

  const W = tracking.meta?.width || 640;
  const H = tracking.meta?.height || 384;

  // Convert normalized coords to pixel coords
  const xPx = xNorm * W;
  const yPx = yNorm * H;

  for (let di = 0; di < fr.detections.length; di++) {
    const det = fr.detections[di];
    if (!det?.mask_rle) continue;

    // Optional quick reject: check if point is in bbox
    if (det.bbox_norm) {
      const [bx, by, bw, bh] = denormBox(det.bbox_norm, W, H);
      if (xPx < bx || yPx < by || xPx > bx + bw || yPx > by + bh) continue;
    }

    // Decode mask (cache)
    const key = frameNum + "#" + di;
    let dec = decodedMaskCache.get(key);
    if (!dec) {
      dec = decodeRLEtoMask(det.mask_rle);
      decodedMaskCache.set(key, dec);
    }

    const { h, w, data } = dec;

    // Map from pixel coords to mask coords
    const mx = Math.max(0, Math.min(w - 1, Math.round(xPx * (w / W))));
    const my = Math.max(0, Math.min(h - 1, Math.round(yPx * (h / H))));

    // Column-major index
    if (data[mx * h + my]) {
      return det.id ?? det.raw_id ?? null;
    }
  }
  return null;
}
