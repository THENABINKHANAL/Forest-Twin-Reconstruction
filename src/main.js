import * as THREE from 'three';
import seedrandom from 'seedrandom';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Delaunator from 'delaunator';
import { GUI } from 'dat.gui';

// — Scene, camera, renderer setup —
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
const scene = new THREE.Scene();

const imgWidth = 1251;
const imgHeight = 703;
// your intrinsics:
const fx = 1059.2537429779845;
const fy = 1030.2564510087936;
const cx = 625.5;
const cy = 351.5;

// Hover picking
const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();
let hoveredTree = null; // the Group you built in createProceduralTree()

const video = document.getElementById("mainVideo");
let videoFPS = 30; // fallback

// derive vertical FOV from fy:
const fov = 2 * Math.atan(imgHeight / (2 * fy)) * THREE.MathUtils.RAD2DEG;
// aspect should match your image:
const aspect = imgWidth / imgHeight;

const camera = new THREE.PerspectiveCamera(fov, aspect, 0.1, 1000);
// shift the projection center to (cx, cy):
//   setViewOffset(fullWidth, fullHeight, offsetX, offsetY, width, height)
camera.setViewOffset(
    imgWidth, imgHeight,
    cx - imgWidth / 2,
    cy - imgHeight / 2,
    imgWidth, imgHeight
);

camera.position.set(0, 2, 10);
let groundMesh = null;   // plane

// — Orbit controls —
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// — GUI & params —
const gui = new GUI();
const params = {
    maxIterations: 40000,
    clusterThreshold: 0.8,
    minClusterSize: 10,
    showGround: true,
    showVegetation: true,
    minTreeHeight: 0.1,
    mergeClusterThreshold: 0.8,
    showPoints: true,
    renderCameraPoints: true,
    showImage: false,
    showAllPoints: true,
    radiusSigmaK: 1.0,//2.0,
    elevationBand: 0.3048, // ± band around mean camera elevation (meters ~ 1 ft)
    showElevationBandPoints: false

};

// — Seeded RNG for deterministic RANSAC —
const rng = seedrandom('fixed-seed');

// — Data placeholders —
let posArray, pointCount, groundSet, vegPointsMesh;
let cameraPoints = [], camIndex = 0, clusterData = [], treeMeshes = [];
const prevCamPos = new THREE.Vector3();
// — 1) Ground segmentation + mesh creation —
function segmentGround() {
    // 1) sample random triangle normals
    const normals = [];
    for (let i = 0; i < params.maxIterations; i++) {
        const picks = new Set();
        while (picks.size < 3) picks.add(Math.floor(Math.random() * pointCount));
        const [i1, i2, i3] = [...picks];
        const p1 = new THREE.Vector3().fromArray(posArray, i1 * 3);
        const p2 = new THREE.Vector3().fromArray(posArray, i2 * 3);
        const p3 = new THREE.Vector3().fromArray(posArray, i3 * 3);
        normals.push(
            new THREE.Vector3().crossVectors(
                p2.clone().sub(p1), p3.clone().sub(p1)
            ).normalize()
        );
    }

    // 2) dominant normal
    let bestNormal = normals[0], bestScore = -Infinity;
    normals.forEach(nC => {
        const score = normals.reduce((sum, n) => sum + Math.abs(nC.dot(n)), 0);
        if (score > bestScore) [bestNormal, bestScore] = [nC, score];
    });

    // 3) centroid & plane offset
    const centroid = new THREE.Vector3();
    for (let i = 0; i < pointCount; i++) centroid.add(new THREE.Vector3().fromArray(posArray, i * 3));
    centroid.divideScalar(pointCount);
    const d = -bestNormal.dot(centroid);

    // 4) basis vectors for mesh
    const e1 = (Math.abs(bestNormal.x) < 0.9
        ? bestNormal.clone().cross(new THREE.Vector3(1, 0, 0))
        : bestNormal.clone().cross(new THREE.Vector3(0, 1, 0))
    ).normalize();
    const e2 = bestNormal.clone().cross(e1).normalize();

    // 5) project centroid corners
    const projUs = [], projVs = [];
    for (let i = 0; i < pointCount; i++) {
        const P = new THREE.Vector3().fromArray(posArray, i * 3);
        const proj = P.clone().sub(bestNormal.clone().multiplyScalar(bestNormal.dot(P) + d));
        const v = proj.clone().sub(centroid);
        projUs.push(v.dot(e1)); projVs.push(v.dot(e2));
    }

    // 6) build rectangular mesh
    let minU = Infinity, maxU = -Infinity;
    let minV = Infinity, maxV = -Infinity;
    
    for (let i = 0; i < projUs.length; i++) {
        const u = projUs[i];
        const v = projVs[i];
    
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
    
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
    }    
    const corners = [
        centroid.clone().add(e1.clone().multiplyScalar(minU)).add(e2.clone().multiplyScalar(minV)),
        centroid.clone().add(e1.clone().multiplyScalar(maxU)).add(e2.clone().multiplyScalar(minV)),
        centroid.clone().add(e1.clone().multiplyScalar(maxU)).add(e2.clone().multiplyScalar(maxV)),
        centroid.clone().add(e1.clone().multiplyScalar(minU)).add(e2.clone().multiplyScalar(maxV))
    ];
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(
        corners.flatMap(c => c.toArray()), 3
    ));
    geom.setIndex([0, 1, 2, 0, 2, 3]); geom.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
        color: 0x228822, side: THREE.DoubleSide,
        transparent: true, opacity: 0.4
    });
    if (groundMesh) scene.remove(groundMesh);
    groundMesh = new THREE.Mesh(geom, mat);
    groundMesh.visible = params.showGround;
    //scene.add(groundMesh);
}

function gatherRaycastables(groups) {
  const list = [];
  groups.forEach(g => g.traverse(obj => {
    if (obj.isMesh) list.push(obj);
  }));
  return list;
}

// --- Hover tooltip ---
const tooltip = document.createElement('div');
tooltip.style.position = 'fixed';
tooltip.style.pointerEvents = 'none';
tooltip.style.padding = '6px 10px';
tooltip.style.borderRadius = '8px';
tooltip.style.background = 'rgba(0,0,0,0.75)';
tooltip.style.color = '#fff';
tooltip.style.font = '12px/1.2 system-ui, sans-serif';
tooltip.style.whiteSpace = 'pre';
tooltip.style.zIndex = '9999';
tooltip.style.transform = 'translate(12px, 12px)'; // small offset from cursor
tooltip.style.opacity = '0';
document.body.appendChild(tooltip);

function showTooltip(html, clientX, clientY) {
  tooltip.innerHTML = html;
  tooltip.style.left = clientX + 'px';
  tooltip.style.top  = clientY + 'px';
  tooltip.style.opacity = '1';
}
function hideTooltip() {
  tooltip.style.opacity = '0';
}

function onTreeHover(treeGroup, intersect, clientX, clientY) {
  const stat = treeGroup?.userData?.treeStat;
  if (!stat) return;

  // Format position & height
  const px = stat.centerX.toFixed(3);
  const py = stat.centerY.toFixed(3);
  const pz = stat.centerZ.toFixed(3);
  const h  = (stat.height ?? stat.heigth ?? 0).toFixed(3); // tolerate "heigth" typo if present
  const r  = (stat.radius ?? stat.radius ?? 0).toFixed(3); // tolerate "heigth" typo if present

  const html = `
<b>Tree</b><br/>
pos: (${px}, ${py}, ${pz})<br/>
height: ${h}<br/>
radius: ${r}
  `.trim();

  showTooltip(html, clientX, clientY);
}


// ensure these are defined at top-level:
let centerCubes = [];

function distanceAlongDirection(p1, p2, direction) {
    const u = direction.clone().normalize();
    return p2.clone().sub(p1).dot(u);
}

function findTreeRoot(obj) {
  // climb up until the Group that represents the tree
  while (obj && obj.parent && obj.parent !== scene && !treeMeshes.includes(obj)) {
    obj = obj.parent;
  }
  // if the mesh was inside a Group that we pushed into treeMeshes, return that Group
  if (treeMeshes.includes(obj)) return obj;
  return null;
}

function handlePointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouseNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNDC.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouseNDC, camera);

  const hits = raycaster.intersectObjects(gatherRaycastables(treeMeshes), true);
  if (hits.length > 0) {
    const hit = hits[0];
    const tree = findTreeRoot(hit.object);

    if (tree) {
      // highlight logic (optional)...
      if (tree !== hoveredTree) {
        if (hoveredTree) hoveredTree.traverse(o => {
          if (o.isMesh && o.material?.emissive) o.material.emissive.setScalar(0);
        });
        hoveredTree = tree;
        hoveredTree.traverse(o => {
          if (o.isMesh && o.material?.emissive) o.material.emissive.setScalar(0.2);
        });
        renderer.domElement.style.cursor = 'pointer';
      }

      // always update tooltip position/content while hovering
      onTreeHover(tree, hit, event.clientX, event.clientY);
      return; // keep tooltip visible
    }
  }

  // no hit: clear highlight & hide tooltip
  if (hoveredTree) {
    hoveredTree.traverse(o => {
      if (o.isMesh && o.material?.emissive) o.material.emissive.setScalar(0);
    });
    hoveredTree = null;
    renderer.domElement.style.cursor = 'default';
  }
  hideTooltip();
}


renderer.domElement.addEventListener('pointermove', handlePointerMove);
renderer.domElement.addEventListener('mouseleave', hideTooltip);



// — Outlier-resistant sizing via iterative sigma clipping —
function computeRobustRadius(radialDistances) {
    if (!radialDistances || radialDistances.length === 0) return 0;
    // Start with all values, iteratively remove values beyond mean + k*std
    // A few iterations converge to a stable core without harsh percentile cuts.
    let values = radialDistances.slice();
    const maxIterations = 5;
    for (let iter = 0; iter < maxIterations; iter++) {
        const n = values.length;
        if (n === 0) break;
        const mean = values.reduce((s, v) => s + v, 0) / n;
        const variance = values.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n;
        const std = Math.sqrt(Math.max(variance, 0));
        const cutoff = mean + params.radiusSigmaK * std;
        const filtered = values.filter(v => v <= cutoff);
        if (filtered.length === values.length) {
            // converged
            values = filtered;
            break;
        }
        // if we over-trimmed to empty, keep previous
        if (filtered.length === 0) break;
        values = filtered;
    }
    if (values.length === 0) return 0;
    // Use a high-end representative without taking the single max (take average of top few)
    values.sort((a, b) => a - b);
    const take = Math.max(1, Math.floor(values.length * 0.05));
    const tail = values.slice(-take);
    const radius = tail.reduce((s, v) => s + v, 0) / tail.length;
    return radius;
}

// — Algebraic circle fit (Taubin-like) on 2D points —
// Returns { cx, cy, r } or null on failure.
function fitCircle2D(points2) {
    // points2: array of [u, v]
    const n = points2.length;
    if (n < 3) return null;
    // compute means
    let meanU = 0, meanV = 0;
    for (let i = 0; i < n; i++) { meanU += points2[i][0]; meanV += points2[i][1]; }
    meanU /= n; meanV /= n;
    // shift to mean
    let Suu = 0, Suv = 0, Svv = 0, Suuu = 0, Svvv = 0, Suvv = 0, Svuu = 0;
    for (let i = 0; i < n; i++) {
        const ui = points2[i][0] - meanU;
        const vi = points2[i][1] - meanV;
        const ui2 = ui * ui, vi2 = vi * vi;
        Suu += ui2;
        Svv += vi2;
        Suv += ui * vi;
        Suuu += ui2 * ui;
        Svvv += vi2 * vi;
        Suvv += ui * vi2;
        Svuu += vi * ui2;
    }
    const A = [[Suu, Suv], [Suv, Svv]];
    const B = [0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu)];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (Math.abs(det) < 1e-12) return null;
    const invA = [
        [ A[1][1] / det, -A[0][1] / det ],
        [ -A[1][0] / det, A[0][0] / det ]
    ];
    const uc = invA[0][0] * B[0] + invA[0][1] * B[1];
    const vc = invA[1][0] * B[0] + invA[1][1] * B[1];
    const cx = uc + meanU;
    const cy = vc + meanV;
    // radius
    let r = 0;
    for (let i = 0; i < n; i++) {
        const du = points2[i][0] - cx;
        const dv = points2[i][1] - cy;
        r += Math.sqrt(du * du + dv * dv);
    }
    r /= n;
    return { cx, cy, r };
}

function createProceduralTree(levels, length, radius, pos, dir) {
    const tree = new THREE.Group();

    // 1) create this segment
    const geom = new THREE.CylinderGeometry(radius * 0.4, radius, length, 8);
    //   cylinder is centered on origin along Y, so shift it up by half-length
    geom.translate(0, length / 2, 0);
    const mat = new THREE.MeshStandardMaterial({
        color: 0x8b5a2b,
        transparent: true,
        opacity: 0.8
    });
    const segment = new THREE.Mesh(geom, mat);

    // 2) orient + position
    // default cylinder points +Y; compute quaternion from Y to dir
    const quat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.clone().normalize()
    );
    segment.applyQuaternion(quat);
    segment.position.copy(pos);

    segment.raycast = THREE.Mesh.prototype.raycast; // (default; usually not needed)
    segment.frustumCulled = false; // optional if you see misses due to culling

    tree.add(segment);

    if (levels > 0) {
        // 3) compute new branch parameters
        const newLength = length * (0.7 + Math.random() * 0.1);
        const newRadius = radius * 0.4;

        // 4) endpoints for this segment
        const end = pos.clone()
            .add(dir.clone().multiplyScalar(length));

        // 5) spawn two child branches at random angles
        for (let i = 0; i < 2; i++) {
            // pick an angle off the dir vector
            const axis = new THREE.Vector3(
                Math.random() - 0.5,
                Math.random(),
                Math.random() - 0.5
            ).normalize();
            const angle = (Math.PI / 4) + (Math.random() * Math.PI / 8);
            const childDir = dir.clone().applyAxisAngle(axis, angle).normalize();

            const child = createProceduralTree(
                levels - 1,
                newLength,
                newRadius,
                end,
                childDir
            );
            tree.add(child);
        }
    }

    return tree;
}
// ---------- small numeric helpers (keep once in the file) ----------
function sqr(x){ return x*x; }
function dist3(a,b){ return Math.hypot(a.x-b.x, a.y-b.y, a.z-b.z); }

function powerIterationSym3(cov, iters=32) {
  // cov is 3x3 symmetric matrix in row-major array [m00,m01,m02, m10,m11,m12, m20,m21,m22]
  // returns dominant eigenvector (unit)
  let v = new THREE.Vector3(1,0,0).normalize();
  for (let k=0;k<iters;k++){
    const x = cov[0]*v.x + cov[1]*v.y + cov[2]*v.z;
    const y = cov[3]*v.x + cov[4]*v.y + cov[5]*v.z;
    const z = cov[6]*v.x + cov[7]*v.y + cov[8]*v.z;
    v.set(x,y,z).normalize();
    if (!isFinite(v.x+v.y+v.z)) break;
  }
  return v;
}

function covariance3(points) {
  // points: array of {x,y,z}
  const n = points.length;
  if (n < 2) return [1,0,0, 0,1,0, 0,0,1];

  let mx=0,my=0,mz=0;
  for (const p of points){ mx+=p.x; my+=p.y; mz+=p.z; }
  mx/=n; my/=n; mz/=n;

  let sxx=0, sxy=0, sxz=0, syy=0, syz=0, szz=0;
  for (const p of points){
    const dx=p.x-mx, dy=p.y-my, dz=p.z-mz;
    sxx += dx*dx; sxy += dx*dy; sxz += dx*dz;
    syy += dy*dy; syz += dy*dz; szz += dz*dz;
  }
  // unbiased divisor n-1 to be closer to numpy.cov(rowvar=False)
  const d = (n>1) ? (n-1) : 1;
  sxx/=d; sxy/=d; sxz/=d; syy/=d; syz/=d; szz/=d;

  // sym:
  return [
    sxx, sxy, sxz,
    sxy, syy, syz,
    sxz, syz, szz
  ];
}

// Basic (brute-force) DBSCAN in 3D for moderate N
function dbscan3D(points, eps, minPts) {
  const n = points.length;
  const labels = new Array(n).fill(-99); // -99=unvisited, -1=noise, >=0 cluster id
  let cid = 0;

  function regionQuery(i){
    const nbrs = [];
    const pi = points[i];
    for (let j=0;j<n;j++){
      if (i===j) continue;
      if (dist3(pi, points[j]) <= eps) nbrs.push(j);
    }
    return nbrs;
  }

  for (let i=0;i<n;i++){
    if (labels[i] !== -99) continue;
    const nbrs = regionQuery(i);
    if (nbrs.length+1 < minPts) { labels[i] = -1; continue; } // noise
    // start new cluster
    labels[i] = cid;
    const seed = nbrs.slice();
    for (let k=0;k<seed.length;k++){
      const j = seed[k];
      if (labels[j] === -1) labels[j] = cid;
      if (labels[j] !== -99) continue;
      labels[j] = cid;
      const nbrs2 = regionQuery(j);
      if (nbrs2.length+1 >= minPts) {
        // expand
        for (const m of nbrs2) if (!seed.includes(m)) seed.push(m);
      }
    }
    cid++;
  }
  return labels;
}

function toPointArrayFromPosArray(posArray) {
  const pts = new Array(posArray.length/3);
  for (let i=0;i<pts.length;i++){
    pts[i] = { x: posArray[3*i], y: posArray[3*i+1], z: posArray[3*i+2], _i: i };
  }
  return pts;
}
// -------------------------------------------------------------------

function clusterTrees() {
  if (!groundMesh) segmentGround();

  // ------ parameters (mirroring Python; fallbacks if not in GUI) ------
  const eps             = params.dbscanEps ?? 1.0;          // DBSCAN radius (meters)
  const minPtsDBSCAN    = params.minPointsDBSCAN ?? 5;      // DBSCAN minPts
  const minPtsFilter    = params.minPointsFilter ?? Math.max(20, params.minClusterSize||20);
  const gravityMinScore = params.gravityScoreMin ?? 0.6;    // |dot(mainAxis, +Y)|
  const minTreeHeight   = params.minTreeHeight ?? 1.0;       // meters
  const gridSize        = params.groundGridSize ?? 0.05;     // meters
  const deltaZ          = params.deltaZGround ?? 0.15;       // elevation-map margin above ground

  // ------ clear prior visuals ------
  centerCubes.forEach(c => scene.remove(c));
  centerCubes = [];
  if (vegPointsMesh) { scene.remove(vegPointsMesh); vegPointsMesh = null; }
  treeMeshes.forEach(t => scene.remove(t));
  treeMeshes = [];
  treeStats = [];
  if (elevationBandPointsMesh) { scene.remove(elevationBandPointsMesh); elevationBandPointsMesh = null; }

  // ------ build point list ------
  const allPts = toPointArrayFromPosArray(posArray);
  const N0 = allPts.length;
  if (!N0) return;

  // (1) Remove farthest 1/3 from origin (like Python keep_ratio=2/3)
  allPts.forEach(p => p._d = Math.hypot(p.x, p.y, p.z));
  allPts.sort((a,b)=>a._d-b._d);
  const keepCount = Math.floor(allPts.length * (2/3));
  const nearPts = allPts.slice(0, Math.max(keepCount, 1));

  // (2) Elevation-map ground removal (y is up in this JS scene)
  let minX=Infinity,minZ=Infinity,maxX=-Infinity,maxZ=-Infinity;
  for (const p of nearPts){ if (p.x<minX)minX=p.x; if (p.z<minZ)minZ=p.z; if (p.x>maxX)maxX=p.x; if (p.z>maxZ)maxZ=p.z; }
  const gw = Math.max(1, Math.floor((maxX-minX)/gridSize)+1);
  const gh = Math.max(1, Math.floor((maxZ-minZ)/gridSize)+1);
  const elev = new Float32Array(gw*gh).fill(Number.POSITIVE_INFINITY);

  function gi(x,z){ 
    const ix = Math.floor((x-minX)/gridSize);
    const iz = Math.floor((z-minZ)/gridSize);
    return iz*gw + ix;
  }

  // min y per cell
  for (const p of nearPts){
    const idx = gi(p.x,p.z);
    if (idx<0 || idx>=elev.length) continue;
    if (p.y < elev[idx]) elev[idx] = p.y;
  }
  // replace +inf (empty) by max of finite
  let maxFinite = -Infinity;
  for (let i=0;i<elev.length;i++){ if (isFinite(elev[i]) && elev[i]>maxFinite) maxFinite = elev[i]; }
  for (let i=0;i<elev.length;i++){ if (!isFinite(elev[i])) elev[i] = maxFinite; }

  // 3x3 median filter on grid
  const elevFilt = new Float32Array(elev.length);
  for (let iz=0; iz<gh; iz++){
    for (let ix=0; ix<gw; ix++){
      const vals = [];
      for (let dz=-1; dz<=1; dz++){
        for (let dx=-1; dx<=1; dx++){
          const x2 = ix+dx, z2 = iz+dz;
          if (x2>=0 && x2<gw && z2>=0 && z2<gh) vals.push(elev[z2*gw+x2]);
        }
      }
      vals.sort((a,b)=>a-b);
      elevFilt[iz*gw+ix] = vals[Math.floor(vals.length/2)];
    }
  }

  // keep points whose y > ground + deltaZ
  const noGround = [];
  for (const p of nearPts){
    const idx = gi(p.x,p.z);
    if (p.y > elevFilt[idx] + deltaZ) noGround.push(p);
  }
  if (noGround.length === 0) return;

  // (3) DBSCAN clustering in 3D
  const labels = dbscan3D(noGround, eps, minPtsDBSCAN);

  // regroup by label
  const byLabel = new Map();
  for (let i=0;i<labels.length;i++){
    const lab = labels[i];
    if (lab < 0) continue; // ignore noise
    if (!byLabel.has(lab)) byLabel.set(lab, []);
    byLabel.get(lab).push(noGround[i]);
  }

  // (4) PCA verticality & height filter (like Python filter_clusters)
  const finalClusters = [];
  for (const [lab, pts] of byLabel.entries()){
    if (pts.length < minPtsFilter) continue;

    // PCA main axis:
    const cov = covariance3(pts);
    const main = powerIterationSym3(cov, 40); // dominant eigenvector
    const gravityScore = Math.abs(main.dot(new THREE.Vector3(0,1,0)));

    if (gravityScore < gravityMinScore) continue;

    // height (y-range)
    let minY=Infinity, maxY=-Infinity;
    for (const p of pts){ if (p.y<minY)minY=p.y; if (p.y>maxY)maxY=p.y; }
    const height = maxY - minY;
    if (height < minTreeHeight) continue;

    finalClusters.push({ lab, pts, meta:{ score:gravityScore, height } });
  }

  // (5) Build visuals & stats (similar to your previous implementation)
  const vegPos = [];
  const vegCols = [];
  const rng = Math.random; // (you already import seedrandom if you want determinism)

  for (const cl of finalClusters){
    // centroid
    const cen = cl.pts.reduce((s,p)=> (s.x+=p.x, s.y+=p.y, s.z+=p.z, s), {x:0,y:0,z:0});
    cen.x/=cl.pts.length; cen.y/=cl.pts.length; cen.z/=cl.pts.length;

    // choose color
    const col = new THREE.Color(rng(), rng(), rng());

    // push colored points
    for (const p of cl.pts){
      vegPos.push(p.x, p.y, p.z);
      vegCols.push(col.r, col.g, col.b);
    }

    // robust radius in XZ (fallback approach from your JS)
    const radialDistances = cl.pts.map(p => Math.hypot(p.x - cen.x, p.z - cen.z));
    const fittedRadius = computeRobustRadius(radialDistances);

    // tree height from cluster y-range we already computed
    const height = cl.meta.height;

    // place a cube marker at centroid
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(0.2,0.2,0.2),
      new THREE.MeshStandardMaterial({ color: col })
    );
    cube.position.set(cen.x, cen.y, cen.z);
    if (params.showPoints) { scene.add(cube); centerCubes.push(cube); }

    // tree base at minY
    let minY=Infinity;
    for (const p of cl.pts) if (p.y<minY) minY=p.y;
    const bottomPos = new THREE.Vector3(cen.x, minY, cen.z);

    const tree = createProceduralTree(
      /*levels=*/3,
      /*length=*/height,
      /*radius=*/fittedRadius,
      bottomPos,
      new THREE.Vector3(0,1,0)
    );
    tree.userData.treeStat = {
      centerX: bottomPos.x,
      centerY: bottomPos.y,
      centerZ: bottomPos.z,
      radius: fittedRadius,
      diameter: 2*fittedRadius,
      height: height,
      points: cl.pts.length,
      gravityScore: cl.meta.score
    };
    treeMeshes.push(tree);
    scene.add(tree);
    treeStats.push(tree.userData.treeStat);
  }

  // (6) render clustered points
  if (params.showPoints && vegPos.length){
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(vegPos, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(vegCols, 3));
    vegPointsMesh = new THREE.Points(
      geom,
      new THREE.PointsMaterial({ size: 0.05, vertexColors: true })
    );
    vegPointsMesh.visible = params.showVegetation;
    scene.add(vegPointsMesh);
  }

  // refresh ground context overlay
  showGroundPoint();
}

function laplacianSmooth(geometry, iterations = 2, lambda = 0.5) {
    const posAttr = geometry.attributes.position;
    const idxAttr = geometry.index;
    if (!idxAttr) return;

    // 1) gather positions
    const V = posAttr.count;
    const P = new Array(V);
    for (let i = 0; i < V; i++) {
        P[i] = new THREE.Vector3().fromBufferAttribute(posAttr, i);
    }

    // 2) build adjacency
    const adj = Array.from({ length: V }, () => new Set());
    const I = idxAttr.array;
    for (let i = 0; i < I.length; i += 3) {
        const [a, b, c] = [I[i], I[i + 1], I[i + 2]];
        adj[a].add(b).add(c);
        adj[b].add(a).add(c);
        adj[c].add(a).add(b);
    }

    // 3) smoothing passes
    for (let pass = 0; pass < iterations; pass++) {
        const Pn = new Array(V);
        for (let i = 0; i < V; i++) {
            const neighbors = adj[i];
            if (!neighbors.size) { Pn[i] = P[i].clone(); continue; }
            // average neighbor positions
            const avg = new THREE.Vector3();
            neighbors.forEach(j => avg.add(P[j]));
            avg.divideScalar(neighbors.size);
            // move towards average
            Pn[i] = P[i].clone().lerp(avg, lambda);
        }
        // copy back
        for (let i = 0; i < V; i++) P[i].copy(Pn[i]);
    }

    // 4) write back & recompute normals
    for (let i = 0; i < V; i++) {
        posAttr.setXYZ(i, P[i].x, P[i].y, P[i].z);
    }
    posAttr.needsUpdate = true;
    geometry.computeVertexNormals();
}


let groundArrow = null;
let belowPointsMesh = null;
let belowMesh = null;
let fullPointsMesh = null; // all points mesh
let cameraSpritesGroup = null;
let elevationBandPointsMesh = null;
let treeStats = [];

function showGroundPoint() {
    if (!groundMesh) segmentGround();

    // remove old helpers
    if (groundArrow) scene.remove(groundArrow);
    if (belowPointsMesh) scene.remove(belowPointsMesh);
    if (belowMesh) scene.remove(belowMesh);
    if (elevationBandPointsMesh) { scene.remove(elevationBandPointsMesh); elevationBandPointsMesh = null; }

    // 2) compute plane normal & centroid
    const posG = groundMesh.geometry.attributes.position.array;
    const A = new THREE.Vector3().fromArray(posG, 0);
    const B = new THREE.Vector3().fromArray(posG, 3);
    const C = new THREE.Vector3().fromArray(posG, 6);
    const D = new THREE.Vector3().fromArray(posG, 9);
    const bestNormal = new THREE.Vector3()
        .crossVectors(B.clone().sub(A), C.clone().sub(A))
        .normalize();
    const centroid = A.clone().add(B).add(C).add(D).multiplyScalar(0.25);

    // 3) build projection matrix P = I – n nᵀ
    const I = new THREE.Matrix3().identity();
    const nnT = new THREE.Matrix3().set(
        bestNormal.x * bestNormal.x, bestNormal.x * bestNormal.y, bestNormal.x * bestNormal.z,
        bestNormal.y * bestNormal.x, bestNormal.y * bestNormal.y, bestNormal.y * bestNormal.z,
        bestNormal.z * bestNormal.x, bestNormal.z * bestNormal.y, bestNormal.z * bestNormal.z
    );
    const Pmat = new THREE.Matrix3();
    Pmat.elements = I.elements.map((v, i) => v - nnT.elements[i]);

    // 4) in‐plane basis e1,e2
    const e1 = (Math.abs(bestNormal.x) < 0.9
        ? bestNormal.clone().cross(new THREE.Vector3(1, 0, 0))
        : bestNormal.clone().cross(new THREE.Vector3(0, 1, 0))
    ).normalize();
    const e2 = bestNormal.clone().cross(e1).normalize();

    // 5) choose “upward” for arrow
    const worldUp = new THREE.Vector3(0, 1, 0);
    const dir = bestNormal.dot(worldUp) >= 0 ? bestNormal : bestNormal.clone().negate();
    groundArrow = new THREE.ArrowHelper(dir, centroid, 2, 0xffff00);
    scene.add(groundArrow);

    // 6) collect below‐plane points
    const clusteredIdx = new Set();
    clusterData.forEach(g => g.indices.forEach(i => clusteredIdx.add(i)));
    const belowPos = [];
    const tmpP = new THREE.Vector3();
    for (let i = 0; i < pointCount; i++) {
        if (clusteredIdx.has(i)) continue;
        tmpP.fromArray(posArray, i * 3);
        if (tmpP.clone().sub(centroid).dot(dir) < 0) {
            belowPos.push(tmpP.x, tmpP.y, tmpP.z);
        }
    }

    if (belowPos.length < 6) return; // need at least 3 points

    if (params.showPoints) {
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(belowPos, 3));
        belowPointsMesh = new THREE.Points(
            geom,
            new THREE.PointsMaterial({ size: 0.05, color: 0x90ee90 })
        );
        scene.add(belowPointsMesh);
    }

    // 8) PROJECT into UV, build Delaunay
    const uv = [];
    const pts3 = [];
    for (let i = 0; i < belowPos.length; i += 3) {
        const P = new THREE.Vector3(belowPos[i], belowPos[i + 1], belowPos[i + 2]);
        // project into plane coords
        const v = P.clone().sub(centroid).applyMatrix3(Pmat);
        const uCoord = v.dot(e1), vCoord = v.dot(e2);
        uv.push([uCoord, vCoord]);
        pts3.push(P);
    }
    const delaunay = Delaunator.from(uv);
    const indices = delaunay.triangles; // flat array of i0,i1,i2,...

    // 9) build mesh geometry
    const meshGeom = new THREE.BufferGeometry();
    // flatten 3D positions
    const posArrayFlat = new Float32Array(pts3.length * 3);
    pts3.forEach((P, i) => posArrayFlat.set(P.toArray(), i * 3));
    meshGeom.setAttribute('position', new THREE.BufferAttribute(posArrayFlat, 3));
    meshGeom.setIndex(Array.from(indices));
    meshGeom.computeVertexNormals();
    laplacianSmooth(meshGeom, 10, 0.5);

    // 10) add transparent ground‐mesh
    const meshMat = new THREE.MeshStandardMaterial({
        color: 0x90ee90,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.2
    });
    belowMesh = new THREE.Mesh(meshGeom, meshMat);
    scene.add(belowMesh);
    
}

function toggleCameraPoints(){

    if (cameraSpritesGroup) {
        cameraSpritesGroup.forEach(element => {
            scene.remove(element)
        });;
    }
    else {
        cameraSpritesGroup = [];
    }

    if (params.renderCameraPoints) {
        // 2) add a numbered sprite at each point
        cameraPoints.forEach((p, i) => {
            // create a tiny canvas
            const size = 128;
            const canvas = document.createElement('canvas');
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');
            ctx.font = '14px Arial';
            ctx.fillStyle = 'white';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(i), size / 2, size / 2);

            // make a texture & sprite
            const tex = new THREE.CanvasTexture(canvas);
            const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
            const sprite = new THREE.Sprite(mat);
            // size on screen: tweak these if you need bigger/smaller
            sprite.scale.set(0.5, 0.5, 1);
            sprite.position.copy(p);
            cameraSpritesGroup.push(sprite);
            scene.add(sprite);
        });
    }
}

function seekVideoFrame(frameIdx) {
    const t = frameIdx / videoFPS;
    // always pause before seeking
    video.pause();
    // set time
    video.currentTime = t;
    // and listen for it…
    video.addEventListener("seeked", function onSeeked() {
        // frame is now updated
        video.removeEventListener("seeked", onSeeked);
    });
}

// — Camera navigation helper —
function moveCamera(delta) {
  if (cameraPoints.length === 0) return;

  // advance index
  prevCamPos.copy(camera.position);
  camIndex = (camIndex + delta + cameraPoints.length) % cameraPoints.length;
  const np = cameraPoints[camIndex];
  camera.position.copy(np);

  // sum directions over the next N points
  const lookAhead = 200;
  const sumDir = new THREE.Vector3(0, 0, 0);
  for (let i = 1; i <= lookAhead || i>=cameraPoints.length; i++) {
    const idx = (camIndex + i) % cameraPoints.length;
    sumDir.add( cameraPoints[idx].clone().sub(np) );
  }

  // (optional) normalize if you want only direction, not magnitude:
  sumDir.normalize();

  // build the target by offsetting current pos by that summed vector
  const tgt = np.clone().add(sumDir);

  // update controls & camera
  controls.target.copy(tgt);
  camera.lookAt(tgt);
  controls.update();

  seekVideoFrame(camIndex);
}

// — Manual ASCII‐PLY loader with color support —
async function loadPLYWithColor(url) {
    const text = await fetch(url).then(r => r.text());
    const lines = text.split('\n');
    let vertexCount = 0, headerEnd = 0;
    const props = [];

    for (let i = 0; i < lines.length; i++) {
        const l = lines[i].trim();
        if (l.startsWith('element vertex')) {
            vertexCount = parseInt(l.split(/\s+/)[2], 10);
        } else if (l.startsWith('property')) {
            props.push(l.split(/\s+/)[2]);
        } else if (l === 'end_header') {
            headerEnd = i;
            break;
        }
    }
    if (!vertexCount || !headerEnd) throw new Error('Invalid PLY header');
    const ix = props.indexOf('x'), iy = props.indexOf('y'), iz = props.indexOf('z');
    const ir = props.indexOf('red'), ig = props.indexOf('green'), ib = props.indexOf('blue');
    if ([ix, iy, iz, ir, ig, ib].some(i => i < 0)) {
        throw new Error(`PLY missing required properties: ${props.join(',')}`);
    }

    const positions = [], colors = [];
    for (let v = 0; v < vertexCount; v++) {
        const parts = lines[headerEnd + 1 + v].trim().split(/\s+/).map(Number);
        const x = -parts[ix], y = -parts[iy], z = parts[iz];
        let r = parts[ir], g = parts[ig], b = parts[ib];
        if (r > 1 || g > 1 || b > 1) { r /= 255; g /= 255; b /= 255; }
        positions.push(x, y, z);
        colors.push(r, g, b);
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return geom;
}

// ---------- Top-View Snapshot Helpers ----------
function getGroundFrame() {
  if (!groundMesh) segmentGround();
  if (!groundMesh) throw new Error("No ground mesh available.");

  const pos = groundMesh.geometry.attributes.position.array;
  if (!pos || pos.length < 12) throw new Error("Ground mesh doesn't have 4 corners.");

  const A = new THREE.Vector3().fromArray(pos, 0);
  const B = new THREE.Vector3().fromArray(pos, 3);
  const C = new THREE.Vector3().fromArray(pos, 6);
  const D = new THREE.Vector3().fromArray(pos, 9);

  const n = new THREE.Vector3().crossVectors(
    B.clone().sub(A),
    C.clone().sub(A)
  ).normalize();

  const centroid = A.clone().add(B).add(C).add(D).multiplyScalar(0.25);

  // Build in-plane basis (e1, e2)
  const e1 = (Math.abs(n.x) < 0.9
    ? n.clone().cross(new THREE.Vector3(1,0,0))
    : n.clone().cross(new THREE.Vector3(0,1,0))
  ).normalize();
  const e2 = n.clone().cross(e1).normalize();

  // Corners in local (u,v) about centroid
  const corners3 = [A,B,C,D];
  const corners2 = corners3.map(P => {
    const v = P.clone().sub(centroid);
    return [v.dot(e1), v.dot(e2)]; // [u,v]
  });

  let minU=Infinity, maxU=-Infinity, minV=Infinity, maxV=-Infinity;
  for (const [u,v] of corners2) {
    if (u<minU) minU=u; if (u>maxU) maxU=u;
    if (v<minV) minV=v; if (v>maxV) maxV=v;
  }

  return { centroid, n, e1, e2, bounds: {minU,maxU,minV,maxV}, corners3 };
}

/**
 * Renders a top-down orthographic view to an offscreen canvas and returns a data URL.
 * @param {object} opt
 *   - width, height: output PNG resolution (px)
 *   - padding: extra meters added around the ground rectangle
 *   - flipNormal: if true, view from the opposite side of the plane normal
 *   - transparent: if true, PNG has alpha background
 */
function captureTopViewDataURL(opt={}) {
  const {
    width = 2048,
    height = 2048,
    padding = 1.0,      // meters
    flipNormal = false, // choose which side to look from
    transparent = false
  } = opt;

  const { centroid, n, e1, e2, bounds } = getGroundFrame();
  const lookN = flipNormal ? n.clone().negate() : n.clone();

  // Ortho frustum in (u,v) plane
  const left   = bounds.minU - padding;
  const right  = bounds.maxU + padding;
  const bottom = bounds.minV - padding;
  const top    = bounds.maxV + padding;

  // Build an orthographic camera whose local X->e1, Y->e2, -Z->lookN
  const near = 0.1, far = 10000;            // generous depth range
  const ortho = new THREE.OrthographicCamera(left, right, top, bottom, near, far);

  // Place camera some distance along +lookN so everything is in front
  const dist = 50;                           // any positive; ortho scale ignores this
  ortho.position.copy(centroid.clone().add(lookN.clone().multiplyScalar(dist)));
  ortho.up.copy(e2);                         // control "north" on the snapshot
  ortho.lookAt(centroid);
  ortho.updateProjectionMatrix();

  // Offscreen renderer so we don't resize your main canvas
  const offCanvas = document.createElement('canvas');
  const offRenderer = new THREE.WebGLRenderer({
    canvas: offCanvas,
    antialias: true,
    preserveDrawingBuffer: true
  });
  offRenderer.setSize(width, height, false);
  if (transparent) {
    offRenderer.setClearColor(0x000000, 0.0);
  } else {
    offRenderer.setClearColor(0xffffff, 1.0);
  }

  // Render once
  offRenderer.render(scene, ortho);

  // PNG as data URL
  const url = offCanvas.toDataURL('image/png');
  // Clean up WebGL context
  offRenderer.dispose();
  return url;
}

/**
 * Convenience: capture and trigger a download.
 * @param {string} filename
 * @param {object} opt - forwarded to captureTopViewDataURL()
 */
function saveTopViewPNG(filename='top_view.png', opt={}) {
  const url = captureTopViewDataURL(opt);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  return url; // also hand back the data URL if you want to embed it somewhere
}


// — Initialization —
(async () => {
    let geometry;
    try {
        geometry = await loadPLYWithColor('/GS_Forest/Red_Pine_1/points_with_cameras_scaled.ply');

        fullPointsMesh = new THREE.Points(
            geometry,
            new THREE.PointsMaterial({ size: 0.05, vertexColors: true })
        );
        fullPointsMesh.visible = false;      // start hidden
        scene.add(fullPointsMesh);
        console.log('Loaded attrs:', Object.keys(geometry.attributes));
    } catch (e) {
        console.error('PLY load error:', e);
        return;
    }

    // split pure‐red camera points
    const posArr = geometry.attributes.position.array;
    const colArr = geometry.attributes.color.array;
    const N = posArr.length / 3;
    posArray = [];
    cameraPoints = [];
    for (let i = 0; i < N; i++) {
        const x = posArr[3 * i], y = posArr[3 * i + 1], z = posArr[3 * i + 2];
        const r = colArr[3 * i], g = colArr[3 * i + 1], b = colArr[3 * i + 2];
        const isRed = (r > 0.9 && g < 0.1 && b < 0.1);
        if (isRed) cameraPoints.push(new THREE.Vector3(x, y, z));
        else posArray.push(x, y, z);
    }
    pointCount = posArray.length / 3;
    cameraPoints.reverse()
    if (cameraPoints.length) camIndex = -1;

    // optional: visualize camera points
    if (cameraPoints.length) {
        // 1) existing point cloud
        const arr = cameraPoints.flatMap(v => [v.x, v.y, v.z]);
        const pointsGeom = new THREE.BufferGeometry().setAttribute(
            'position',
            new THREE.Float32BufferAttribute(arr, 3)
        );
        //scene.add(new THREE.Points(
        //  pointsGeom,
        //  new THREE.PointsMaterial({ size:0.1, color:0xff0000 })
        //));


    }

    // run segmentation + clustering
    segmentGround();
    clusterTrees();


    if (cameraPoints.length) {

        // 2) Read back the plane normal & pick an in-plane axis e1
        const gp = groundMesh.geometry.attributes.position.array;
        const A = new THREE.Vector3().fromArray(gp, 0);
        const B = new THREE.Vector3().fromArray(gp, 3);
        const C = new THREE.Vector3().fromArray(gp, 6);
        const n = new THREE.Vector3().crossVectors(
            B.clone().sub(A),
            C.clone().sub(A)
        ).normalize();

        const e1 = (Math.abs(n.x) < 0.9
            ? n.clone().cross(new THREE.Vector3(1, 0, 0))
            : n.clone().cross(new THREE.Vector3(0, 1, 0))
        ).normalize();

        // 3) (Optional) find a centroid so your sort is centered, not origin-biased
        const centroid = cameraPoints
            .reduce((sum, p) => sum.add(p), new THREE.Vector3())
            .divideScalar(cameraPoints.length);

        // 4) Build the “project onto plane” matrix: P = I – n nᵀ
        const I = new THREE.Matrix3().identity();
        const nnT = new THREE.Matrix3().set(
            n.x * n.x, n.x * n.y, n.x * n.z,
            n.y * n.x, n.y * n.y, n.y * n.z,
            n.z * n.x, n.z * n.y, n.z * n.z
        );
        const Pmat = new THREE.Matrix3();
        Pmat.elements = I.elements.map((v, i) => v - nnT.elements[i]);

        // 5) Sort cameraPoints by their projected coordinate along e1
        cameraPoints.sort((pA, pB) => {
            const uA = pA.clone().sub(centroid).applyMatrix3(Pmat).dot(e1);
            const uB = pB.clone().sub(centroid).applyMatrix3(Pmat).dot(e1);
            return uA - uB;
        });

        toggleCameraPoints()

    }


    // once metadata + at least one frame is available…
    video.addEventListener("loadeddata", async () => {
        // in modern browsers:
        if (video.getVideoPlaybackQuality) {
            // play & pause so some frames are decoded
            await video.play();
            video.pause();
            const q = video.getVideoPlaybackQuality();
            videoFPS = q.totalVideoFrames / video.duration;
        }
        // fallback for WebKit:
        else if (typeof video.webkitDecodedFrameCount !== "undefined") {
            await video.play();
            video.pause();
            videoFPS = video.webkitDecodedFrameCount / video.duration;
        }
        console.log("Detected video FPS:", videoFPS);
    });


    // GUI controls
    gui.add(params, 'maxIterations', 10, 10000, 10).name('Ground plane iters')
        .onChange(() => { segmentGround(); clusterTrees(); });
    gui.add(params, 'clusterThreshold', 0.1, 5.0, 0.1).name('Cluster ε')
        .onChange(clusterTrees);
    gui.add(params, 'minClusterSize', 1, 500, 1).name('Min Cluster')
        .onChange(clusterTrees);
    gui.add(params, 'minTreeHeight', 0.1, 5, 0.1).name('Min Tree Height')
        .onChange(clusterTrees);
    gui.add(params, 'mergeClusterThreshold', 0.1, 5, 0.1).name('Merge Cluster Thres')
        .onChange(clusterTrees);
    gui.add(params, 'radiusSigmaK', 0.5, 5.0, 0.1).name('Radius Sigma K')
        .onChange(clusterTrees);
    gui.add(params, 'elevationBand', 0.05, 2.0, 0.01).name('Elev Band (m)')
        .onChange(clusterTrees);
    gui.add(params, 'showElevationBandPoints').name('Show Elev Band Pts')
        .onChange(clusterTrees);
    gui.add(params, 'showGround').name('Show Ground')
        .onChange(v => groundMesh.visible = v);
    gui.add(params, 'showVegetation').name('Show Trees')
        .onChange(v => vegPointsMesh.visible = v);
    gui.add(params, 'showPoints').name('Show Points')
        .onChange(v => { segmentGround(); clusterTrees(); });
    gui.add(params, 'renderCameraPoints').name('Show Cam Points')
        .onChange(v => { toggleCameraPoints(); });
    gui.add(params, 'showImage').name('Show Image')
        .onChange(v => {
            if (v) {
                document.querySelector("#mainVideo").style.display = "block";
            }
            else {
                document.querySelector("#mainVideo").style.display = "none";
            }
        });
    gui.add(params, 'showAllPoints')
        .name('Show All Points')
        .onChange(v => fullPointsMesh.visible = v);
    gui.add({ export: () => exportTreeStatsCSV() }, 'export').name('Export Trees CSV');
    gui.add({ prev: () => moveCamera(-1) }, 'prev').name('◀ Camera');
    gui.add({ next: () => moveCamera(+1) }, 'next').name('Camera ▶');

    gui.add({ saveTop: () => saveTopViewPNG('top_view.png', {
        width: 2048,
        height: 2048,
        padding: 1.0,        // meters around ground rect
        flipNormal: false,   // set true if you need the opposite side
        transparent: false   // set true for alpha background
        }) }, 'saveTop').name('Save Top View PNG');
    window.addEventListener('keydown', e => {
        if (e.key === 'ArrowLeft') moveCamera(-1);
        if (e.key === 'ArrowRight') moveCamera(+1);
    });
})();

const clock = new THREE.Clock();
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
(function animate() {
    requestAnimationFrame(animate);
    controls.update();

    const t = clock.getElapsedTime();            // seconds since start
    const scaleFactor = 1 + 0.3 * Math.sin(t * 3); // oscillate ±30% at ~3Hz

    centerCubes.forEach(cube => {
        cube.scale.setScalar(scaleFactor);         // uniform scale on x/y/z
    });

    renderer.render(scene, camera);
})();

// — CSV export of tree stats —
function exportTreeStatsCSV() {
    if (!treeStats || treeStats.length === 0) {
        console.warn('No tree stats to export. Run clustering first.');
        return;
    }
    const header = ['centerX','centerY','centerZ','radius','diameter','height','points'];
    const rows = treeStats.map(t => [
        t.centerX, t.centerY, t.centerZ, t.radius, t.diameter, t.height, t.points
    ]);
    const csv = [header.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'trees.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
