import * as THREE from 'three';
import seedrandom from 'seedrandom';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Delaunator from 'delaunator';
import { GUI } from 'dat.gui';
import { getMaskIdAtPoint, frameMap, decodeRLEtoMask } from './tree_info.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
const gltfLoader = new GLTFLoader();


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
    maxIterations: 20000,
    clusterThreshold: 0.3,
    minClusterSize: 120,
    showGround: true,
    showVegetation: true,
    minTreeHeight: 0.6,
    mergeClusterThreshold: 1.5,
    showPoints: true,
    renderCameraPoints: false,
    showImage: false,
    showAllPoints: true
};

// — Seeded RNG for deterministic behavior —
const rng = seedrandom('fixed-seed');
function rngInt(min, max) { return Math.floor(rng() * (max - min + 1)) + min; }

// — Data placeholders —
let posArray, pointCount, groundSet, vegPointsMesh;
let cameraPoints = [], camIndex = 0, clusterData = [], treeMeshes = [];
const prevCamPos = new THREE.Vector3();

let treeMetrics = [];
// — 1) Ground segmentation + mesh creation —
function segmentGround() {
    // 1) sample random triangle normals
    const normals = [];
    for (let i = 0; i < params.maxIterations; i++) {
        const picks = new Set();
        while (picks.size < 3) picks.add(rngInt(0, pointCount - 1));
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
    const minU = projUs.reduce((a, b) => Math.min(a, b), Infinity);
    const maxU = projUs.reduce((a, b) => Math.max(a, b), -Infinity);
    const minV = projVs.reduce((a, b) => Math.min(a, b), Infinity);
    const maxV = projVs.reduce((a, b) => Math.max(a, b), -Infinity);

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

// ensure these are defined at top-level:
let centerCubes = [];

function distanceAlongDirection(p1, p2, direction) {
    const u = direction.clone().normalize();
    return p2.clone().sub(p1).dot(u);
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

    tree.add(segment);

    if (levels > 0) {
        // 3) compute new branch parameters
        const newLength = length * (0.7 + rng() * 0.1);
        const newRadius = radius * 0.4;

        // 4) endpoints for this segment
        const end = pos.clone()
            .add(dir.clone().multiplyScalar(length));

        // 5) spawn two child branches at seeded random angles
        for (let i = 0; i < 2; i++) {
            // pick an angle off the dir vector
            const axis = new THREE.Vector3(
                rng() - 0.5,
                rng(),
                rng() - 0.5
            ).normalize();
            const angle = (Math.PI / 4) + (rng() * Math.PI / 8);
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

function forceBrownTransparentUnlit(object3D, {
  color = 0x8B5A2B,  // or a THREE.Color
  opacity = 0.35,
  doubleSided = true
} = {}) {
  const color3 = (color.isColor ? color : new THREE.Color(color));

  object3D.traverse((child) => {
    if (!child.isMesh) return;

    // 1) Strip materials & textures
    const mats = Array.isArray(child.material) ? child.material : [child.material];
    mats.forEach((m) => {
      if (!m) return;
      [
        'map','aoMap','metalnessMap','roughnessMap','normalMap','emissiveMap','specularMap',
        'alphaMap','envMap','clearcoatNormalMap','sheenColorMap','transmissionMap','thicknessMap'
      ].forEach((k) => { if (m[k]) { m[k].dispose?.(); m[k] = null; } });
      m.dispose?.();
    });

    // 2) Remove vertex colors (can force everything dark)
    const g = child.geometry;
    if (g && g.attributes && g.attributes.color) {
      g.deleteAttribute('color');
      g.attributes.color = undefined;
    }

    // 3) Ensure normals exist for good measure
    if (g && (!g.attributes.normal || g.attributes.normal.count === 0)) {
      g.computeVertexNormals();
    }

    // 4) Replace with UNLIT material (lighting-independent)
    const newMat = new THREE.MeshBasicMaterial({
      color: color3,
      transparent: true,
      opacity,
      side: doubleSided ? THREE.DoubleSide : THREE.FrontSide,
      depthWrite: false
    });

    // Just in case any flags were odd on the old material
    newMat.colorWrite = true;
    newMat.toneMapped = false;  // keep exact color under any tone mapping

    child.material = newMat;
    child.castShadow = false;   // not necessary for unlit; avoids odd artifacts
    child.receiveShadow = false;
  });

  object3D.renderOrder = 1;
}




function clusterTrees() {
    if (!groundMesh) segmentGround();

    // — clear previous visuals —
    centerCubes.forEach(c => scene.remove(c));
    centerCubes = [];
    if (vegPointsMesh) scene.remove(vegPointsMesh);

    treeMetrics = [];


    // 1) extract ground‐mesh corners & compute plane normal + centroid
    const gp = groundMesh.geometry.attributes.position.array;
    const A = new THREE.Vector3().fromArray(gp, 0);
    const B = new THREE.Vector3().fromArray(gp, 3);
    const C = new THREE.Vector3().fromArray(gp, 6);
    const D = new THREE.Vector3().fromArray(gp, 9);

    const bestNormal = new THREE.Vector3()
        .crossVectors(B.clone().sub(A), C.clone().sub(A))
        .normalize();
    const centroid = A.clone().add(B).add(C).add(D).multiplyScalar(0.25);

    // 2) build projection matrix P = I – n nᵀ for in‐plane coords
    const I = new THREE.Matrix3().identity();
    const nnT = new THREE.Matrix3().set(
        bestNormal.x * bestNormal.x, bestNormal.x * bestNormal.y, bestNormal.x * bestNormal.z,
        bestNormal.y * bestNormal.x, bestNormal.y * bestNormal.y, bestNormal.y * bestNormal.z,
        bestNormal.z * bestNormal.x, bestNormal.z * bestNormal.y, bestNormal.z * bestNormal.z
    );
    const Pmat = new THREE.Matrix3();
    Pmat.elements = I.elements.map((v, i) => v - nnT.elements[i]);

    // 3) choose two in‐plane axes e1, e2
    const e1 = (Math.abs(bestNormal.x) < 0.9
        ? bestNormal.clone().cross(new THREE.Vector3(1, 0, 0))
        : bestNormal.clone().cross(new THREE.Vector3(0, 1, 0))
    ).normalize();
    const e2 = bestNormal.clone().cross(e1).normalize();

    // 4) bin points into 2D grid cells of size = clusterThreshold
    const bins = new Map();
    const invT = 1 / params.clusterThreshold;
    const tmp = new THREE.Vector3();
    for (let i = 0; i < pointCount; i++) {
        tmp.fromArray(posArray, i * 3).sub(centroid);
        const inPlane = tmp.clone().applyMatrix3(Pmat);
        const u = inPlane.dot(e1), v = inPlane.dot(e2);
        const ku = Math.floor(u * invT), kv = Math.floor(v * invT);
        const key = `${ku},${kv}`;
        if (!bins.has(key)) bins.set(key, []);
        bins.get(key).push(i);
    }

    // 5) initial clusters ≥ minClusterSize
    let clusters = Array.from(bins.values())
        .filter(c => c.length >= params.minClusterSize);

    // 6) IQR‐based height filter along normal (seeded sampling)
    clusters = clusters.filter(cluster => {
        const n = cluster.length;
        if (n < 2) return false;
        const sampleCount = Math.min(10000, n * (n - 1) / 2);
        const dists = [];
        for (let k = 0; k < sampleCount; k++) {
            let i = rngInt(0, n - 1);
            let j;
            do { j = rngInt(0, n - 1); } while (j === i);
            const p1 = new THREE.Vector3().fromArray(posArray, cluster[i] * 3);
            const p2 = new THREE.Vector3().fromArray(posArray, cluster[j] * 3);
            dists.push(Math.abs(distanceAlongDirection(p1, p2, bestNormal)));
        }
        if (dists.length < 4) return false;
        dists.sort((a, b) => a - b);
        const q1 = dists[Math.floor(dists.length * 0.25)];
        const q3 = dists[Math.floor(dists.length * 0.75)];
        return (q3 - q1) > params.minTreeHeight;
    });

    // 7) compute centroids for merging step
    clusterData = clusters.map(cluster => {
        const cen = new THREE.Vector3();
        cluster.forEach(idx => cen.add(new THREE.Vector3().fromArray(posArray, idx * 3)));
        cen.divideScalar(cluster.length);
        return { indices: cluster.slice(), centroid: cen };
    });

    // 8) merge clusters whose centroids are very close (Euclidean distance)
    for (let i = 0; i < clusterData.length; i++) {
        for (let j = i + 1; j < clusterData.length; j++) {
            const d = clusterData[i].centroid.distanceTo(clusterData[j].centroid);
            if (d < params.mergeClusterThreshold) {
                // merge j into i
                clusterData[i].indices.push(...clusterData[j].indices);
                // recompute centroid of merged cluster
                const allPts = clusterData[i].indices.map(idx =>
                    new THREE.Vector3().fromArray(posArray, idx * 3)
                );
                const newCen = allPts.reduce((sum, p) => sum.add(p), new THREE.Vector3())
                    .divideScalar(allPts.length);
                clusterData[i].centroid.copy(newCen);
                // remove j
                clusterData.splice(j, 1);
                j--;
            }
        }
    }

    // 9) visualize merged clusters
    const vegPos = [];
    const vegCols = [];

    // Clear old trees
    treeMeshes.forEach(t => scene.remove(t));
    treeMeshes.length = 0;
    if (params.showPoints) {
    centerCubes.forEach(c => scene.remove(c));
    centerCubes.length = 0;
    }
    vegToCluster = {};

    const tryAddGLBOrProcedural = (data, clusterIndex, col) => {
    // 1) accumulate points + color + mapping
    data.indices.forEach(idx => {
        vegPos.push(...posArray.slice(idx * 3, idx * 3 + 3));
        vegCols.push(col.r, col.g, col.b);
        const x = posArray[idx * 3], y = posArray[idx * 3 + 1], z = posArray[idx * 3 + 2];
        vegToCluster[`${x},${y},${z}`] = clusterIndex;
    });

    // 2) centroid cube
    const cube = new THREE.Mesh(
        new THREE.BoxGeometry(0.2, 0.2, 0.2),
        new THREE.MeshStandardMaterial({ color: col })
    );
    cube.position.copy(data.centroid);
    scene.add(cube);
    if (params.showPoints) centerCubes.push(cube);

    // 3) compute vertical span, base radius, base position
    let minY = Infinity, maxY = -Infinity, maxDist = 0;
    const cx = data.centroid.x, cz = data.centroid.z;
    data.indices.forEach(idx => {
        const x = posArray[idx * 3], y = posArray[idx * 3 + 1], z = posArray[idx * 3 + 2];
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        const dx = x - cx, dz = z - cz;
        const d = Math.hypot(dx, dz);
        if (d > maxDist) maxDist = d;
    });
    const height = Math.max(0.01, maxY - minY);
    const radius = Math.max(0.01, maxDist);
    const bottomPos = new THREE.Vector3(cx, minY, cz);

    // 4) metrics
    if (!Array.isArray(treeMetrics)) treeMetrics = [];
    const metricId = treeMetrics.length;
    const metrics = {
        id: metricId,
        base_position: { x: bottomPos.x, y: bottomPos.y, z: bottomPos.z },
        centroid: { x: data.centroid.x, y: data.centroid.y, z: data.centroid.z },
        height: height,
        width: 2 * radius
    };
    treeMetrics.push(metrics);

    // 5) attempt to load GLB; fallback to procedural on error
    const url = `./GS_Forest/Red_Pine_1/segmented_images/${clusterIndex}.glb`;

    const useProcedural = () => {
        const tree = createProceduralTree(
        /*levels=*/3,
        /*length=*/height,
        /*radius=*/radius,
        bottomPos,
        new THREE.Vector3(0, 1, 0)
        );
        scene.add(tree);
        treeMeshes.push(tree);
    };

    gltfLoader.load(
        url,
        (gltf) => {
        // Put model in a group for easy transforms
        const group = new THREE.Group();
        const model = gltf.scene || gltf.scenes?.[0];
        if (!model) {
            useProcedural();
            return;
        }
        group.add(model);

        // Compute size BEFORE scaling
        const preBox = new THREE.Box3().setFromObject(group);
        const preSize = new THREE.Vector3();
        preBox.getSize(preSize);

        // If model has zero dims somehow, fallback
        if (preSize.y <= 1e-6 || (preSize.x <= 1e-6 && preSize.z <= 1e-6)) {
            useProcedural();
            return;
        }

        // Target height and width (diameter). Fit uniformly.
        const targetH = height;
        const targetW = 2 * radius;
        const srcH = preSize.y;
        const srcW = Math.max(preSize.x, preSize.z);

        // Fit to both height and width conservatively (no overgrow): pick the smaller scale
        const sH = targetH / srcH;
        const sW = targetW / Math.max(1e-6, srcW);
        const s = Math.min(sH, sW);

        model.scale.setScalar(s);

        // Recompute bounds AFTER scaling to align base to minY
        const postBox = new THREE.Box3().setFromObject(group);
        const yShift = -postBox.min.y; // how much to lift so base is at y=0 in group space

        group.position.set(bottomPos.x, bottomPos.y + yShift, bottomPos.z);

        // Optional: orient if models are not Y-up or need random yaw
        // group.rotation.y = Math.random() * Math.PI * 2;
        forceBrownTransparentUnlit(group, { color: 0x8B5A2B, opacity: 0.7 });


        scene.add(group);
        treeMeshes.push(group);
        },
        undefined,
        // onError -> fallback to procedural
        (_err) => {
        useProcedural();
        }
    );
    };

    // ----- main cluster loop -----
    clusterData.forEach((data, clusterIndex) => { 
    const col = new THREE.Color(rng(), rng(), rng());
    tryAddGLBOrProcedural(data, clusterIndex, col);
    });


    if (params.showPoints) {
        // 10) draw clustered points mesh
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
let vegToCluster = {};

function showGroundPoint() {
    if (!groundMesh) segmentGround();

    // remove old helpers
    if (groundArrow) scene.remove(groundArrow);
    if (belowPointsMesh) scene.remove(belowPointsMesh);
    if (belowMesh) scene.remove(belowMesh);

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

function toggleCameraPoints() {
    if (cameraSpritesGroup) {
        cameraSpritesGroup.forEach(element => {
            scene.remove(element);
        });
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

// Helper: get video/canvas dimensions (fallbacks are safe)
function getVideoDims() {
    const W = (video && video.videoWidth) || renderer.domElement.clientWidth || 640;
    const H = (video && video.videoHeight) || renderer.domElement.clientHeight || 384;
    return { W, H };
}

function getAllTreePointsAtDistanceFromCamera(minDistance, maxDistance) {
    if (!vegPointsMesh || !vegPointsMesh.geometry?.attributes?.position) return [];

    // Ensure camera matrices are fresh
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();

    const { W, H } = getVideoDims();

    // Camera world position & forward direction (unit)
    const camPos = new THREE.Vector3();
    const viewDir = new THREE.Vector3();
    camera.getWorldPosition(camPos);
    camera.getWorldDirection(viewDir); // normalized

    // Iterate vegetation vertices
    const posAttr = vegPointsMesh.geometry.attributes.position;
    const V = posAttr.count;

    const out = [];
    const pWorld = new THREE.Vector3();
    const pNDC = new THREE.Vector3();

    for (let i = 0; i < V; i++) {
        // world position of vertex
        pWorld.fromBufferAttribute(posAttr, i);

        // signed distance ALONG the camera's forward direction:
        const along = pWorld.clone().sub(camPos).dot(viewDir);

        // Only accept points within [minDistance, maxDistance]
        if (along < minDistance || along > maxDistance) continue;

        // Project to NDC
        pNDC.copy(pWorld).project(camera);

        // Clip-space visibility check
        if (pNDC.z < -1 || pNDC.z > 1) continue;

        // Convert NDC -> pixel coords
        const x = (pNDC.x + 1) * 0.5 * W;
        const y = (1 - (pNDC.y + 1) * 0.5) * H;

        // Discard if off-screen
        if (x < 0 || y < 0 || x > W || y > H) continue;

        out.push({
            x,
            y,
            index: i,
            treeIndex: vegToCluster[
                pWorld.x + "," + pWorld.y + "," + pWorld.z
            ]
        });
    }

    return out;
}


const images = {}
async function downloadImagesAsZip(images) {
  const capturedTreeIndexes = {};
  const zip = new JSZip();

  // Group by treeIndex
  for (const id of Object.keys(images)) {
    for (const { img, treeIndex, pointsCount } of images[id]) {
      (capturedTreeIndexes[treeIndex] ??= []).push({ img, pointsCount });
    }
  }

  // Build async tasks (one image per treeIndex: the max pointsCount)
  const tasks = Object.entries(capturedTreeIndexes).map(async ([treeIndex, imageObj]) => {
    const { img } = imageObj.reduce((max, cur) =>
      cur.pointsCount > max.pointsCount ? cur : max, imageObj[0]
    );

    if (!img) return;

    let blob = null;

    // If it's a canvas, use toBlob
    if (img instanceof HTMLCanvasElement) {
      blob = await new Promise(resolve => img.toBlob(resolve, "image/png"));
    } else {
      // Otherwise assume <img> element with a src
      const src = img.src;
      if (!src) return;

      const res = await fetch(src, { mode: "cors" }); // CORS must be allowed for remote URLs
      if (!res.ok) {
        console.warn(`Failed to fetch ${src}: ${res.status} ${res.statusText}`);
        return;
      }
      blob = await res.blob();
    }

    if (blob) {
      zip.file(`${treeIndex}.png`, blob);
    }
  });

  // WAIT for all files to be added
  await Promise.allSettled(tasks);

  // Generate ZIP and trigger download
  const zipBlob = await zip.generateAsync({ type: "blob" });
  saveAs(zipBlob, "segmented_images.zip");
}


function denormBox(bn, W, H) {
    const [x, y, w, h] = bn;
    return [x * W, y * H, w * W, h * H];
}

function downloadJSON(obj, filename = "trees.json") {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    saveAs(blob, filename); // you already use FileSaver's saveAs()
}


const decodedMaskCache = new Map();

function makeMaskedCropForId(id, camIndex, { background = "transparent" } = {}) {
    const fr = frameMap.get(camIndex);
    if (!fr || !Array.isArray(fr.detections)) return null;

    const W = video.videoWidth || tracking.meta?.width || 640;
    const H = video.videoHeight || tracking.meta?.height || 384;

    // Find detection by id
    const di = fr.detections.findIndex(d => (d.id ?? d.raw_id) == id);
    if (di < 0) return null;
    const det = fr.detections[di];
    if (!det?.mask_rle) return null;

    // Decode mask (cached)
    const key = camIndex + "#" + di;
    let dec = decodedMaskCache.get(key);
    if (!dec) {
        dec = decodeRLEtoMask(det.mask_rle); // { data (Uint8Array), h, w }
        decodedMaskCache.set(key, dec);
    }
    const { data, h: mh, w: mw } = dec;

    // Get bbox in pixel coords (prefer provided bbox; fallback to mask bounds)
    let bx, by, bw, bh;
    if (det.bbox_norm) {
        [bx, by, bw, bh] = denormBox(det.bbox_norm, W, H);
    } else {
        // Fallback: compute bbox from mask (in mask space, then map to image space)
        let minX = mw, minY = mh, maxX = -1, maxY = -1;
        for (let x = 0; x < mw; x++) {
            const colBase = x * mh;
            for (let y = 0; y < mh; y++) {
                if (data[colBase + y]) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return null; // no pixels
        bx = (minX / mw) * W;
        by = (minY / mh) * H;
        bw = ((maxX - minX + 1) / mw) * W;
        bh = ((maxY - minY + 1) / mh) * H;
    }

    // Clamp & round bbox to integer crop canvas
    const ix = Math.max(0, Math.floor(bx));
    const iy = Math.max(0, Math.floor(by));
    const iw = Math.max(1, Math.min(W - ix, Math.round(bw)));
    const ih = Math.max(1, Math.min(H - iy, Math.round(bh)));

    // 1) Draw the video crop
    const crop = document.createElement("canvas");
    crop.width = iw;
    crop.height = ih;
    const cctx = crop.getContext("2d");
    cctx.drawImage(video, ix, iy, iw, ih, 0, 0, iw, ih);

    // 2) Build a crop-sized alpha mask from the detection mask
    const maskCan = document.createElement("canvas");
    maskCan.width = iw;
    maskCan.height = ih;
    const mctx = maskCan.getContext("2d");
    const maskImg = mctx.createImageData(iw, ih);
    const mdata = maskImg.data;

    // Map from mask coords (mw x mh, column-major) to crop coords (iw x ih)
    const scaleX = W / mw;
    const scaleY = H / mh;

    // Only loop within mask-space region that overlaps the bbox → faster
    const sx = Math.max(0, Math.floor(ix / scaleX));
    const ex = Math.min(mw - 1, Math.ceil((ix + iw) / scaleX) - 1);
    const sy = Math.max(0, Math.floor(iy / scaleY));
    const ey = Math.min(mh - 1, Math.ceil((iy + ih) / scaleY) - 1);

    // Coverage for upscales to avoid holes
    const covX = Math.max(1, Math.ceil(scaleX));
    const covY = Math.max(1, Math.ceil(scaleY));

    for (let mx = sx; mx <= ex; mx++) {
        const colBase = mx * mh;
        const ixFloat = mx * scaleX;
        for (let my = sy; my <= ey; my++) {
            if (!data[colBase + my]) continue;
            const iyFloat = my * scaleY;

            // Convert to crop coords
            const cx0 = Math.floor(ixFloat - ix);
            const cy0 = Math.floor(iyFloat - iy);

            // Fill a small block to cover scaling gaps
            for (let dx = 0; dx < covX; dx++) {
                const cx = cx0 + dx;
                if (cx < 0 || cx >= iw) continue;
                for (let dy = 0; dy < covY; dy++) {
                    const cy = cy0 + dy;
                    if (cy < 0 || cy >= ih) continue;
                    const aIdx = (cy * iw + cx) * 4 + 3; // alpha channel
                    mdata[aIdx] = 255;
                }
            }
        }
    }
    mctx.putImageData(maskImg, 0, 0);

    // 3) Apply mask to crop (everything else becomes transparent)
    cctx.globalCompositeOperation = "destination-in";
    cctx.drawImage(maskCan, 0, 0);
    cctx.globalCompositeOperation = "source-over";

    // 4) Optional black background
    let outCan = crop;
    if (background === "black") {
        outCan = document.createElement("canvas");
        outCan.width = iw;
        outCan.height = ih;
        const octx = outCan.getContext("2d");
        octx.fillStyle = "black";
        octx.fillRect(0, 0, iw, ih);
        octx.drawImage(crop, 0, 0);
    }

    // 5) To <img>
    const img = new Image();
    img.src = outCan.toDataURL("image/png");
    return img;
}

// — Camera navigation helper —
function moveCamera(delta) {
    console.log(cameraPoints)
    if (cameraPoints.length === 0) return;

    // advance index
    prevCamPos.copy(camera.position);
    camIndex+=delta
    const np = cameraPoints[camIndex];
    camera.position.copy(np);

    // sum directions over the next N points
    const lookAhead = 300;
    const sumDir = new THREE.Vector3(0, 0, 0);
    for (let i = 1; i <= lookAhead &&  (camIndex + i) < cameraPoints.length; i++) {
        const idx = (camIndex + i);
        sumDir.add(cameraPoints[idx].clone().sub(np));
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

    let points = getAllTreePointsAtDistanceFromCamera(7, 15.0);
    // === REMOVE OLD OVERLAY IF IT EXISTS ===
    //if (window.debugPointsMesh) {
    //  scene.remove(window.debugPointsMesh);
    //  window.debugPointsMesh.geometry.dispose();
    //  window.debugPointsMesh.material.dispose();
    //  window.debugPointsMesh = null;
    //}
    //
    //// === BUILD OVERLAY GEOMETRY ===
    //const posAttr = vegPointsMesh.geometry.attributes.position;
    //const debugVerts = [];
    //
    //for (const pt of points) {
    //  const idx = pt.index * 3; // 3 floats per vertex
    //  const x = posAttr.array[idx];
    //  const y = posAttr.array[idx + 1];
    //  const z = posAttr.array[idx + 2];
    //  debugVerts.push(x, y, z);
    //}
    //
    //const debugGeom = new THREE.BufferGeometry();
    //debugGeom.setAttribute('position', new THREE.Float32BufferAttribute(debugVerts, 3));
    //
    //// === MATERIAL ===
    //const debugMat = new THREE.PointsMaterial({
    //  size: 0.2,        // adjust for your scene scale
    //  color: 0xff0000,  // red
    //  sizeAttenuation: true
    //});
    //
    //// === MESH & ADD TO SCENE ===
    //window.debugPointsMesh = new THREE.Points(debugGeom, debugMat);
    //scene.add(window.debugPointsMesh);
    let ids = {}
    for (const pt of points) {
        let id = getMaskIdAtPoint(camIndex, pt.x / window.innerWidth, pt.y / window.innerHeight);

        if (id) {
            if (ids[id]) {
                ids[id].push(pt.index);
            }
            else {
                ids[id] = [pt.index];
            }
        }
    }
    Object.keys(ids).forEach(id => {
        if (ids[id].length >= 50) {

            let treeIndexes = {}
            ids[id].forEach(idx => {
                let treeIndex = vegToCluster[posArray[idx * 3] + "," + posArray[idx * 3 + 1] + "," + posArray[idx * 3 + 2]];
                if (treeIndex !== undefined) {
                    if (!treeIndexes[treeIndex]) treeIndexes[treeIndex] = [];
                    treeIndexes[treeIndex].push(idx);
                }
            });
            const img = makeMaskedCropForId(id, camIndex, { background: "transparent" }); // or "black"
            if (!img) return;
            if (!images[id]) images[id] = [];
            Object.entries(treeIndexes).forEach(([treeIndex, arr]) => {
                images[id].push({img:img, treeIndex: treeIndex, pointsCount: arr.length});
            });
        }
    });



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

// — Initialization —
(async () => {
    let geometry;
    try {
        geometry = await loadPLYWithColor('/GS_Forest/Red_Pine_1/points.ply');

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
        const isRed = (r === 1 && g === 0 && b === 0);
        if (isRed){
            cameraPoints.push(new THREE.Vector3(x, y, z));
        }
        else posArray.push(x, y, z);
    }
    pointCount = posArray.length / 3;

    cameraPoints.reverse();
    if (cameraPoints.length) camIndex = -1;

    // optional: visualize camera points
    if (cameraPoints.length) {
        const arr = cameraPoints.flatMap(v => [v.x, v.y, v.z]);
        const pointsGeom = new THREE.BufferGeometry().setAttribute(
            'position',
            new THREE.Float32BufferAttribute(arr, 3)
        );
        //scene.add(new THREE.Points(pointsGeom, new THREE.PointsMaterial({ size:0.1, color:0xff0000 })));
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

        toggleCameraPoints();
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
    gui.add({ prev: () => moveCamera(-1) }, 'prev').name('◀ Camera');
    gui.add({ next: () => moveCamera(+1) }, 'next').name('Camera ▶');
    window.addEventListener('keydown', e => {
        if (e.key === 'ArrowLeft') moveCamera(-1);
        if (e.key === 'ArrowRight') moveCamera(+1);
    });

    //runCameraAndDownload();
})();

async function runCameraAndDownload() {
    for (const p of cameraPoints) {
        moveCamera(+1);
        await new Promise(resolve => setTimeout(resolve, 200)); // wait 1 sec
    }

    if (images && Object.keys(images).length > 0) {
        await downloadImagesAsZip(images);
    }

    // Export tree metrics gathered from clusterTrees()
    if (Array.isArray(treeMetrics) && treeMetrics.length > 0) {
        downloadJSON({ trees: treeMetrics });
    } else {
        console.warn("No tree metrics to export; clusterTrees() may not have run or found any clusters.");
    }

}
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
