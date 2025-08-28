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
    maxIterations: 30000,
    clusterThreshold: 0.3,
    minClusterSize: 120,
    showGround: true,
    showVegetation: true,
    minTreeHeight: 0.6,
    mergeClusterThreshold: 1.5,
    showPoints: true,
    renderCameraPoints: true,
    showImage: false,
    showAllPoints: true,
    radiusSigmaK: 3.0,
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

// ensure these are defined at top-level:
let centerCubes = [];

function distanceAlongDirection(p1, p2, direction) {
    const u = direction.clone().normalize();
    return p2.clone().sub(p1).dot(u);
}

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

function clusterTrees() {
    if (!groundMesh) segmentGround();

    // — clear previous visuals —
    centerCubes.forEach(c => scene.remove(c));
    centerCubes = [];
    if (vegPointsMesh) scene.remove(vegPointsMesh);

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

    // 6) IQR‐based height filter along normal (rand. sampling)
    clusters = clusters.filter(cluster => {
        const n = cluster.length;
        if (n < 2) return false;
        const sampleCount = Math.min(10000, n * (n - 1) / 2);
        const dists = [];
        for (let k = 0; k < sampleCount; k++) {
            let i = Math.floor(Math.random() * n);
            let j;
            do { j = Math.floor(Math.random() * n); } while (j === i);
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

    // reset exported stats
    treeStats = [];

    // 9) visualize merged clusters
    const vegPos = [];
    const vegCols = [];

    treeMeshes.forEach(tree => scene.remove(tree));

    clusterData.forEach(data => {
        const col = new THREE.Color(Math.random(), Math.random(), Math.random());
        data.indices.forEach(idx => {
            vegPos.push(...posArray.slice(idx * 3, idx * 3 + 3));
            vegCols.push(col.r, col.g, col.b);
        });
        const cube = new THREE.Mesh(
            new THREE.BoxGeometry(0.2, 0.2, 0.2),
            new THREE.MeshStandardMaterial({ color: col })
        );
        cube.position.copy(data.centroid);
        scene.add(cube);

        // compute height and circle-based footprint using a slice near average camera elevation
        let minY = Infinity, maxY = -Infinity;
        const gp2 = groundMesh.geometry.attributes.position.array;
        const A2 = new THREE.Vector3().fromArray(gp2, 0);
        const B2 = new THREE.Vector3().fromArray(gp2, 3);
        const C2 = new THREE.Vector3().fromArray(gp2, 6);
        const groundNormal = new THREE.Vector3().crossVectors(B2.clone().sub(A2), C2.clone().sub(A2)).normalize();
        const I3 = new THREE.Matrix3().identity();
        const nnT2 = new THREE.Matrix3().set(
            groundNormal.x * groundNormal.x, groundNormal.x * groundNormal.y, groundNormal.x * groundNormal.z,
            groundNormal.y * groundNormal.x, groundNormal.y * groundNormal.y, groundNormal.y * groundNormal.z,
            groundNormal.z * groundNormal.x, groundNormal.z * groundNormal.y, groundNormal.z * groundNormal.z
        );
        const Pplane = new THREE.Matrix3();
        Pplane.elements = I3.elements.map((v, i) => v - nnT2.elements[i]);
        const e1p = (Math.abs(groundNormal.x) < 0.9
            ? groundNormal.clone().cross(new THREE.Vector3(1, 0, 0))
            : groundNormal.clone().cross(new THREE.Vector3(0, 1, 0))
        ).normalize();
        const e2p = groundNormal.clone().cross(e1p).normalize();

        // average camera elevation along groundNormal
        let meanCamH = 0;
        if (cameraPoints.length > 0) {
            const camCentroid = cameraPoints.reduce((s, p) => s.add(p), new THREE.Vector3()).divideScalar(cameraPoints.length);
            // elevation proxy: dot with normal relative to ground centroid (A2)
            meanCamH = camCentroid.clone().sub(A2).dot(groundNormal);
        }

        const band = params.elevationBand;
        const projectedSlice = [];
        const elevationBandPositions = [];
        const cx = data.centroid.x, cz = data.centroid.z;
        data.indices.forEach(idx => {
            const x = posArray[idx * 3], y = posArray[idx * 3 + 1], z = posArray[idx * 3 + 2];
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            // elevation along ground normal
            const elev = new THREE.Vector3(x, y, z).sub(A2).dot(groundNormal);
            if (Math.abs(elev - meanCamH) <= band) {
                // project into plane UV
                const v = new THREE.Vector3(x, y, z).sub(A2).applyMatrix3(Pplane);
                const uCoord = v.dot(e1p), vCoord = v.dot(e2p);
                projectedSlice.push([uCoord, vCoord]);
                elevationBandPositions.push(x, y, z);
            }
        });
        const height = maxY - minY;

        let fittedCenter = null;
        let fittedRadius = 0;
        const circle = fitCircle2D(projectedSlice);
        if (circle) {
            fittedCenter = { u: circle.cx, v: circle.cy };
            fittedRadius = circle.r;
        } else {
            // fallback to sigma-clipped radius around centroid
            const radialDistances = [];
            data.indices.forEach(idx => {
                const x = posArray[idx * 3], z = posArray[idx * 3 + 2];
                const dx = x - cx, dz = z - cz;
                radialDistances.push(Math.sqrt(dx * dx + dz * dz));
            });
            fittedRadius = computeRobustRadius(radialDistances);
        }

        // 2) place tree so its base sits at the cluster bottom
        let bottomPos;
        if (fittedCenter) {
            // map fitted (u,v) back to world using plane frame at A2
            const centerWorld = A2.clone()
                .add(e1p.clone().multiplyScalar(fittedCenter.u))
                .add(e2p.clone().multiplyScalar(fittedCenter.v));
            bottomPos = new THREE.Vector3(centerWorld.x, minY, centerWorld.z);
        } else {
            bottomPos = new THREE.Vector3(cx, minY, cz);
        }
        const tree = createProceduralTree(
      /*levels=*/3,
      /*length=*/height,
      /*radius=*/fittedRadius,
            bottomPos,
            new THREE.Vector3(0, 1, 0)
        );

        scene.add(tree);
        // record stats for export
        treeStats.push({
            centerX: bottomPos.x,
            centerY: bottomPos.y,
            centerZ: bottomPos.z,
            radius: fittedRadius,
            diameter: 2 * fittedRadius,
            height: height,
            points: data.indices.length,
        });
        treeMeshes.push(tree);
        if(params.showPoints){
            centerCubes.push(cube);
        }
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

    // visualize elevation-band points if requested
    if (elevationBandPointsMesh) { scene.remove(elevationBandPointsMesh); elevationBandPointsMesh = null; }
    if (params.showElevationBandPoints && elevationBandPositions.length >= 3) {
        const geomEB = new THREE.BufferGeometry();
        geomEB.setAttribute('position', new THREE.Float32BufferAttribute(elevationBandPositions, 3));
        elevationBandPointsMesh = new THREE.Points(
            geomEB,
            new THREE.PointsMaterial({
                size: 0.12,
                color: 0x0000ff,
                sizeAttenuation: false,
                depthTest: false,
                transparent: true,
                opacity: 1.0
            })
        );
        scene.add(elevationBandPointsMesh);
    }

    showGroundPoint()
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
