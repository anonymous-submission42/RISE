import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'mujoco';
import { sampleLocalSteps, toGlobal, terrainXml, injectTerrain, planCrossesItself } from './js/plan.js';
import { Mlp } from './js/policy.js';
import { Controller } from './js/controller.js';

const ASSETS = 'assets/';
const ROBOT_FILE = 'g1_29dof.xml';
const N_STEPS = 100;
const DEFAULT_SEED = 6;
const POLICY = { file: 'policy/260808_1834_3d_nolcp.bin', dims: [660, 512, 256, 128, 12] };
const TARGET_GROUP = 5; // foot-target markers (plan.js terrainXml)

const $ = id => document.getElementById(id);
const setProgress = (frac, msg) => {
  $('loading-text').textContent = msg;
  document.querySelector('#loading .fill').style.width = `${(100 * frac).toFixed(0)}%`;
};

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

// Robot MJCF + meshes into a MuJoCo virtual file system (kept for recompiles).
async function loadRobotVfs(mujoco) {
  const robotXml = await fetchBytes(ASSETS + 'g1/' + ROBOT_FILE);
  const doc = new DOMParser().parseFromString(new TextDecoder().decode(robotXml), 'text/xml');
  const meshDir = doc.querySelector('compiler')?.getAttribute('meshdir') ?? '';
  const files = [...new Set([...doc.querySelectorAll('mesh[file]')].map(e => e.getAttribute('file')))];
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer(ROBOT_FILE, robotXml);
  let done = 0;
  await Promise.all(files.map(async f => {
    const path = meshDir ? `${meshDir}/${f}` : f;
    vfs.addBuffer(path, await fetchBytes(ASSETS + 'g1/' + path));
    setProgress(++done / files.length, `Loading meshes (${done}/${files.length})…`);
  }));
  return vfs;
}

// gen_cmd.py N --realistic + convert_footcommand_2_global.py, skipping plans that
// curl back onto themselves (see planCrossesItself).
function samplePlan(seed) {
  for (;; seed++) {
    const plan = toGlobal(sampleLocalSteps(N_STEPS, seed));
    if (!planCrossesItself(plan)) return { plan, seed };
  }
}

// ---------------------------------------------------------------- rendering

function checkerTexture(rgba, tiles) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const shade = k => `rgb(${rgba.slice(0, 3).map(v => Math.round(255 * v * k)).join(',')})`;
  g.fillStyle = shade(1.0); g.fillRect(0, 0, 64, 64);
  g.fillStyle = shade(0.85); g.fillRect(0, 0, 32, 32); g.fillRect(32, 32, 32, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(tiles[0], tiles[1]);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// One Three.js mesh per visual geom: robot visuals (group 1), terrain (group 2)
// and foot-target markers (group 5). Group 0 holds the robot's collision geoms.
function buildGeoms(mujoco, model, group) {
  const T = mujoco.mjtGeom;
  const meshCache = new Map();
  const objects = [];
  for (let g = 0; g < model.ngeom; g++) {
    const grp = model.geom_group[g];
    if (grp !== 1 && grp !== 2 && grp !== TARGET_GROUP) continue;
    const type = model.geom_type[g];
    const size = model.geom_size.subarray(3 * g, 3 * g + 3);
    const rgba = Array.from(model.geom_rgba.subarray(4 * g, 4 * g + 4));
    const color = new THREE.Color(rgba[0], rgba[1], rgba[2]).convertSRGBToLinear();
    let geometry, material;
    if (type === T.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(2 * size[0], 2 * size[1]);
      material = new THREE.MeshStandardMaterial({ map: checkerTexture(rgba, [size[0], size[1]]), roughness: 0.95 });
    } else {
      material = new THREE.MeshStandardMaterial({ color, roughness: grp === 1 ? 0.55 : 0.85, metalness: 0.05,
                                                  flatShading: type === T.mjGEOM_MESH.value,
                                                  transparent: rgba[3] < 1, opacity: rgba[3], depthWrite: rgba[3] >= 1 });
      if (type === T.mjGEOM_MESH.value) {
        const id = model.geom_dataid[g];
        if (!meshCache.has(id)) {
          const va = model.mesh_vertadr[id], vn = model.mesh_vertnum[id];
          const fa = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.BufferAttribute(model.mesh_vert.slice(3 * va, 3 * (va + vn)), 3));
          geo.setIndex(new THREE.BufferAttribute(new Uint32Array(model.mesh_face.slice(3 * fa, 3 * (fa + fn))), 1));
          geo.computeVertexNormals();
          meshCache.set(id, geo);
        }
        geometry = meshCache.get(id);
      } else if (type === T.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0], 16, 12);
      } else if (type === T.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
      } else {
        continue;
      }
    }
    const obj = new THREE.Mesh(geometry, material);
    obj.matrixAutoUpdate = false;
    obj.castShadow = type !== T.mjGEOM_PLANE.value && grp !== TARGET_GROUP;
    obj.receiveShadow = true;
    group.add(obj);
    objects.push({ g, obj, grp, static: grp !== 1 });
  }
  return objects;
}

function syncGeoms(data, objects, all) {
  const xpos = data.geom_xpos, xmat = data.geom_xmat;
  for (const { g, obj, static: s } of objects) {
    if (s && !all) continue;
    const p = 3 * g, r = 9 * g;
    obj.matrix.set(
      xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
      xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
      xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
      0, 0, 0, 1);
  }
}

function disposeGroup(group) {
  group.traverse(o => {
    if (!o.isMesh) return;
    o.geometry.dispose();
    o.material.map?.dispose();
    o.material.dispose();
  });
  group.clear();
}

// ---------------------------------------------------------------- app

async function main() {
  setProgress(0, 'Loading MuJoCo…');
  const mujoco = await loadMujoco();
  const [vfs, sceneXml, inertia, weights] = await Promise.all([
    loadRobotVfs(mujoco),
    fetch(ASSETS + 'g1/scene_footstep.xml').then(r => r.text()),
    fetch(ASSETS + 'g1/urdf_inertia.json').then(r => r.json()),
    fetchBytes(ASSETS + POLICY.file),
  ]);
  const mlp = new Mlp(POLICY.dims, weights.buffer);

  // three.js
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  $('viewport').appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1b2330);
  scene.fog = new THREE.Fog(0x1b2330, 10, 30);
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 200);
  camera.up.set(0, 0, 1);
  camera.position.set(-1.6, -2.6, 1.6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.6);
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x303030, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 20 });
  scene.add(sun, sun.target);
  const world = new THREE.Group();
  scene.add(world);
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // simulation
  let sim = null;   // { model, data, ctl, geoms, plan, seed }
  let paused = false;

  function load(seed) {
    if (sim) {
      sim.ctl.dispose();
      sim.data.delete();
      sim.model.delete();
      disposeGroup(world);
    }
    const { plan, seed: used } = samplePlan(seed);
    const model = mujoco.MjModel.from_xml_string(injectTerrain(sceneXml, terrainXml(plan)), vfs);
    const data = new mujoco.MjData(model);
    const ctl = new Controller(mujoco, model, data, { inertia, mlp, plan });
    const geoms = buildGeoms(mujoco, model, world);
    syncGeoms(data, geoms, true);
    showTargets(geoms);
    sim = { model, data, ctl, geoms, plan, seed: used };
    $('seed').value = used;
    resetCamera();
  }

  function restart() {
    sim.ctl.reset();
    resetCamera();
  }

  function resetCamera() {
    const p = sim.data.qpos;
    controls.target.set(p[0], p[1], p[2] - 0.1);
    camera.position.set(p[0] - 1.6, p[1] - 2.6, p[2] + 0.9);
    controls.update();
  }

  // Keep the robot in view: move the orbit target (and camera with it) toward the pelvis.
  const follow = new THREE.Vector3();
  function followRobot() {
    const p = sim.data.qpos;
    follow.set(p[0], p[1], p[2] - 0.1).sub(controls.target).multiplyScalar(0.05);
    controls.target.add(follow);
    camera.position.add(follow);
    sun.position.set(p[0] + 2, p[1] - 3, p[2] + 5);
    sun.target.position.set(p[0], p[1], p[2]);
  }

  function showTargets(geoms) {
    const on = $('chk-targets').checked;
    for (const { obj, grp } of geoms) if (grp === TARGET_GROUP) obj.visible = on;
  }

  $('chk-targets').onchange = () => showTargets(sim.geoms);
  $('btn-new').onclick = () => load(Math.floor(Math.random() * 1e6));
  $('btn-seed').onclick = () => load(Math.max(0, parseInt($('seed').value, 10) || 0));
  $('btn-restart').onclick = restart;
  $('btn-pause').onclick = () => { paused = !paused; $('btn-pause').textContent = paused ? 'Resume' : 'Pause'; };

  setProgress(1, 'Compiling model…');
  load(DEFAULT_SEED);
  $('loading').remove();

  const status = $('status');
  let last = performance.now();
  function frame(now) {
    const wall = Math.min((now - last) / 1000, 1 / 30); // cap catch-up after tab switches
    last = now;
    const { data, ctl } = sim;
    if (!paused) {
      const tEnd = data.time + wall;
      while (data.time < tEnd) ctl.step();
    }
    syncGeoms(data, sim.geoms, false);
    followRobot();
    controls.update();
    renderer.render(scene, camera);
    status.innerHTML = statusText(sim);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function statusText({ ctl, plan, data }) {
  const n = ctl.stepErrors.length;
  const errs = ctl.stepErrors.map(e => Math.hypot(...e) * 100);
  const mean = n ? errs.reduce((a, b) => a + b, 0) / n : 0;
  const state = {
    fixstand: 'Standing (FixStand)',
    footstep: 'Walking (Footstep)',
    hold: 'Plan complete — joint control',
    passive: 'Fell — Passive',
  }[ctl.mode];
  return `<b>${state}</b><br>step ${n} / ${plan.length} · t = ${data.time.toFixed(1)} s<br>` +
    (n ? `landing error: last ${errs[n - 1].toFixed(1)} cm · mean ${mean.toFixed(1)} cm` : '&nbsp;');
}

main().catch(err => {
  console.error(err);
  $('loading-text').textContent = `Error: ${err.message}`;
});
