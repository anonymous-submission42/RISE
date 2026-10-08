import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'mujoco';
import { sampleLocalSteps, toGlobal, terrainXml, ROCK_GROUP, injectTerrain, planCrossesItself } from './js/plan.js';
import { Mlp } from './js/policy.js';
import { Controller } from './js/controller.js';
import { ROBOTS } from './js/robots.js';

const ASSETS = 'assets/';
const SCENE_FILE = 'scene_footstep.xml';
const N_STEPS = 100;
const POLICY_DIMS = [660, 512, 256, 128, 12];
const TARGET_GROUP = 5; // foot-target markers (plan.js terrainXml)

const $ = id => document.getElementById(id);
const setProgress = (frac, msg) => {
  $('loading').hidden = false;
  $('loading-text').textContent = msg;
  document.querySelector('#loading .fill').style.width = `${(100 * frac).toFixed(0)}%`;
};

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

// Robot MJCF + meshes into a MuJoCo virtual file system (kept for recompiles).
async function loadRobotVfs(mujoco, robot) {
  const dir = ASSETS + robot.dir;
  const robotXml = await fetchBytes(dir + robot.robotFile);
  const doc = new DOMParser().parseFromString(new TextDecoder().decode(robotXml), 'text/xml');
  const meshDir = (doc.querySelector('compiler')?.getAttribute('meshdir') ?? '').replace(/\/+$/, '');
  const files = [...new Set([...doc.querySelectorAll('mesh[file]')].map(e => e.getAttribute('file')))];
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer(robot.robotFile, robotXml);
  let done = 0;
  await Promise.all(files.map(async f => {
    const path = meshDir ? `${meshDir}/${f}` : f;
    vfs.addBuffer(path, await fetchBytes(dir + path));
    setProgress(++done / files.length, `Loading ${robot.name} meshes (${done}/${files.length})…`);
  }));
  return vfs;
}

// Everything a robot needs besides the terrain and the policy: model files, URDF inertia.
async function loadRobotAssets(mujoco, robot) {
  const [vfs, sceneXml, inertia] = await Promise.all([
    loadRobotVfs(mujoco, robot),
    fetch(ASSETS + robot.dir + SCENE_FILE).then(r => r.text()),
    fetch(ASSETS + robot.dir + 'urdf_inertia.json').then(r => r.json()),
  ]);
  return { vfs, sceneXml, inertia };
}

async function loadPolicy(file) {
  return new Mlp(POLICY_DIMS, (await fetchBytes(ASSETS + file)).buffer);
}

// gen_cmd.py N --realistic + convert_footcommand_2_global.py, skipping plans that
// curl back onto themselves (see planCrossesItself).
function samplePlan(seed, robot) {
  for (;; seed++) {
    const plan = toGlobal(sampleLocalSteps(N_STEPS, seed, robot.plan), robot.plan);
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

// Rock columns are static world boxes, often thousands of them: one instanced
// draw call, posed once from the model (world body at the origin).
function rockInstances(model, ids) {
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0.05 }), ids.length);
  const { geom_pos: pos, geom_quat: quat, geom_size: size, geom_rgba: rgba } = model;
  const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion();
  const s = new THREE.Vector3(), c = new THREE.Color();
  ids.forEach((g, k) => {
    p.set(pos[3 * g], pos[3 * g + 1], pos[3 * g + 2]);
    q.set(quat[4 * g + 1], quat[4 * g + 2], quat[4 * g + 3], quat[4 * g]);
    s.set(2 * size[3 * g], 2 * size[3 * g + 1], 2 * size[3 * g + 2]);
    mesh.setMatrixAt(k, m.compose(p, q, s));
    mesh.setColorAt(k, c.setRGB(rgba[4 * g], rgba[4 * g + 1], rgba[4 * g + 2]).convertSRGBToLinear());
  });
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

// One Three.js mesh per visual geom: robot visuals (group 1), terrain (group 2)
// and foot-target markers (group 5), plus one instanced mesh for the rock
// columns (group 4). Group 0 holds the robot's collision geoms.
function buildGeoms(mujoco, model, group) {
  const T = mujoco.mjtGeom;
  const meshCache = new Map();
  const objects = [];
  const rocks = [];
  for (let g = 0; g < model.ngeom; g++) {
    const grp = model.geom_group[g];
    if (grp === ROCK_GROUP) { rocks.push(g); continue; }
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
  if (rocks.length) group.add(rockInstances(model, rocks));
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
  const assets = {};  // robot key -> loaded assets
  const policies = {}; // policy file -> Mlp
  let robotKey = 'g1';
  let policyFile = null;

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
    const robot = ROBOTS[robotKey];
    const { vfs, sceneXml, inertia } = assets[robotKey];
    const mlp = policies[policyFile];
    const { plan, seed: used } = samplePlan(seed, robot);
    const model = mujoco.MjModel.from_xml_string(injectTerrain(sceneXml, terrainXml(plan, robot.terrain)), vfs);
    const data = new mujoco.MjData(model);
    const ctl = new Controller(mujoco, model, data, { robot, inertia, mlp, plan });
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

  $('chk-targets').onchange = () => showTargets(sim.geoms);  $('btn-new').onclick = () => load(Math.floor(Math.random() * 1e6));
  $('btn-seed').onclick = () => load(Math.max(0, parseInt($('seed').value, 10) || 0));
  $('btn-restart').onclick = restart;
  $('btn-pause').onclick = () => { paused = !paused; $('btn-pause').textContent = paused ? 'Resume' : 'Pause'; };
  $('btn-ui').onclick = () => {
    const hidden = $('panel').classList.toggle('hidden');
    $('btn-ui').textContent = hidden ? 'Show UI' : 'Hide UI';
  };

  // Switch robot / policy: fetch assets and weights on first use, then build
  // the terrain for `seed`.
  let switching = false;
  async function select(key, file, seed) {
    switching = true;
    try {
      if (!assets[key]) assets[key] = await loadRobotAssets(mujoco, ROBOTS[key]);
      if (!policies[file]) {
        setProgress(1, 'Loading policy…');
        policies[file] = await loadPolicy(file);
      }
      robotKey = key;
      policyFile = file;
      setProgress(1, 'Compiling model…');
      load(seed);
    } finally {
      $('loading').hidden = true;
      $('robot').value = robotKey;
      $('policy').innerHTML = ROBOTS[robotKey].policies
        .map(p => `<option value="${p.file}">${p.name}</option>`).join('');
      $('policy').value = policyFile;
      switching = false;
    }
  }
  const selectRobot = key => select(key, ROBOTS[key].policies[0].file, ROBOTS[key].defaultSeed);
  $('robot').onchange = () => selectRobot($('robot').value).catch(showError);
  // a new policy restarts on the same terrain, so policies can be compared
  $('policy').onchange = () => select(robotKey, $('policy').value, sim.seed).catch(showError);

  await selectRobot(robotKey);

  const status = $('status');
  let last = performance.now();
  function frame(now) {
    const wall = Math.min((now - last) / 1000, 1 / 30); // cap catch-up after tab switches
    last = now;
    const { data, ctl } = sim;
    if (!paused && !switching) {
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

function showError(err) {
  console.error(err);
  $('loading').hidden = false;
  $('loading-text').textContent = `Error: ${err.message}`;
}

main().catch(showError);
