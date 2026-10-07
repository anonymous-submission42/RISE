// Footstep plan + terrain, ported from g1_controller/cmd:
//   gen_cmd.py N --realistic          -> sampleLocalSteps()
//   convert_footcommand_2_global.py   -> toGlobal()
//   gen_rocky_mountain.py             -> terrainXml()
// The CSV round trip is kept only as its rounding (3 decimals local, 4 global).

import { wrapToPi } from './math.js';

// Narrower x / y than gen_cmd.py "02. 3d random footstep sampling" (x, y: [0.2, 0.4])
// so that most random 100-step plans are completed: 97/100 seeds walked to the
// end and stood with these ranges, 8/30 with the original.
const RANGE = { x: [0.2, 0.3], y: [0.2, 0.3], z: [-0.15, 0.2], yaw: [-0.4, 0.4] };
const NOMINAL_Y = 0.237;
const SMALL_FRAC = 0.5; // --realistic
const DEFAULTS = { ssp: 0.7, dsp: 0.1, height: 0.07, comZ: 0.01, start: 'R' };

// Spawn foot world poses [x, y, z, yaw] (deploy.yaml global_init_lfoot / rfoot).
export const INIT_LFOOT = { x: -0.01259, y: 0.1185, z: 0.03458, yaw: 0 };
export const INIT_RFOOT = { x: -0.01259, y: -0.1185, z: 0.03458, yaw: 0 };

const round = (v, d) => Number(v.toFixed(d));

export function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// gen_cmd.py build_rows(): first step fixed, last step a stop step, and with
// --realistic a large x / y / |z| sample is followed by a non-extreme one.
export function sampleLocalSteps(n, seed) {
  const rand = mulberry32(seed);
  const uniform = (lo, hi) => lo + (hi - lo) * rand();
  const capped = (lo, hi, cap) => uniform(lo, cap ? lo + SMALL_FRAC * (hi - lo) : hi);
  const isLarge = (v, [lo, hi]) => hi > lo && v > lo + SMALL_FRAC * (hi - lo);
  const zInner = ([lo, hi]) => (lo < 0 && hi > 0) ? [SMALL_FRAC * lo, SMALL_FRAC * hi]
                                                  : [lo, lo + SMALL_FRAC * (hi - lo)];
  const zLarge = v => { const [lo, hi] = zInner(RANGE.z); return v < lo || v > hi; };

  const rows = [];
  let recX = false, recY = false, recZ = false;
  for (let i = 0; i < n; i++) {
    const foot = i % 2 === 0 ? DEFAULTS.start : (DEFAULTS.start === 'R' ? 'L' : 'R');
    let x, y, z, yaw;
    if (i === n - 1) {
      [x, y, z, yaw] = [0, NOMINAL_Y, 0, 0];
    }
    else if (i === 0) {
      [x, y, z, yaw] = [0.2, 0.237, 0, 0];
    } else {
      x = round(capped(...RANGE.x, recX), 3);
      y = round(capped(...RANGE.y, recY), 3);
      z = round(recZ ? uniform(...zInner(RANGE.z)) : uniform(...RANGE.z), 3);
      yaw = uniform(...RANGE.yaw);
      recX = isLarge(x, RANGE.x);
      recY = isLarge(y, RANGE.y);
      recZ = zLarge(z);
    }
    // The stop step (x = 0) lands beside the stance foot, where the same foot's
    // previous stone also reaches; a flat second-to-last step keeps the last
    // three stones level so that overlap cannot catch the foot.
    if (i === n - 2) z = 0;
    rows.push({ foot, phase: foot === 'R' ? 0 : 1, x: round(x, 3), y: round(y, 3), z: round(z, 3),
                yaw: round(yaw, 3), ssp_t: DEFAULTS.ssp, dsp_t: DEFAULTS.dsp,
                height: DEFAULTS.height, com_z: DEFAULTS.comZ });
  }
  return rows;
}

// Accumulate local steps into world-frame swing-foot landings.
export function toGlobal(rows) {
  let st = { ...(rows[0].phase === 0 ? INIT_LFOOT : INIT_RFOOT) };
  return rows.map(r => {
    const sy = (r.phase === 0 ? -1 : 1) * Math.abs(r.y);
    const c = Math.cos(st.yaw), s = Math.sin(st.yaw);
    const sw = { x: st.x + c * r.x - s * sy, y: st.y + s * r.x + c * sy,
                 z: st.z + r.z, yaw: wrapToPi(st.yaw + r.yaw) };
    st = sw;
    return { foot: r.foot, phase: r.phase,
             x: round(sw.x, 4), y: round(sw.y, 4), z: round(sw.z, 4), yaw: round(sw.yaw, 4),
             ssp_t: r.ssp_t, dsp_t: r.dsp_t, height: r.height, com_z: r.com_z };
  });
}

const STONE_OFFSET = [0.035, 0, -0.03458];
const f = v => v.toFixed(4);
const yawQuat = yaw => `${Math.cos(yaw / 2).toFixed(6)} 0 0 ${Math.sin(yaw / 2).toFixed(6)}`;

// Stone landings: target shifted by `off` in the foot yaw frame (x, y) and world up (z).
function stoneLandings(targets, off) {
  return targets.map(t => ({
    x: t.x + Math.cos(t.yaw) * off[0] - Math.sin(t.yaw) * off[1],
    y: t.y + Math.sin(t.yaw) * off[0] + Math.cos(t.yaw) * off[1],
    z: t.z + off[2], yaw: t.yaw,
  }));
}

function groundPlane(stones, zGround, margin, rgba) {
  const xs = stones.map(s => s.x), ys = stones.map(s => s.y);
  const cx = 0.5 * (Math.min(...xs) + Math.max(...xs)), cy = 0.5 * (Math.min(...ys) + Math.max(...ys));
  const px = 0.5 * (Math.max(...xs) - Math.min(...xs)) + margin;
  const py = 0.5 * (Math.max(...ys) - Math.min(...ys)) + margin;
  return `<geom name="footstep_ground" type="plane" group="2" pos="${f(cx)} ${f(cy)} ${f(zGround)}" size="${f(px)} ${f(py)} 0.1" rgba="${rgba}"/>`;
}

// Target markers as in gen_rocky_mountain.py: a translucent copy of the foot
// link (ankle_roll_link mesh) posed at the raw target, group 5, no collision.
function targetMarkers(targets) {
  return targets.map((t, i) => {
    const left = t.foot === 'L';
    const color = left ? '0.2 0.25 0.6 0.5' : '0.6 0.2 0.25 0.5';
    const mesh = left ? 'left_ankle_roll_link' : 'right_ankle_roll_link';
    return `<geom name="target_${t.foot}_${i}" type="mesh" mesh="${mesh}" group="5" ` +
      `pos="${f(t.x)} ${f(t.y)} ${f(t.z)}" quat="${yawQuat(t.yaw)}" rgba="${color}" contype="0" conaffinity="0"/>`;
  });
}

// gen_rocky_mountain.py defaults (box rocks on a square grid, box stones).
const ROCKY = {
  spacing: 0.2, width: 0.2, startClear: 0.3, startClearFront: 0.3, clearance: 0,
  flushRadius: 0.3, flushBlend: 0.3, flushDrop: 0.2, noise: 0.03, maxCellYaw: 1.79,
  corridor: [0.45, 10.5], terrainMargin: 1.0, idwPower: 2, minHeight: 0.015, seed: 7,
  stone: [0.1, 0.1], planeMargin: 3.0,
};
const ROCK_RGB = [0.6, 0.5, 0.4];
export const ROCK_GROUP = 4;

const clamp01 = v => Math.max(0, Math.min(1, v));

// Deterministic pseudo-random value in [0, 1) for grid cell (i, j).
function cellHash(i, j, seed, salt = 0) {
  const v = Math.sin(i * 12.9898 + j * 78.233 + seed * 37.719 + salt * 4.581) * 43758.5453;
  return v - Math.floor(v);
}

// Height-shaded rock color with per-cell variation.
function rockRgba(top, zLo, zSpan, i, j, seed, salt = 2) {
  const shade = 0.72 + 0.38 * clamp01((top - zLo) / zSpan);
  const v = (cellHash(i, j, seed, salt) - 0.5) * 0.12;
  return [v, 0.9 * v, 0.8 * v].map((d, k) => clamp01((ROCK_RGB[k] + d) * shade).toFixed(3)).join(' ') + ' 1';
}

function segmentDist(px, py, a, b) {
  const abx = b.x - a.x, aby = b.y - a.y, apx = px - a.x, apy = py - a.y;
  const ab2 = abx * abx + aby * aby;
  if (ab2 < 1e-12) return Math.hypot(apx, apy);
  const t = clamp01((apx * abx + apy * aby) / ab2);
  return Math.hypot(px - (a.x + t * abx), py - (a.y + t * aby));
}

function pathDistance(px, py, path) {
  let d = Infinity;
  for (let k = 0; k < path.length - 1; k++) d = Math.min(d, segmentDist(px, py, path[k], path[k + 1]));
  return d;
}

// Inverse-distance-weighted surface height; eps keeps weights finite on top of a point.
function idwHeight(px, py, points, power, eps = 0.02) {
  let wsum = 0, zsum = 0;
  for (const p of points) {
    const d2 = (px - p.x) ** 2 + (py - p.y) ** 2 + eps * eps;
    const w = 1 / d2 ** (power / 2);
    wsum += w;
    zsum += w * p.z;
  }
  return zsum / wsum;
}

// --flush-drop behind the stone (stone local -x, the approach side); 0 ahead.
function flushDrop(cx, cy, s, drop) {
  return Math.cos(s.yaw) * (cx - s.x) + Math.sin(s.yaw) * (cy - s.y) < 0 ? drop : 0;
}

// Rock columns filling the corridor around the path, heights interpolated from
// the stone landings (gen_rocky_mountain.py build_rock_cells()).
function rockCells(stones, zGround, R) {
  const points = [{ x: 0, y: 0, z: 0 }, ...stones];
  const xs = points.map(p => p.x), ys = points.map(p => p.y), zs = points.map(p => p.z);
  const zLo = Math.min(...zs), zSpan = Math.max(Math.max(...zs) - zLo, 1e-6);
  const cell = R.spacing, half = 0.5 * R.width;
  const x0 = Math.min(...xs) - R.terrainMargin, x1 = Math.max(...xs) + R.terrainMargin;
  const y0 = Math.min(...ys) - R.terrainMargin, y1 = Math.max(...ys) + R.terrainMargin;
  const nx = Math.max(1, Math.ceil((x1 - x0) / cell)), ny = Math.max(1, Math.ceil((y1 - y0) / cell));
  const [w0, w1] = R.corridor;
  const keepOut = R.startClear + half * Math.SQRT2;
  const r0 = R.flushRadius, r1 = r0 + R.flushBlend;

  const lines = [];
  for (let j = 0; j < ny; j++) {
    const cy = y0 + (j + 0.5) * cell;
    for (let i = 0; i < nx; i++) {
      const cx = x0 + (i + 0.5) * cell;
      // keep the ground behind the spawn point bare so the robot starts on the platform
      if (R.startClear > 0 && cx < R.startClearFront && Math.hypot(cx, cy) < keepOut) continue;
      const d = pathDistance(cx, cy, points);
      if (d >= w1) continue;
      const falloff = d <= w0 ? 1 : 0.5 * (1 + Math.cos(Math.PI * (d - w0) / (w1 - w0)));
      const baseTop = idwHeight(cx, cy, points, R.idwPower) - R.clearance - R.noise * cellHash(i, j, R.seed);

      // flush zone: min over stones in range so a higher neighbor never blocks a lower landing
      let zIn = Infinity, near = null, dNear = Infinity;
      for (const s of stones) {
        const ds = Math.hypot(cx - s.x, cy - s.y);
        if (ds <= r0) zIn = Math.min(zIn, s.z - flushDrop(cx, cy, s, R.flushDrop));
        if (ds < dNear) { dNear = ds; near = s; }
      }
      let top = baseTop;
      if (zIn < Infinity) top = zIn;
      else if (dNear < r1) {
        const t = 0.5 * (1 - Math.cos(Math.PI * (dNear - r0) / R.flushBlend));
        top = (1 - t) * (near.z - flushDrop(cx, cy, near, R.flushDrop)) + t * baseTop;
      }
      top = zGround + (top - zGround) * falloff;

      const h = top - zGround;
      if (h < R.minHeight) continue;
      const yaw = (cellHash(i, j, R.seed, 1) - 0.5) * 2 * R.maxCellYaw;
      lines.push(`<geom name="rock_${lines.length}" type="box" group="${ROCK_GROUP}" size="${f(half)} ${f(half)} ${f(h / 2)}" ` +
        `pos="${f(cx)} ${f(cy)} ${f(zGround + h / 2)}" quat="${yawQuat(yaw)}" rgba="${rockRgba(top, zLo, zSpan, i, j, R.seed)}"/>`);
    }
  }
  return { lines, zLo, zSpan };
}

// gen_rocky_mountain.py: spawn platform, ground plane, rock columns forming a
// continuous mountain under the path, stepping stones and target markers.
export function terrainXml(targets) {
  const R = ROCKY, off = STONE_OFFSET, rock = '0.6 0.5 0.4 1';
  const stones = stoneLandings(targets, off);
  const zGround = Math.min(0, ...stones.map(s => s.z));
  const { lines: rocks, zLo, zSpan } = rockCells(stones, zGround, R);

  const lines = [
    `<geom name="start_platform" type="box" group="2" size="0.1000 0.1700 0.1000" pos="${f(off[0])} 0 -0.1000" rgba="${rock}"/>`,
    groundPlane(stones, zGround, R.planeMargin, rock),
    ...rocks,
  ];
  const [hx, hy] = R.stone;
  stones.forEach((s, i) => {
    const h = s.z - zGround;
    if (h < 1e-4) return;
    lines.push(`<geom name="stone_${i}" type="box" group="2" size="${f(hx)} ${f(hy)} ${f(h / 2)}" ` +
      `pos="${f(s.x)} ${f(s.y)} ${f(zGround + h / 2)}" quat="${yawQuat(s.yaw)}" rgba="${rockRgba(s.z, zLo, zSpan, i, 0, R.seed, 3)}"/>`);
  });
  return [...lines, ...targetMarkers(targets)].join('\n');
}

// The sampled random walk can curl back onto itself, putting a later (often
// metres tall) pillar on the spawn or on an earlier stone. Such plans are
// skipped: true if a target comes within `r` (xy) of the spawn or of a target
// three or more steps earlier.
export function planCrossesItself(targets, r = 0.4) {
  for (let j = 2; j < targets.length; j++) {
    if (Math.hypot(targets[j].x, targets[j].y) < r) return true;
    for (let i = 0; i < j - 2; i++)
      if (Math.hypot(targets[j].x - targets[i].x, targets[j].y - targets[i].y) < r) return true;
  }
  return false;
}

// Insert the terrain block before the last </worldbody> (gen_footstep_scene.py inject()).
export function injectTerrain(sceneXml, block) {
  const i = sceneXml.lastIndexOf('</worldbody>');
  if (i < 0) throw new Error('no </worldbody> in scene XML');
  return sceneXml.slice(0, i) + block + '\n' + sceneXml.slice(i);
}
