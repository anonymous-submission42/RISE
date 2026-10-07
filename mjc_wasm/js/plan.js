// Footstep plan + terrain, ported from g1_controller/cmd:
//   gen_cmd.py N --realistic          -> sampleLocalSteps()
//   convert_footcommand_2_global.py   -> toGlobal()
//   gen_footstep_scene.py             -> terrainXml()
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

// gen_footstep_scene.py defaults: box stones 0.1 x 0.05 half-size, pillars from
// the ground to each stone top, stones shifted 0.035 m forward in the foot yaw
// frame and lowered by the ankle height, a 10 cm spawn platform with its top at
// z=0, a ground plane at min(stone top, 0) and foot-shaped target markers.
export function terrainXml(targets) {
  const [hx, hy] = [0.1, 0.05], off = [0.035, 0, -0.03458], margin = 3.0;
  const stone = '0.45 0.28 0.12 1';
  const f = v => v.toFixed(4);
  const stones = targets.map(t => ({
    x: t.x + Math.cos(t.yaw) * off[0] - Math.sin(t.yaw) * off[1],
    y: t.y + Math.sin(t.yaw) * off[0] + Math.cos(t.yaw) * off[1],
    z: t.z + off[2], yaw: t.yaw,
  }));
  const zMin = Math.min(...stones.map(s => s.z));
  const zGround = zMin >= 0 ? 0 : zMin;
  const xs = stones.map(s => s.x), ys = stones.map(s => s.y);
  const cx = 0.5 * (Math.min(...xs) + Math.max(...xs)), cy = 0.5 * (Math.min(...ys) + Math.max(...ys));
  const px = 0.5 * (Math.max(...xs) - Math.min(...xs)) + margin;
  const py = 0.5 * (Math.max(...ys) - Math.min(...ys)) + margin;

  const lines = [
    `<geom name="start_platform" type="box" group="2" size="0.1000 0.1700 0.1000" pos="${f(off[0])} 0 -0.1000" rgba="${stone}"/>`,
    `<geom name="footstep_ground" type="plane" group="2" pos="${f(cx)} ${f(cy)} ${f(zGround)}" size="${f(px)} ${f(py)} 0.1" rgba="0.7 0.6 0.5 1"/>`,
  ];
  stones.forEach((s, i) => {
    const h = s.z - zGround;
    if (h < 1e-4) return;
    lines.push(`<geom name="footstep_box_${i}" type="box" group="2" size="${f(hx)} ${f(hy)} ${f(h / 2)}" ` +
      `pos="${f(s.x)} ${f(s.y)} ${f(zGround + h / 2)}" quat="${Math.cos(s.yaw / 2).toFixed(6)} 0 0 ${Math.sin(s.yaw / 2).toFixed(6)}" rgba="${stone}"/>`);
  });
  // Target markers as in gen_rocky_mountain.py: a translucent copy of the foot
  // link (ankle_roll_link mesh) posed at the raw target, group 5, no collision.
  targets.forEach((t, i) => {
    const left = t.foot === 'L';
    const color = left ? '0.2 0.25 0.6 0.5' : '0.6 0.2 0.25 0.5';
    const mesh = left ? 'left_ankle_roll_link' : 'right_ankle_roll_link';
    lines.push(`<geom name="target_${t.foot}_${i}" type="mesh" mesh="${mesh}" group="5" ` +
      `pos="${f(t.x)} ${f(t.y)} ${f(t.z)}" quat="${Math.cos(t.yaw / 2).toFixed(6)} 0 0 ${Math.sin(t.yaw / 2).toFixed(6)}" ` +
      `rgba="${color}" contype="0" conaffinity="0"/>`);
  });
  return lines.join('\n');
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
