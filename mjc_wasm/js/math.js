// Port of g1_controller include/isaaclab/utils/math_utils.h.
// Vectors are [x, y, z]; quaternions are [w, x, y, z] (Isaac Lab convention).

export const f32 = Math.fround;

export function wrapToPi(a) {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const norm = a => Math.hypot(a[0], a[1], a[2]);

export function quatNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export const quatConj = q => [q[0], -q[1], -q[2], -q[3]];

function mulRaw(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

export const quatMul = (a, b) => quatNormalize(mulRaw(a, b));

export function quatApply(q, v) {
  const [w, x, y, z] = quatNormalize(q);
  // v + 2w (u x v) + 2 u x (u x v)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty),
          v[1] + w * ty + (z * tx - x * tz),
          v[2] + w * tz + (x * ty - y * tx)];
}

export const quatApplyInverse = (q, v) => quatApply(quatConj(q), v);

export function yawQuat(q) {
  const [qw, qx, qy, qz] = q;
  const yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
  return [Math.cos(0.5 * yaw), 0, 0, Math.sin(0.5 * yaw)];
}

export function quatFromEulerXyz(roll, pitch, yaw) {
  const cr = Math.cos(roll * 0.5), sr = Math.sin(roll * 0.5);
  const cp = Math.cos(pitch * 0.5), sp = Math.sin(pitch * 0.5);
  const cy = Math.cos(yaw * 0.5), sy = Math.sin(yaw * 0.5);
  return quatNormalize([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ]);
}

export function eulerXyzFromQuat(q) {
  const [qw, qx, qy, qz] = q;
  const roll = Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
  const sinp = Math.max(-1, Math.min(1, 2 * (qw * qy - qz * qx)));
  const pitch = Math.asin(sinp);
  const yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
  return [roll, pitch, yaw];
}

export const yawOf = q => wrapToPi(eulerXyzFromQuat(q)[2]);

// t02 = t01 + q01 * t12
export const combineFrameTransformsPos = (t01, q01, t12) => add(t01, quatApply(q01, t12));

// (t12, q12) = subtract_frame_transforms(t01, q01, t02, q02)
export function subtractFrameTransforms(t01, q01, t02, q02) {
  const inv = quatConj(q01);
  return [quatApply(inv, sub(t02, t01)), quatMul(inv, q02)];
}

export const subtractFrameTransformsPos = (t01, q01, t02) => quatApply(quatConj(q01), sub(t02, t01));

// Cubic with f(0)=p0, f(T)=p1, f'(0)=v0, f'(T)=v1. Returns [pos, vel].
export function cubic(p0, v0, p1, v1, T, t) {
  if (T <= 1e-9) return [p1, v1];
  t = Math.max(0, Math.min(T, t));
  const a = (2 * (p0 - p1) + v0 * T + v1 * T) / (T * T * T);
  const b = (3 * (p1 - p0) - 2 * v0 * T - v1 * T) / (T * T);
  return [a * t * t * t + b * t * t + v0 * t + p0, 3 * a * t * t + 2 * b * t + v0];
}

// static_cast<int>(a / b) evaluated in float32, as the C++ controller does, so
// tick counts such as 0.7 / 0.02 land on the same integer.
export const ticks = (a, b) => Math.trunc(f32(f32(a) / f32(b)));
