// Port of g1_controller include/isaaclab/utils/kinematics.h.
// Pinocchio is replaced by a scratch MjData on the same model with the root
// (pelvis / trunk) pinned at the origin, so every quantity is in the ROOT frame
// and base velocity is taken as zero (same assumption as the C++ helper).

import { wrapToPi, quatNormalize, quatConj, quatMul } from './math.js';

export const LEFT = 0, RIGHT = 1;

// Axis-angle vector of a unit quaternion (matches Kinematics::axis_angle_from_quat).
function axisAngle(q) {
  q = quatNormalize(q);
  if (q[0] < 0) q = q.map(v => -v);
  const mag = Math.hypot(q[1], q[2], q[3]);
  const half = Math.atan2(mag, q[0]);
  const angle = 2 * half;
  const f = Math.abs(angle) > 1e-6 ? angle / Math.sin(half) : 2;
  return [q[1] * f, q[2] * f, q[3] * f];
}

// Solve the symmetric positive-definite system A x = b (A: n x n row-major).
function solveSpd(A, b, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = i === j ? Math.sqrt(s) : s / L[j * n + j];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

export class Kinematics {
  // `inertia`: per-joint [{joint, mass, lever}] of the URDF the controller uses
  // (assets/<robot>/urdf_inertia.json, dumped from Pinocchio), applied over
  // MuJoCo body frames. Like pinocchio::centerOfMass on the fixed-base model,
  // the "universe" entry (root + its fixed links) is left out of the CoM.
  // `robot`: sdkJoints (motor order), legIds (left 6 then right 6), footBodies.
  constructor(mujoco, model, inertia, robot) {
    this.mj = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.ikData = new mujoco.MjData(model);
    this.jacp = new mujoco.DoubleBuffer(3 * model.nv);
    this.jacr = new mujoco.DoubleBuffer(3 * model.nv);

    const jointIds = robot.sdkJoints.map(n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n));
    this.n = jointIds.length;
    this.qadr = jointIds.map(j => model.jnt_qposadr[j]);
    this.dadr = jointIds.map(j => model.jnt_dofadr[j]);
    this.legIds = [robot.legIds.slice(0, 6), robot.legIds.slice(6, 12)];
    this.footBody = robot.footBodies.map(n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n));
    this.links = inertia.filter(l => l.joint !== 'universe').map(({ joint, mass, lever }) => ({
      body: model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, joint)],
      mass, lever,
    }));
    this.totalMass = this.links.reduce((m, l) => m + l.mass, 0);
    this.q = new Float64Array(this.n);
    this.qd = new Float64Array(this.n);
    this.foot = [null, null];
  }

  // q, qd: full-body arrays in SDK motor order.
  setState(q, qd) {
    this.q.set(q);
    this.qd.set(qd);
    this._fk(this.data, this.q);
    const d = this.data;
    for (const s of [LEFT, RIGHT]) {
      const b = this.footBody[s];
      // Pinocchio LOCAL_WORLD_ALIGNED linear velocity of the frame origin = Jp * qd.
      this.mj.mj_jacBody(this.model, d, this.jacp, this.jacr, b);
      const jp = this.jacp.GetView(), nv = this.model.nv;
      const lin = [0, 0, 0];
      for (let r = 0; r < 3; r++)
        for (let i = 0; i < this.n; i++) lin[r] += jp[r * nv + this.dadr[i]] * qd[i];
      this.foot[s] = {
        pos: Array.from(d.xpos.subarray(3 * b, 3 * b + 3)),
        quat: quatNormalize(Array.from(d.xquat.subarray(4 * b, 4 * b + 4))),
        lin,
      };
    }
    const xpos = d.xpos, xmat = d.xmat, com = [0, 0, 0];
    for (const { body: b, mass, lever: c } of this.links) {
      for (let r = 0; r < 3; r++) {
        const R = xmat.subarray(9 * b + 3 * r, 9 * b + 3 * r + 3);
        com[r] += mass * (xpos[3 * b + r] + R[0] * c[0] + R[1] * c[1] + R[2] * c[2]);
      }
    }
    this.com = com.map(v => v / this.totalMass);
  }

  dispose() {
    for (const o of [this.data, this.ikData, this.jacp, this.jacr]) o.delete();
  }

  footPos(s) { return this.foot[s].pos; }
  footQuat(s) { return this.foot[s].quat; }
  footLinVel(s) { return this.foot[s].lin; }
  comPos() { return this.com; }
  legQ(s) { return this.legIds[s].map(i => this.q[i]); }

  _fk(d, q) {
    const qpos = d.qpos;
    qpos.fill(0, 0, 7);
    qpos[3] = 1;
    for (let i = 0; i < this.n; i++) qpos[this.qadr[i]] = q[i];
    this.mj.mj_kinematics(this.model, d);
    this.mj.mj_comPos(this.model, d);
  }

  // Damped least-squares IK for one leg (targets in the root frame).
  diffIkLeg(s, targetPos, targetQuat, qInit, iters, lambda, posTol) {
    const leg = this.legIds[s];
    const q = Float64Array.from(this.q);
    for (let i = 0; i < 6; i++) q[leg[i]] = qInit[i];
    const d = this.ikData, b = this.footBody[s], nv = this.model.nv;
    const lam2 = lambda * lambda;
    const cols = leg.map(i => this.dadr[i]);

    for (let it = 0; it < iters; it++) {
      this._fk(d, q);
      const cur = d.xpos.subarray(3 * b, 3 * b + 3);
      const posErr = [targetPos[0] - cur[0], targetPos[1] - cur[1], targetPos[2] - cur[2]];
      if (Math.hypot(...posErr) < posTol) break;
      let qErr = quatMul(targetQuat, quatConj(Array.from(d.xquat.subarray(4 * b, 4 * b + 4))));
      if (qErr[0] < 0) qErr = qErr.map(v => -v);
      const err = [...posErr, ...axisAngle(qErr)];

      this.mj.mj_jacBody(this.model, d, this.jacp, this.jacr, b);
      const jp = this.jacp.GetView(), jr = this.jacr.GetView();
      const J = new Float64Array(36); // 6 x 6 (rows: lin, ang; cols: leg dofs)
      for (let c = 0; c < 6; c++) {
        for (let r = 0; r < 3; r++) {
          J[r * 6 + c] = jp[r * nv + cols[c]];
          J[(r + 3) * 6 + c] = jr[r * nv + cols[c]];
        }
      }
      const JJt = new Float64Array(36);
      for (let i = 0; i < 6; i++)
        for (let j = 0; j < 6; j++) {
          let sum = 0;
          for (let k = 0; k < 6; k++) sum += J[i * 6 + k] * J[j * 6 + k];
          JJt[i * 6 + j] = sum + (i === j ? lam2 : 0);
        }
      const y = solveSpd(JJt, err, 6);
      for (let c = 0; c < 6; c++) {
        let dq = 0;
        for (let r = 0; r < 6; r++) dq += J[r * 6 + c] * y[r];
        q[leg[c]] += dq;
      }
    }
    return leg.map(i => wrapToPi(q[i]));
  }
}
