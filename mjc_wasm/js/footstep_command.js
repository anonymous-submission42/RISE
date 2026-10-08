// Port of the parts of g1_controller used by this demo:
//   include/isaaclab/envs/mdp/commands/vrp_generator.h      (generate_vrp_online)
//   include/isaaclab/envs/mdp/commands/preview_controller.h
//   include/isaaclab/envs/mdp/commands/footstep_command.h   (FootstepCommand)
// restricted to command_source = csv_global, foot_state_source = sim_odom and
// com_generate_type = prev.

import {
  f32, ticks, wrapToPi, add, sub, scale, cubic,
  quatApply, quatApplyInverse, quatMul, quatConj, yawQuat, yawOf,
  quatFromEulerXyz, eulerXyzFromQuat,
  combineFrameTransformsPos, subtractFrameTransforms, subtractFrameTransformsPos,
} from './math.js';
import { LEFT, RIGHT } from './kinematics.js';

// --- VrpGenerator ------------------------------------------------------------

class VrpGenerator {
  constructor(cfg) {
    this.dt = cfg.step_dt;
    this.LA = cfg.future_foot_step_num;
    this.vrpHeight = cfg.vrp_height;
    this.vrpxOffset = cfg.vrpx_offset;
    this.vrpyOffset = cfg.vrpy_offset;
    // Tocabi stepping-stone leftover (0 disables): a swing whose XY travel is
    // under this keeps the swing foot's current height instead of step_z.
    this.cubeDiagonal = cfg.cube_diagonal_length;
    this.NL = ticks(cfg.vrp_horizon_length, this.dt);
    this.vrpRef = Array.from({ length: this.NL }, () => [0, 0, 0]);
    this.comYawRef = new Float64Array(this.NL);
  }

  // footstepCmd: LA x 9 (x, y, z, r, p, yaw, ssp_t, dsp_t, height) in the first-stance frame.
  // swingFootStart: current swing foot position in that frame. The step heights
  // actually used (after the stepping-stone rewrite) are left in this.stepZ.
  generate(footstepCmd, vrpStateInit, comYawInit, swingFootStart, comZ) {
    const { LA, dt, NL } = this;
    const stepX = [], stepY = [], stepZ = [], stepYaw = [], ssp = [], dsp = [], phase = [];
    for (let s = 0; s < LA; s++) {
      const fc = footstepCmd[s];
      stepX.push(fc[0]); stepY.push(fc[1]); stepZ.push(fc[2]); stepYaw.push(fc[5]);
      ssp.push(fc[6]); dsp.push(fc[7]);
      phase.push(Math.sign(fc[1]));
    }

    // target foot points [x, y, z, yaw]
    const stance = [], swing = [], cube = this.cubeDiagonal;
    stance.push([0, 0, this.vrpHeight + comZ[0], 0]);
    if (Math.hypot(stepX[0] - swingFootStart[0], stepY[0] - swingFootStart[1]) < cube)
      stepZ[0] = swingFootStart[2];
    swing.push([stepX[0], stepY[0], stance[0][2] + stepZ[0] + (comZ[1] - comZ[0]), stepYaw[0]]);
    for (let s = 1; s < LA; s++) {
      const st = swing[s - 1].slice();
      const c = Math.cos(st[3]), sn = Math.sin(st[3]);
      stance.push(st);
      const x = st[0] + c * stepX[s] - sn * stepY[s], y = st[1] + sn * stepX[s] + c * stepY[s];
      const d = s === 1 ? Math.hypot(x, y) : Math.hypot(x - stance[s - 2][0], y - stance[s - 2][1]);
      if (d < cube) stepZ[s] = -stepZ[s - 1];
      swing.push([x, y, st[2] + stepZ[s] + (comZ[s + 1] - comZ[s]), st[3] + stepYaw[s]]);
    }
    this.stepZ = stepZ;
    // VRP horizontal offsets
    for (let s = 0; s < LA; s++) {
      const a = stance[s], b = swing[s], p = phase[s];
      a[0] += -p * this.vrpyOffset * Math.sin(a[3]) + this.vrpxOffset * Math.cos(a[3]);
      a[1] += p * this.vrpyOffset * Math.cos(a[3]) + this.vrpxOffset * Math.sin(a[3]);
      b[0] += p * this.vrpyOffset * Math.sin(b[3]) + this.vrpxOffset * Math.cos(b[3]);
      b[1] += -p * this.vrpyOffset * Math.cos(b[3]) + this.vrpxOffset * Math.sin(b[3]);
    }

    const mid3 = (a, b) => [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
    let dsp1Start = 0, dsp2End = 0;
    for (let s = 0; s < LA; s++) {
      const dsp1End = dsp1Start + ticks(dsp[s], dt);
      const sspEnd = dsp1End + ticks(ssp[s], dt);
      dsp2End = sspEnd + ticks(dsp[s], dt);
      const stanceP = stance[s].slice(0, 3);
      const prevMid = s === 0 ? vrpStateInit : mid3(stance[s - 1], swing[s - 1]);
      const prevYaw = s === 0 ? comYawInit : 0.5 * (stance[s - 1][3] + swing[s - 1][3]);
      const curYaw = 0.5 * (stance[s][3] + swing[s][3]);
      const mid = mid3(stance[s], swing[s]);

      for (let tk = dsp1Start; tk < dsp1End && tk < NL; tk++) {
        const ct = (tk - dsp1Start) * dt;
        this.vrpRef[tk] = [0, 1, 2].map(i => cubic(prevMid[i], 0, stanceP[i], 0, dsp[s], ct)[0]);
        this.comYawRef[tk] = prevYaw;
      }
      for (let tk = dsp1End; tk < sspEnd && tk < NL; tk++) {
        this.vrpRef[tk] = stanceP;
        this.comYawRef[tk] = cubic(prevYaw, 0, curYaw, 0, ssp[s], (tk - dsp1End) * dt)[0];
      }
      for (let tk = sspEnd; tk < dsp2End && tk < NL; tk++) {
        const ct = (tk - sspEnd) * dt;
        this.vrpRef[tk] = [0, 1, 2].map(i => cubic(stanceP[i], 0, mid[i], 0, dsp[s], ct)[0]);
        this.comYawRef[tk] = curYaw;
      }
      dsp1Start = dsp2End;
    }
    const lastMid = mid3(stance[LA - 1], swing[LA - 1]);
    const lastYaw = 0.5 * (stance[LA - 1][3] + swing[LA - 1][3]);
    for (let tk = dsp2End; tk < NL; tk++) {
      this.vrpRef[tk] = lastMid;
      this.comYawRef[tk] = lastYaw;
    }
  }
}

// --- PreviewController ---------------------------------------------------------

const matMul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const transpose = A => A[0].map((_, j) => A.map(r => r[j]));

class PreviewController {
  constructor(horizon, dt, zc) {
    const hz = Math.trunc(f32(1 / f32(dt)));
    this.NL = Math.trunc(f32(f32(horizon) * hz));
    // A, B, C are float32 matrices in the C++ controller.
    dt = f32(dt);
    const dt2 = f32(dt * dt);
    this.A = [[1, dt, f32(dt2 / 2)], [0, 1, dt], [0, 0, 1]];
    this.B = [f32(f32(dt2 * dt) / 6), f32(dt2 / 2), dt];
    this.C = [1, 0, f32(-f32(zc) / f32(9.81))];
    const { A, B, C } = this;

    // augmented system
    const CA = [0, 1, 2].map(j => C[0] * A[0][j] + C[1] * A[1][j] + C[2] * A[2][j]);
    const CB = C[0] * B[0] + C[1] * B[1] + C[2] * B[2];
    const Ab = [[1, ...CA], [0, ...A[0]], [0, ...A[1]], [0, ...A[2]]];
    const Bb = [[CB], [B[0]], [B[1]], [B[2]]];
    const Q = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const R = 1e-6;

    // DARE by iterating the Riccati recursion
    const AbT = transpose(Ab), BbT = transpose(Bb);
    let P = Q.map(r => r.slice());
    for (let i = 0; i < 100000; i++) {
      const PB = matMul(P, Bb), denom = R + matMul(BbT, PB)[0][0];
      const AtPB = matMul(AbT, PB), BtPA = matMul(BbT, matMul(P, Ab));
      const AtPA = matMul(AbT, matMul(P, Ab));
      let diff = 0;
      const Pn = AtPA.map((row, r) => row.map((v, c) => {
        const val = v - AtPB[r][0] * BtPA[0][c] / denom + Q[r][c];
        diff = Math.max(diff, Math.abs(val - P[r][c]));
        return val;
      }));
      P = Pn;
      if (diff < 1e-12) break;
    }
    const K = P;
    const inv = 1 / (R + matMul(BbT, matMul(K, Bb))[0][0]);
    const G = matMul(BbT, matMul(K, Ab))[0].map(v => v * inv);
    this.Gi = G[0];
    this.Gx = G.slice(1);

    const BKA = matMul(Bb, matMul(BbT, matMul(K, Ab)));
    const AcT = transpose(Ab.map((row, r) => row.map((v, c) => v - inv * BKA[r][c])));
    const RBT = BbT[0].map(v => v * inv);
    const KI = K.map(r => [r[0]]); // K * I_bar
    let X = matMul(AcT, KI).map(r => [-r[0]]);
    this.Gd = new Float64Array(this.NL);
    this.Gd[0] = -this.Gi;
    for (let l = 1; l < this.NL; l++) {
      this.Gd[l] = RBT.reduce((s, v, k) => s + v * X[k][0], 0);
      X = matMul(AcT, X);
    }
    this.state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // rows: pos, vel, acc; cols: x, y, z
    this.errInt = [0, 0, 0];
  }

  setState(s) { this.state = s.map(r => r.slice()); }
  resetErrorIntegral() { this.errInt = [0, 0, 0]; }

  computeTargetState(vrpRef) {
    const { A, B, C, state } = this;
    const input = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const cart = C[0] * state[0][k] + C[1] * state[1][k] + C[2] * state[2][k];
      this.errInt[k] += cart - vrpRef[0][k];
      const gx = this.Gx[0] * state[0][k] + this.Gx[1] * state[1][k] + this.Gx[2] * state[2][k];
      let preview = 0;
      for (let l = 1; l < this.NL; l++) preview += this.Gd[l] * vrpRef[l][k];
      input[k] = -this.Gi * this.errInt[k] - gx - preview;
    }
    return [0, 1, 2].map(r => [0, 1, 2].map(k =>
      A[r][0] * state[0][k] + A[r][1] * state[1][k] + A[r][2] * state[2][k] + B[r] * input[k]));
  }
}

// --- FootstepCommand -----------------------------------------------------------

const IDENTITY = [1, 0, 0, 0];

export class FootstepCommand {
  // plan: [{x, y, z, yaw, ssp_t, dsp_t, height, com_z, phase}] world-frame swing
  // landings; initStance: world pose {x, y, z, yaw} of the initial stance foot.
  constructor(cfg, kin, plan, initStance) {
    this.cfg = cfg;
    this.kin = kin;
    this.vrp = new VrpGenerator(cfg);
    this.preview = new PreviewController(cfg.preview_horizon_length, cfg.step_dt, cfg.vrp_height);
    const LA = cfg.future_foot_step_num;
    this.footCommand = Array.from({ length: LA }, () => new Array(9).fill(0));
    this.comZCommand = new Array(LA + 1).fill(0);
    this.phase = new Array(LA).fill(0);
    this.plan = plan;
    this.initStance = { ...initStance };
    this.plannerIndex = 0;
    this.stanceWorld = { ...initStance };
    this.walkingTick = 0;
    this.timeLeft = 0;
    this.stepCompleted = false;
    this.stepCounter = 0;
    this.lastStepError = [0, 0, 0];
    this.targetJointPos = new Array(12).fill(0);
    this.commandVec = new Array(24).fill(0);
    // set by the caller every control tick
    this.robotQuat = IDENTITY;
    this.basePosWorld = [0, 0, 0];
    this.targetComGlobal = { pos: [0, 0, 0], vel: [0, 0, 0], acc: [0, 0, 0] };
  }

  get globalPlanDone() { return this.plannerIndex >= this.plan.length; }
  stanceSide() { return this.phase[0] === 0 ? LEFT : RIGHT; }
  swingSide() { return this.phase[0] === 0 ? RIGHT : LEFT; }

  reset() {
    this.stepCounter = 0;
    // resample_command_global_
    this.walkingTick = 0;
    this.plannerIndex = 0;
    this.stanceWorld = { ...this.initStance };
    this.fillGlobalBuffer();
    this.updateLinkStates();
    this.generateVrpRefTrajectory();
    this.preview.resetErrorIntegral();
    this.preview.setState([this.comPosStance.slice(), [0, 0, 0], [0, 0, 0]]);
    this.computeCommandVec();
  }

  compute() {
    const dt = this.cfg.step_dt;
    this.updateLinkStates();
    this.generateRefTrajectory();
    this.timeLeft = f32(this.timeLeft - f32(dt));
    this.walkingTick += 1;
    this.stepCompleted = this.timeLeft <= 0;
    if (this.stepCompleted) {
      const fc = this.footCommand[0], sw = this.swingFootStancePos;
      this.lastStepCommand = [fc[0], fc[1], fc[2]];
      this.lastStepError = [fc[0] - sw[0], fc[1] - sw[1], fc[2] - sw[2]];
      this.stepCounter += 1;
      // update_command_global_
      this.walkingTick = 0;
      this.stanceWorld = this.worldPoseFromFoot(this.swingSide()); // sim_odom: foot that just landed
      this.plannerIndex += 1;
      this.fillGlobalBuffer();
      this.updateLinkStates();
      this.generateVrpRefTrajectory();
      this.generateRefTrajectory();
    }
    this.computeCommandVec();
  }

  // Leg joints for standing still on the current footholds: the IK target of
  // hold_standby() in footstep_command.h (pelvis over the midpoint of the feet,
  // at the walking height, heading between the two feet).
  standingJointPos(comZ) {
    const sw = this.swingFootStancePos;
    const pelvPos = [0.5 * sw[0], 0.5 * sw[1], this.cfg.vrp_height + comZ + this.cfg.pelv_com_offset];
    const midYaw = 0.5 * yawOf(this.swingFootStanceQuat);
    this.solveLegIk(pelvPos, quatFromEulerXyz(0, 0, midYaw), sw, this.swingFootStanceQuat);
    return this.targetJointPos.slice();
  }

  // --- sim_odom world foot pose: odom base position + IMU-rotated FK offset ---
  worldPoseFromFoot(s) {
    const pos = add(this.basePosWorld, quatApply(this.robotQuat, this.kin.footPos(s)));
    const yaw = yawOf(quatMul(this.robotQuat, this.kin.footQuat(s)));
    return { x: pos[0], y: pos[1], z: pos[2], yaw };
  }

  // --- global plan buffer ---
  phaseFor(idx) {
    const p0 = this.plan[0].phase;
    return (idx & 1) ? 1 - p0 : p0;
  }

  static buildGlobalStep(from, tgt) {
    const c = Math.cos(-from.yaw), s = Math.sin(-from.yaw);
    const dx = tgt.x - from.x, dy = tgt.y - from.y;
    return [c * dx - s * dy, s * dx + c * dy, tgt.z - from.z, 0, 0,
            wrapToPi(tgt.yaw - from.yaw), tgt.ssp_t, tgt.dsp_t, tgt.height];
  }

  planPose(idx) {
    const t = this.plan[Math.min(idx, this.plan.length - 1)];
    return { x: t.x, y: t.y, z: t.z, yaw: t.yaw };
  }

  heldPose(idx) {
    const N = this.plan.length;
    const sameFootAsLast = ((idx - (N - 1)) % 2) === 0;
    if (!sameFootAsLast && N < 2) return { ...this.initStance };
    return this.planPose(sameFootAsLast ? N - 1 : N - 2);
  }

  fillGlobalBuffer() {
    const LA = this.cfg.future_foot_step_num, N = this.plan.length;
    for (let s = 0; s < LA; s++) {
      const idx = this.plannerIndex + s;
      this.phase[s] = this.phaseFor(idx);
      const from = s === 0 ? this.stanceWorld : (idx - 1 < N ? this.planPose(idx - 1) : this.heldPose(idx - 1));
      if (idx < N) {
        this.footCommand[s] = FootstepCommand.buildGlobalStep(from, this.plan[idx]);
        this.comZCommand[s] = this.plan[idx].com_z;
      } else {
        // past the plan end: station-keep on the final plan poses
        const tgt = { ...this.plan[N - 1], ...this.heldPose(idx) };
        this.footCommand[s] = FootstepCommand.buildGlobalStep(from, tgt);
        this.comZCommand[s] = this.plan[N - 1].com_z;
      }
    }
    const last = this.plannerIndex + LA - 1;
    this.comZCommand[LA] = (last < N ? this.plan[last] : this.plan[N - 1]).com_z;
    const fc = this.footCommand[0];
    this.timeLeft = f32(f32(fc[6]) + f32(f32(fc[7]) * 2));
  }

  // --- measured state, re-anchored at the pelvis every tick ---
  updateLinkStates() {
    const R = this.robotQuat, kin = this.kin;
    this.comPosGlobal = quatApply(R, kin.comPos());
    const fill = s => ({
      pos: quatApply(R, kin.footPos(s)),
      quat: quatMul(R, kin.footQuat(s)),
      lin: quatApply(R, kin.footLinVel(s)),
    });
    this.stanceFoot = fill(this.stanceSide());
    this.swingFoot = fill(this.swingSide());
    const stanceYaw = yawQuat(this.stanceFoot.quat);
    [this.swingFootStancePos, this.swingFootStanceQuat] = subtractFrameTransforms(
      this.stanceFoot.pos, stanceYaw, this.swingFoot.pos, this.swingFoot.quat);
    this.comPosStance = subtractFrameTransformsPos(this.stanceFoot.pos, stanceYaw, this.comPosGlobal);
  }

  generateVrpRefTrajectory() {
    const sw = this.swingFootStancePos;
    const vrpState = [sw[0] / 2, sw[1] / 2, this.comPosStance[2]];
    const swingYaw = yawOf(this.swingFootStanceQuat);
    this.vrp.generate(this.footCommand, vrpState, swingYaw / 2, sw, this.comZCommand);
    this.footCommand.forEach((fc, s) => { fc[2] = this.vrp.stepZ[s]; });

    const fc = this.footCommand[0];
    this.swingStartPos = sw.slice();
    this.swingStartQuat = this.swingFootStanceQuat.slice();
    this.swingEndPos = [fc[0], fc[1], fc[2]];
    this.swingEndQuat = quatFromEulerXyz(fc[3], fc[4], fc[5]);

    // preview state in the (updated) stance foot frame
    const stanceYaw = yawQuat(this.stanceFoot.quat), t = this.targetComGlobal;
    this.preview.setState([
      quatApplyInverse(stanceYaw, sub(t.pos, this.stanceFoot.pos)),
      quatApplyInverse(stanceYaw, sub(t.vel, this.stanceFoot.lin)),
      quatApplyInverse(stanceYaw, t.acc),
    ]);
  }

  generateRefTrajectory() {
    const NL = this.preview.NL, vNL = this.vrp.NL, tick = this.walkingTick;
    const ref = [];
    for (let l = 0; l < NL; l++) ref.push(this.vrp.vrpRef[Math.min(tick + l, vNL - 1)]);
    const next = this.preview.computeTargetState(ref);
    this.preview.setState(next);
    const [comPos, comVel, comAcc] = next;

    const stanceYaw = yawQuat(this.stanceFoot.quat);
    this.targetComGlobal = {
      pos: combineFrameTransformsPos(this.stanceFoot.pos, stanceYaw, comPos),
      vel: quatApply(stanceYaw, comVel),
      acc: quatApply(stanceYaw, comAcc),
    };
    const comYaw = this.vrp.comYawRef[Math.min(tick + 1, vNL - 1)];
    const pelvPos = [comPos[0], comPos[1], comPos[2] + this.cfg.pelv_com_offset];
    this.generateFeetRefTrajectory();
    this.solveLegIk(pelvPos, quatFromEulerXyz(0, 0, comYaw), this.swingTargetPos, this.swingTargetQuat);
  }

  // Both legs for a pelvis pose and swing-foot pose given in the stance-foot frame.
  solveLegIk(pelvPos, pelvQuat, swingPos, swingQuat) {
    const stanceIsLeft = this.phase[0] === 0;
    const lPos = stanceIsLeft ? [0, 0, 0] : swingPos, lQuat = stanceIsLeft ? IDENTITY : swingQuat;
    const rPos = stanceIsLeft ? swingPos : [0, 0, 0], rQuat = stanceIsLeft ? swingQuat : IDENTITY;
    const inv = quatConj(pelvQuat), c = this.cfg, kin = this.kin;
    const ql = kin.diffIkLeg(LEFT, quatApply(inv, sub(lPos, pelvPos)), quatMul(inv, lQuat),
                             kin.legQ(LEFT), c.ik_iters, c.ik_lambda, c.ik_pos_tol);
    const qr = kin.diffIkLeg(RIGHT, quatApply(inv, sub(rPos, pelvPos)), quatMul(inv, rQuat),
                             kin.legQ(RIGHT), c.ik_iters, c.ik_lambda, c.ik_pos_tol);
    this.targetJointPos = [...ql, ...qr];
  }

  generateFeetRefTrajectory() {
    const dt = this.cfg.step_dt, fc = this.footCommand[0];
    const ssp = fc[6], dsp = fc[7], height = fc[8], tick = this.walkingTick;
    const up = this.cfg.swing_up_timing, down = this.cfg.swing_down_timing;
    const dsp1End = ticks(dsp, dt);
    const sspEnd = ticks(f32(f32(ssp) + f32(dsp)), dt);
    const p0 = this.swingStartPos, p1 = this.swingEndPos;
    let pos, quat;
    if (tick < dsp1End) {
      pos = p0; quat = this.swingStartQuat;
    } else if (tick < sspEnd) {
      const ct = (tick - dsp1End) * dt;
      pos = [cubic(p0[0], 0, p1[0], 0, ssp, ct)[0], cubic(p0[1], 0, p1[1], 0, ssp, ct)[0], 0];
      const e0 = eulerXyzFromQuat(this.swingStartQuat), e1 = eulerXyzFromQuat(this.swingEndQuat);
      const et = [0, 1, 2].map(i => cubic(wrapToPi(e0[i]), 0, wrapToPi(e1[i]), 0, ssp, ct)[0]);
      quat = quatFromEulerXyz(et[0], et[1], et[2]);
      const lift = Math.max(0, Math.max(p0[2], p1[2])) + height;
      const upEnd = ticks(f32(f32(dsp) + f32(f32(ssp) * f32(up))), dt);
      const downStart = ticks(f32(f32(dsp) + f32(f32(ssp) * f32(down))), dt);
      if (tick < upEnd) pos[2] = cubic(p0[2], 0, lift, 0, ssp * up, (tick - dsp1End) * dt)[0];
      else if (tick < downStart) pos[2] = lift;
      else pos[2] = cubic(lift, 0, p1[2], 0, ssp * (1 - down), (tick - downStart) * dt)[0];
    } else {
      pos = p1; quat = this.swingEndQuat;
    }
    this.swingTargetPos = pos;
    this.swingTargetQuat = quat;
  }

  // 24-dim command: [ik_target(12), phase_cos, phase_sin, foot_command0(9), com_z(1)]
  computeCommandVec() {
    const fc = this.footCommand[0], v = this.commandVec;
    const total = fc[6] + fc[7] * 2;
    const ph = total > 1e-6 ? this.walkingTick * this.cfg.step_dt / total : 0;
    for (let i = 0; i < 12; i++) v[i] = this.targetJointPos[i];
    v[12] = Math.cos(2 * Math.PI * ph);
    v[13] = Math.sin(2 * Math.PI * ph);
    for (let i = 0; i < 9; i++) v[14 + i] = fc[i];
    v[23] = this.comZCommand[0];
  }
}
