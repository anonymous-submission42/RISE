// The controller FSM path this demo runs (g1_controller / t1_controller):
//   FixStand (hold the LowCoM pose) -> Footstep (csv_global plan) -> joint hold
//   once the plan is done, or Passive on a fall
// on top of the unitree_mujoco bridge's joint PD: tau = kp (q* - q) + kd (0 - dq).
// Robot-specific values come from robots.js; arrays are in SDK motor order.

import { Kinematics } from './kinematics.js';
import { FootstepCommand } from './footstep_command.js';
import { FootstepPolicy } from './policy.js';

const HOLD_BLEND = 1.0; // [s] switch -> standing pose on the final footholds
const INIT_BLEND = 2.0; // [s] standing pose -> init (FixStand) pose
const PASSIVE_KD = 3;
const FIXSTAND_TIME = 1.0; // [s] stand before walking (the operator presses g on hardware)

export class Controller {
  constructor(mujoco, model, data, { robot, inertia, mlp, plan }) {
    this.mj = mujoco;
    this.model = model;
    this.data = data;
    this.robot = robot;
    const ids = robot.sdkJoints.map(n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n));
    this.n = ids.length;
    this.qadr = ids.map(j => model.jnt_qposadr[j]);
    this.dadr = ids.map(j => model.jnt_dofadr[j]);
    // actuator driving each SDK motor
    this.act = ids.map(j => {
      for (let a = 0; a < model.nu; a++) if (model.actuator_trnid[2 * a] === j) return a;
      throw new Error(`no actuator for joint ${j}`);
    });
    this.kin = new Kinematics(mujoco, model, inertia, robot);
    this.mlp = mlp;
    this.plan = plan;
    this.decimation = Math.round(robot.footstep.step_dt / model.opt.timestep);
    this.reset();
  }

  dispose() { this.kin.dispose(); }

  reset() {
    const { mj, model, data, robot } = this;
    mj.mj_resetDataKeyframe(model, data, mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY.value, 'init'));
    mj.mj_forward(model, data);
    this.physicsStep = 0;
    this.mode = 'fixstand';
    this.kp = robot.fixstand.kp.slice();
    this.kd = robot.fixstand.kd.slice();
    this.qTarget = robot.fixstand.q.slice();
    this.command = null;
    this.policy = null;
    this.stepErrors = [];
  }

  // Robot state as the controller sees it through lowstate / odom.
  readState() {
    const d = this.data, qpos = d.qpos, qvel = d.qvel;
    const q = this.qadr.map(a => qpos[a]), qd = this.dadr.map(a => qvel[a]);
    return {
      q, qd,
      quat: [qpos[3], qpos[4], qpos[5], qpos[6]], // imu site sits on the root body origin
      gyro: [qvel[3], qvel[4], qvel[5]],
      basePos: [qpos[0], qpos[1], qpos[2]],
    };
  }

  // Advance one physics step; the 50 Hz controller runs every `decimation` steps.
  step() {
    if (this.physicsStep % this.decimation === 0) this.controlTick();
    if (this.mode === 'hold') this.qTarget = this.holdTarget(this.data.time - this.hold.t0);
    const d = this.data, qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl;
    for (let i = 0; i < this.n; i++) {
      ctrl[this.act[i]] = this.kp[i] * (this.qTarget[i] - qpos[this.qadr[i]]) - this.kd[i] * qvel[this.dadr[i]];
    }
    this.mj.mj_step(this.model, d);
    this.physicsStep += 1;
  }

  controlTick() {
    const s = this.readState();
    // bad_orientation(1.0) -> Passive
    if (this.mode === 'footstep') {
      const [w, x, y] = s.quat;
      const gz = -(1 - 2 * (x * x + y * y));
      if (Math.abs(Math.acos(-gz)) > 1.0) this.enterPassive(s);
    }
    if (this.mode === 'fixstand' && this.data.time >= FIXSTAND_TIME) this.enterFootstep(s);
    if (this.mode === 'footstep') this.footstepTick(s);
  }

  enterPassive(s) {
    this.mode = 'passive';
    this.kp = new Array(this.n).fill(0);
    this.kd = new Array(this.n).fill(PASSIVE_KD);
    this.qTarget = s.q.slice();
  }

  // State_Footstep::enter + the start of its policy thread.
  enterFootstep(s) {
    this.mode = 'footstep';
    const { legIds, upperIds, footstep: c } = this.robot;
    legIds.forEach((m, i) => { this.kp[m] = c.stiffness[i]; this.kd[m] = c.damping[i]; });
    upperIds.forEach((m, j) => {
      this.kp[m] = c.upper_stiffness[j]; this.kd[m] = c.upper_damping[j]; this.qTarget[m] = c.upper_default[j];
    });
    const initStance = this.plan[0].phase === 0 ? c.init_lfoot : c.init_rfoot;
    this.command = new FootstepCommand(c, this.kin, this.plan, initStance);
    this.policy = new FootstepPolicy(this.mlp, c);
    // csv_global starts walking right away
    this.syncCommand(s);
    this.command.reset();
    this.policy.reset(this.policyState(s), this.command.commandVec);
  }

  syncCommand(s) {
    this.kin.setState(s.q, s.qd);
    this.command.robotQuat = s.quat;
    this.command.basePosWorld = s.basePos;
  }

  policyState(s) {
    const legIds = this.robot.legIds;
    return { gyro: s.gyro, quat: s.quat, q: legIds.map(i => s.q[i]), qd: legIds.map(i => s.qd[i]) };
  }

  // One iteration of State_Footstep's policy loop.
  footstepTick(s) {
    const cmd = this.command;
    this.syncCommand(s);
    cmd.compute();
    if (cmd.stepCompleted) this.stepErrors.push(cmd.lastStepError);
    if (cmd.globalPlanDone) {
      // The policy was not trained to stand still, so instead of the deploy
      // code's standby command, hand over to joint PD once the last foot lands.
      this.enterJointHold(s);
      return;
    }
    const action = this.policy.step(this.policyState(s), cmd.commandVec);
    this.robot.legIds.forEach((m, i) => { this.qTarget[m] = action[i]; });
  }

  // Joint PD (hold gains) that brings the robot to rest. The target starts where
  // the hold gains reproduce the policy's last PD torque (so the switch does not
  // jolt the robot), blends to a standing pose on the final footholds (IK,
  // pelvis centred over the feet) and then to the init (FixStand) pose. Freezing
  // the measured joint angles instead keeps the mid-gait lean and the robot
  // usually tips over.
  enterJointHold(s) {
    const { hold, legIds, upperIds, footstep: c, fixstand } = this.robot;
    const from = s.q.map((q, i) =>
      q + (this.kp[i] * (this.qTarget[i] - q) - this.kd[i] * s.qd[i] + hold.kd[i] * s.qd[i]) / hold.kp[i]);
    const stance = s.q.slice();
    const legs = this.command.standingJointPos(this.command.comZCommand[0]);
    legIds.forEach((m, i) => { stance[m] = legs[i]; });
    upperIds.forEach((m, j) => { stance[m] = c.upper_default[j]; });
    this.mode = 'hold';
    this.kp = hold.kp.slice();
    this.kd = hold.kd.slice();
    // [time since the switch, target]: piecewise-linear blend between them
    this.hold = { t0: this.data.time, keys: [[0, from], [HOLD_BLEND, stance], [HOLD_BLEND + INIT_BLEND, fixstand.q]] };
  }

  holdTarget(t) {
    const keys = this.hold.keys;
    let k = 1;
    while (k < keys.length - 1 && t > keys[k][0]) k++;
    const [t0, a] = keys[k - 1], [t1, b] = keys[k];
    const w = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    return a.map((q, i) => q + w * (b[i] - q));
  }
}
