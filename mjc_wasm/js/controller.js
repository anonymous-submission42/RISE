// The g1_controller FSM path this demo runs:
//   FixStand (hold the LowCoM pose) -> Footstep (csv_global plan) -> joint hold
//   once the plan is done, or Passive on a fall
// on top of the unitree_mujoco bridge's joint PD: tau = kp (q* - q) + kd (0 - dq).

import { Kinematics, SDK_JOINTS } from './kinematics.js';
import { FootstepCommand } from './footstep_command.js';
import { FootstepPolicy } from './policy.js';
import { INIT_LFOOT, INIT_RFOOT } from './plan.js';

// config/policy/footstep/deploy_base.yaml (+ obs_full/params/deploy.yaml)
export const FOOTSTEP_CFG = {
  step_dt: 0.02,
  future_foot_step_num: 2,
  vrp_height: 0.6258,
  pelv_com_offset: 0.0678,
  vrpx_offset: 0.03,
  vrpy_offset: 0.02,
  vrp_horizon_length: 4.0,
  preview_horizon_length: 1.6,
  swing_up_timing: 0.2,
  swing_down_timing: 0.6,
  ik_iters: 10,
  ik_lambda: 0.05,
  ik_pos_tol: 0.0001,
  default_joint_pos: [-0.4, 0, 0, 0.9, -0.5, 0, -0.4, 0, 0, 0.9, -0.5, 0],
  action_scale: new Array(12).fill(0.5),
  action_offset: [-0.4, 0, 0, 0.9, -0.5, 0, -0.4, 0, 0, 0.9, -0.5, 0],
  stiffness: [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
  damping: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
  upper_default: [0, 0, 0.2, 0.1, 0.2, 0, 1.1, 0, 0, 0, 0.1, -0.2, 0, 1.1, 0, 0, 0],
  upper_stiffness: [200, 200, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
  upper_damping: [5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
};

// config/config.yaml FSM: FixStand (LowCoM stand) and Passive
const FIXSTAND = {
  kp: [200, 100, 100, 350, 540, 40, 200, 100, 100, 350, 540, 40, 200, 200, 200,
       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
  kd: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
  q: [-0.4, 0, 0, 0.9, -0.5, 0, -0.4, 0, 0, 0.9, -0.5, 0, 0, 0, 0.2,
      0.1, 0.2, 0, 1.1, 0, 0, 0, 0.1, -0.2, 0, 1.1, 0, 0, 0],
};
// Joint hold after the plan: FixStand gains with stiffer hip / ankle roll. With
// FixStand's roll gains the stance is close to neutrally stable sideways, and the
// lateral momentum left from walking tips it over.
const HOLD = {
  kp: FIXSTAND.kp.map((k, i) => (i === 1 || i === 7) ? 300 : (i === 5 || i === 11) ? 150 : k),
  kd: FIXSTAND.kd.map((k, i) => [1, 5, 7, 11].includes(i) ? 5 : k),
};
const HOLD_BLEND = 1.0; // [s]
const PASSIVE_KD = 3;
const FIXSTAND_TIME = 1.0; // [s] stand before walking (the operator presses g on hardware)

export class Controller {
  constructor(mujoco, model, data, { inertia, mlp, plan }) {
    this.mj = mujoco;
    this.model = model;
    this.data = data;
    const ids = SDK_JOINTS.map(n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n));
    this.qadr = ids.map(j => model.jnt_qposadr[j]);
    this.dadr = ids.map(j => model.jnt_dofadr[j]);
    // actuator driving each SDK motor
    this.act = ids.map(j => {
      for (let a = 0; a < model.nu; a++) if (model.actuator_trnid[2 * a] === j) return a;
      throw new Error(`no actuator for joint ${j}`);
    });
    this.kin = new Kinematics(mujoco, model, inertia);
    this.mlp = mlp;
    this.plan = plan;
    this.decimation = Math.round(FOOTSTEP_CFG.step_dt / model.opt.timestep);
    this.reset();
  }

  dispose() { this.kin.dispose(); }

  reset() {
    const { mj, model, data } = this;
    mj.mj_resetDataKeyframe(model, data, mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY.value, 'init'));
    mj.mj_forward(model, data);
    this.physicsStep = 0;
    this.mode = 'fixstand';
    this.kp = FIXSTAND.kp.slice();
    this.kd = FIXSTAND.kd.slice();
    this.qTarget = FIXSTAND.q.slice();
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
      quat: [qpos[3], qpos[4], qpos[5], qpos[6]], // imu site sits on the pelvis origin
      gyro: [qvel[3], qvel[4], qvel[5]],
      basePos: [qpos[0], qpos[1], qpos[2]],
    };
  }

  // Advance one physics step; the 50 Hz controller runs every `decimation` steps.
  step() {
    if (this.physicsStep % this.decimation === 0) this.controlTick();
    if (this.mode === 'hold') {
      const { from, to, t0 } = this.hold, a = Math.min(1, (this.data.time - t0) / HOLD_BLEND);
      this.qTarget = from.map((q, i) => q + a * (to[i] - q));
    }
    const d = this.data, qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl;
    for (let i = 0; i < 29; i++) {
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
    this.kp = new Array(29).fill(0);
    this.kd = new Array(29).fill(PASSIVE_KD);
    this.qTarget = s.q.slice();
  }

  // State_Footstep::enter + the start of its policy thread.
  enterFootstep(s) {
    this.mode = 'footstep';
    const c = FOOTSTEP_CFG;
    this.kp = [...c.stiffness, ...c.upper_stiffness];
    this.kd = [...c.damping, ...c.upper_damping];
    this.qTarget = [...this.qTarget.slice(0, 12), ...c.upper_default];
    const initStance = this.plan[0].phase === 0 ? INIT_LFOOT : INIT_RFOOT;
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
    return { gyro: s.gyro, quat: s.quat, q: s.q.slice(0, 12), qd: s.qd.slice(0, 12) };
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
    for (let i = 0; i < 12; i++) this.qTarget[i] = action[i];
  }

  // Joint PD (HOLD gains) toward a standing pose on the final footholds (IK,
  // pelvis centred over the feet). Freezing the measured joint angles instead
  // keeps the mid-gait lean and the robot usually tips over. The target starts
  // where the HOLD gains reproduce the policy's last PD torque and blends to
  // the standing pose over HOLD_BLEND, so the switch does not jolt the robot.
  enterJointHold(s) {
    const from = s.q.map((q, i) =>
      q + (this.kp[i] * (this.qTarget[i] - q) - this.kd[i] * s.qd[i] + HOLD.kd[i] * s.qd[i]) / HOLD.kp[i]);
    const legs = this.command.standingJointPos(this.command.comZCommand[0]);
    this.mode = 'hold';
    this.kp = HOLD.kp.slice();
    this.kd = HOLD.kd.slice();
    this.hold = { from, to: [...legs, ...FOOTSTEP_CFG.upper_default], t0: this.data.time };
  }
}
