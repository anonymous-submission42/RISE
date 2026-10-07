// Footstep policy runtime: the exported actor MLP plus the observation history
// and action processing of g1_controller's ManagerBasedRLEnv for
// config/policy/footstep/obs_full (deploy.yaml).

// Actor exported from the ONNX file (Gemm + ELU layers) as raw float32:
// for each layer, weight [out x in] row-major followed by bias [out].
export class Mlp {
  constructor(dims, buffer) {
    const f = new Float32Array(buffer);
    this.layers = [];
    let o = 0;
    for (let i = 0; i + 1 < dims.length; i++) {
      const nin = dims[i], nout = dims[i + 1];
      const W = f.subarray(o, o + nin * nout); o += nin * nout;
      const b = f.subarray(o, o + nout); o += nout;
      this.layers.push({ nin, nout, W, b });
    }
    if (o !== f.length) throw new Error(`policy weights: expected ${o} floats, got ${f.length}`);
  }

  run(x) {
    const last = this.layers.length - 1;
    this.layers.forEach(({ nin, nout, W, b }, li) => {
      const y = new Float32Array(nout);
      for (let r = 0; r < nout; r++) {
        let s = b[r];
        const row = r * nin;
        for (let c = 0; c < nin; c++) s += W[row + c] * x[c];
        y[r] = li === last ? s : (s > 0 ? s : Math.expm1(s)); // ELU(alpha=1)
      }
      x = y;
    });
    return x;
  }
}

// Observation terms (deploy.yaml order), each with history_length 10 and
// skip_history_tick 2, stacked history-major (use_gym_history: true):
//   base_ang_vel(3) projected_gravity(3) joint_pos_ordered_rel(12)
//   joint_vel_ordered(12) joint_ik_target(12) phase(2)
//   foot_commands_3d_w_comz(10) last_processed_action(12)        = 66 per frame
const HISTORY = 10, SKIP = 2, FRAME = 66;

export class FootstepPolicy {
  constructor(mlp, cfg) {
    this.mlp = mlp;
    this.defaultPos = cfg.default_joint_pos;
    this.scale = cfg.action_scale;
    this.offset = cfg.action_offset;
    this.processed = new Array(12).fill(0); // JointAction starts at zeros
    this.buffer = [];
  }

  // s: {gyro, quat, q, qd} (q/qd: 12 leg joints); command: 24-dim FootstepCommand vector.
  frame(s, command) {
    const [w, x, y, z] = s.quat;
    // projected gravity = R^T (0, 0, -1)
    const g = [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))];
    return [
      ...s.gyro, ...g,
      ...s.q.map((v, i) => v - this.defaultPos[i]),
      ...s.qd,
      ...command.slice(0, 12), command[12], command[13],
      ...command.slice(14, 24),
      ...this.processed,
    ];
  }

  // ObservationManager::reset: fill the whole history with the current frame.
  reset(s, command) {
    const f = this.frame(s, command);
    this.buffer = Array.from({ length: HISTORY * SKIP }, () => f);
  }

  // ManagerBasedRLEnv::step: push a frame, run the actor, process the action.
  step(s, command) {
    this.buffer.push(this.frame(s, command));
    if (this.buffer.length > HISTORY * SKIP) this.buffer.shift();
    const obs = new Float32Array(HISTORY * FRAME);
    const n = this.buffer.length;
    for (let h = 0; h < HISTORY; h++) {
      const idx = Math.max(0, Math.min(n - 1, n - 1 - (HISTORY - 1 - h) * SKIP));
      obs.set(this.buffer[idx], h * FRAME);
    }
    this.obs = obs;
    this.action = this.mlp.run(obs);
    this.processed = Array.from(this.action, (a, i) => a * this.scale[i] + this.offset[i]);
    return this.processed;
  }
}
