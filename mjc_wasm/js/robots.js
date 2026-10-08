// Per-robot configuration. Everything here comes from the robot's controller
// (deploy_base.yaml + obs_full/params/deploy.yaml, config.yaml FSM, cmd/ scripts)
// and its unitree_mujoco model; the algorithm code is shared.
//
// Joint arrays are in the controller's SDK motor order (`sdkJoints`); leg arrays
// in joint_ids_map order (left hip pitch .. ankle roll, then right).

const G1_JOINTS = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
  'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
  'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
  'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

const T1_JOINTS = [
  'AAHead_yaw', 'Head_pitch',
  'Left_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw',
  'Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw',
  'Waist',
  'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
  'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
];

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const legs = side => [...side, ...side];

// gen_rocky_mountain.py parameters shared by both robots.
const ROCKY = {
  spacing: 0.2, width: 0.2, startClear: 0.3, startClearFront: 0.3, noise: 0.03,
  maxCellYaw: 1.79, corridor: [0.45, 10.5], terrainMargin: 1.0, idwPower: 2,
  minHeight: 0.015, seed: 7, stone: [0.1, 0.1], planeMargin: 3.0, platform: [0.1, 0.17],
};

const G1_FIXSTAND_KP = [200, 100, 100, 350, 540, 40, 200, 100, 100, 350, 540, 40, 200, 200, 200,
                        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40];
const G1_FIXSTAND_KD = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
                        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];

export const ROBOTS = {
  g1: {
    name: 'Unitree G1',
    dir: 'g1/',
    robotFile: 'g1_29dof.xml',
    policy: 'policy/g1_260808_1834_3d_nolcp.bin',
    sdkJoints: G1_JOINTS,
    legIds: range(0, 12),
    upperIds: range(12, 29),
    footBodies: ['left_ankle_roll_link', 'right_ankle_roll_link'],
    footstep: {
      step_dt: 0.02, future_foot_step_num: 2,
      vrp_height: 0.6258, pelv_com_offset: 0.0678, vrpx_offset: 0.03, vrpy_offset: 0.02,
      vrp_horizon_length: 4.0, preview_horizon_length: 1.6,
      // vrp_generator.h: the stepping-stone z rewrite is disabled for G1
      cube_diagonal_length: 0.0,
      swing_up_timing: 0.2, swing_down_timing: 0.6,
      ik_iters: 10, ik_lambda: 0.05, ik_pos_tol: 0.0001,
      default_joint_pos: legs([-0.4, 0, 0, 0.9, -0.5, 0]),
      action_scale: new Array(12).fill(0.5),
      action_offset: legs([-0.4, 0, 0, 0.9, -0.5, 0]),
      stiffness: legs([100, 100, 100, 150, 40, 40]),
      damping: legs([2, 2, 2, 4, 2, 2]),
      upper_default: [0, 0, 0.2, 0.1, 0.2, 0, 1.1, 0, 0, 0, 0.1, -0.2, 0, 1.1, 0, 0, 0],
      upper_stiffness: [200, 200, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
      upper_damping: [5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
      // global_init_lfoot / rfoot: anchor of the csv_global plan frame
      init_lfoot: { x: -0.01259, y: 0.1185, z: 0.03458, yaw: 0 },
      init_rfoot: { x: -0.01259, y: -0.1185, z: 0.03458, yaw: 0 },
    },
    fixstand: {
      kp: G1_FIXSTAND_KP,
      kd: G1_FIXSTAND_KD,
      q: [-0.4, 0, 0, 0.9, -0.5, 0, -0.4, 0, 0, 0.9, -0.5, 0, 0, 0, 0.2,
          0.1, 0.2, 0, 1.1, 0, 0, 0, 0.1, -0.2, 0, 1.1, 0, 0, 0],
    },
    // Joint hold after the plan: FixStand gains with stiffer hip / ankle roll.
    // With FixStand's roll gains the stance is close to neutrally stable
    // sideways, and the lateral momentum left from walking tips it over.
    hold: {
      kp: G1_FIXSTAND_KP.map((k, i) => (i === 1 || i === 7) ? 300 : (i === 5 || i === 11) ? 150 : k),
      kd: G1_FIXSTAND_KD.map((k, i) => [1, 5, 7, 11].includes(i) ? 5 : k),
    },
    // cmd/gen_cmd.py (+ convert_footcommand_2_global.py anchors)
    plan: {
      // Narrower x / y than gen_cmd.py "02. 3d random footstep sampling" (x, y:
      // [0.2, 0.4]) so that most random 100-step plans are completed: 97/100
      // seeds walked to the end and stood with these ranges, 8/30 with the original.
      range: { x: [0.2, 0.3], y: [0.2, 0.3], z: [-0.15, 0.2], yaw: [-0.4, 0.4] },
      nominalY: 0.237, firstStep: [0.2, 0.237],
      ssp: 0.7, dsp: 0.1, height: 0.07, comZ: 0.01,
      init_lfoot: { x: -0.01259, y: 0.1185, z: 0.03458, yaw: 0 },
      init_rfoot: { x: -0.01259, y: -0.1185, z: 0.03458, yaw: 0 },
    },
    // cmd/gen_rocky_mountain.py
    terrain: {
      ...ROCKY,
      stoneOffset: [0.035, 0, -0.03458],
      clearance: 0, flushRadius: 0.3, flushBlend: 0.3, flushDrop: 0.2,
      flushDropBehindOnly: true,  // drop only on the stone's approach side
      platformToGround: false,    // fixed 10 cm platform with its top at z = 0
      targetMeshes: ['left_ankle_roll_link', 'right_ankle_roll_link'],
    },
    defaultSeed: 6,
  },

  t1: {
    name: 'Booster T1',
    dir: 't1/',
    robotFile: 'T1_23dof.xml',
    policy: 'policy/t1_260908_0133_3d.bin',
    sdkJoints: T1_JOINTS,
    legIds: range(11, 23),
    upperIds: range(0, 11),
    footBodies: ['left_foot_link', 'right_foot_link'],
    footstep: {
      step_dt: 0.02, future_foot_step_num: 2,
      vrp_height: 0.5515, pelv_com_offset: 0.0844, vrpx_offset: 0.08, vrpy_offset: 0.02,
      vrp_horizon_length: 4.0, preview_horizon_length: 1.6,
      // t1_controller's vrp_generator.h keeps the Tocabi stepping-stone rewrite
      cube_diagonal_length: 0.361,
      swing_up_timing: 0.2, swing_down_timing: 0.6,
      ik_iters: 10, ik_lambda: 0.05, ik_pos_tol: 0.0001,
      default_joint_pos: legs([-0.2, 0, 0, 0.4, -0.2, 0]),
      action_scale: legs([0.24, 0.17, 0.17, 0.325, 0.38, 0.38]),
      action_offset: legs([-0.2, 0, 0, 0.4, -0.2, 0]),
      stiffness: legs([100, 100, 100, 100, 50, 50]),
      damping: legs([2, 2, 2, 2, 1, 1]),
      upper_default: [0, 0, 0.2, -1.3, 0, -0.5, 0.2, 1.3, 0, 0.5, 0],
      upper_stiffness: [7.11, 7.11, 111.54, 111.54, 111.54, 111.54, 111.54, 111.54, 111.54, 111.54, 188.76],
      upper_damping: [0.45, 0.45, 7.10, 7.10, 7.10, 7.10, 7.10, 7.10, 7.10, 7.10, 12.02],
      init_lfoot: { x: 0, y: 0.10625, z: 0.043, yaw: 0 },
      init_rfoot: { x: 0, y: -0.10625, z: 0.043, yaw: 0 },
    },
    fixstand: {
      kp: [7.11, 7.11, 111.54, 111.54, 111.54, 111.54, 111.54, 111.54, 111.54, 111.54, 188.76,
           ...legs([206.83, 188.76, 188.76, 251.09, 268.10, 268.10])],
      // config.yaml lists 22 kd values for the 23 motors (no waist entry), so
      // State_FixStand applies them shifted by one from the waist on; the last
      // motor reads past the end of the list (taken as 1.0 here).
      kd: [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1],
      q: [0, 0, 0.2, -1.3, 0, -0.5, 0.2, 1.3, 0, 0.5, 0, ...legs([-0.2, 0, 0, 0.4, -0.2, 0])],
    },
    hold: null, // set below: FixStand gains
    plan: {
      range: { x: [0.2, 0.25], y: [0.2, 0.3], z: [-0.1, 0.15], yaw: [-0.4, 0.4] },
      nominalY: 0.212, firstStep: [0.2, 0.212],
      ssp: 0.8, dsp: 0.1, height: 0.04, comZ: 0,
      init_lfoot: { x: 0, y: 0.10625, z: 0.043, yaw: 0 },
      init_rfoot: { x: 0, y: -0.10625, z: 0.043, yaw: 0 },
    },
    terrain: {
      ...ROCKY,
      stoneOffset: [0.01, 0, -0.043],
      clearance: 0.01, flushRadius: 0.35, flushBlend: 0.25, flushDrop: 0.002,
      flushDropBehindOnly: false, // uniform 2 mm drop (z-fighting) around each stone
      platformToGround: true,     // platform reaches down to the lowest terrain
      targetMeshes: ['left_foot_link', 'right_foot_link'],
    },
    defaultSeed: 6,
  },
};
ROBOTS.t1.hold = { kp: ROBOTS.t1.fixstand.kp, kd: ROBOTS.t1.fixstand.kd };
