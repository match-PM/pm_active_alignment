# Simulation batches and launch

`active_alignment_simulation` is a separate package and ROS node. It exposes
`/active_alignment_batch/execute`, using
`active_alignment_interfaces/action/RunAlignmentBatch`. The server continues to
offer the eight ordinary optimization actions; batch logic is not embedded there.

The complete editable example is
[`simulation_parameter.yaml`](../active_alignment_simulation/config/simulation_parameter.yaml).
Copy it into your experiment directory before running a batch. Results are written
beside this file by default, so using a copied file also keeps results out of the
package installation.

## Launch

With an existing robot and assembly-manager environment:

```bash
ros2 launch active_alignment_bringup active_alignment.launch.py \
  optimization_parameter_file:=/home/niklas/experiments/optimization_parameter.yaml \
  simulation_parameter_file:=/home/niklas/experiments/simulation_parameter.yaml
```

This starts the optimizer, optical simulation publisher, and batch action node.
It advertises the actions without automatically sending a goal or moving joints.

Launch arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `optimization_parameter_file` | Installed optimization defaults | Server startup YAML |
| `simulation_parameter_file` | Installed simulation example | Batch and publisher configuration |
| `use_sim_time` | `false` | Pass simulation-clock mode to these three nodes |

The launch file starts only nodes from this repository. Start the robot simulator,
controllers, and assembly manager separately. When using simulated time, the
surrounding environment must publish `/clock`.

Start a batch explicitly:

```bash
ros2 action send_goal /active_alignment_batch/execute \
  active_alignment_interfaces/action/RunAlignmentBatch \
  "{simulation_parameter_file: /home/niklas/experiments/simulation_parameter.yaml}" --feedback
```

An empty goal path uses the batch node's `simulation_parameter_file` parameter.
All paths are interpreted by the node that uses them; distributed machines need
shared paths for per-run YAML and result directories.

## Compact simulation example

```yaml
alignment_server: /active_alignment_server
publisher_node: /multi_frame_deviation_node
world_frame: world
results_directory: simulation_results
mode: 2D_TRANS
seed_base: 42
optimizer_seed_base: 1000
noise_seed_base: 2000
initial_joint_positions:
  SP_X_Joint: 0.0
  SP_Y_Joint: 0.0

model:
  policy: optical
  mode_field_radius: 0.0000052
  wavelength: 0.00000155
  refractive_index: 1.0
  search_factor: 2.0
  noise_mode: low

frame_pairs:
  - name: Test1
    frame_a: AL_Test_Frames_Align_1
    frame_b: AL_Test_Frames_Target_1
target:
  frame_name: AL_Test_Frames_Target_1
  reference_frame: AL_Test_Frames_Align_1

signal_verification:
  topic: /Test1/eta_clean
  tolerance: 0.001
  timeout: 5.0

alignment_topics:
  - topic_name: /Test1/signal
    minimize_not_maximize: false
    weight_factor: 1.0

seed_groups:
  - power_percent: 20.0
    num_seeds: 3

optimization_parameters:
  general:
    max_evaluations: 80
    wait_for_settle: 0.0
    stop_on_objective_threshold: true
    objective_threshold: -0.99  # Optional, for this maximized 0..1 simulation signal.

tasks:
  - algorithm: nelder_mead
    iterations: 5
    params:
      simplex_step: 0.05
  - algorithm: nelder_mead_manual
    iterations: 5
    params:
      simplex_step: 0.05
```

This example assumes both frames already exist. See the complete example's
`frames` list to create missing frames with parent-relative positions and xyzw
quaternions. Poses and translations use metres; rotations use radians internally.

The configuration above runs 3 seeds × 2 tasks × 5 repetitions = 30 complete runs.
`iterations: 0` disables a task. `general.max_evaluations` limits measurements within
one run. You can repeat an algorithm in separate tasks with different parameters.

## Parameter resolution for tasks

At batch startup the runner reads a full snapshot of the server's optimizer
parameters and validates every enabled task before moving the robot. Resolution
for each task is:

1. The server's pre-batch snapshot.
2. Optional top-level `optimization_parameter_file`.
3. Optional task `parameter_file`.
4. Top-level `optimization_parameters`, containing `general` and algorithm namespaces.
5. Task `general`, applied to shared settings for that run.
6. Task `params`, applied only to that task's algorithm namespace.

Files resolve relative to the simulation YAML. The result is written to a new
per-run `optimization_parameter.yaml` and passed in `ActiveAlign.parameter_file`.
`ActiveAlign.rsap_path` is the run directory. This makes parameter loading and
goal execution one server operation rather than a racy set-parameters service
call followed by a goal. Each run includes all namespaces, preventing parameters
from leaking between tasks. Batch cleanup restores the pre-batch server snapshot.

The optimizer RNG defaults to `optimizer_seed_base + repetition_index`; an explicit
task `general.random_seed` takes precedence. Measurement noise uses
`noise_seed_base + repetition_index`; problem seeds use a separate base. The noise
seed determines the publisher's stream; asynchronous sampling means algorithms
are not guaranteed to receive identical noise draws at the same evaluation index.
Every run records the actual seeds used.

The batch temporarily reserves the alignment server using `batch_directory`.
Ordinary optimization goals outside that directory are rejected until cleanup.
A second batch cannot replace an existing reservation. Other independent motion
clients must also leave the robot under the experiment's control.

## Setup loop and restoration

The simulator remains running. For every seed, task, and repetition, the runner:

1. Restores fixed physical starting joint positions and verifies the movement.
2. Applies the seed's target pose through `/assembly_manager/modify_frame_absolut`.
3. Waits for the requested world pose to propagate through TF.
4. Resets the publisher's noise seed, settles, and verifies a fresh clean initial
   coupling sample against the generated problem's expected coupling.
5. Calls the selected ordinary optimization action and records its outcome.
6. Restores starting joint positions before the next run.

At startup the runner subscribes to `/assembly_manager/scene` and obtains a current
snapshot through `/assembly_manager/get_scene`. It saves complete reference-frame
definitions, including parent, constraints, and properties. Missing configured
frames are created using `/assembly_manager/create_ref_frame`. Existing frames
are retained. No description JSON is rewritten.

`ModifyPoseAbsolut` expects a **world pose**. The runner obtains the world pose of
`target.reference_frame` after resetting the robot and composes seed translation
and rotation in that reference's local coordinates. If `reference_frame` is
omitted, the target's initial world pose becomes the baseline; it must represent
alignment for target-power verification to succeed.
If `world_frame` names another experiment coordinate frame, the composed pose is
converted into the assembly manager's `world` coordinates before the service call.

`initial_joint_positions` specifies physical positions explicitly. If omitted,
`reset_position` is mapped once using the optimization intervals captured at batch
startup, then the resulting physical positions are reused. It does not repeatedly
recompute a midpoint around each run's final pose.

Cleanup restores existing frame definitions, deletes frames created by the batch,
restores publisher settings and optimizer parameters, and releases the reservation.
On cancellation the child optimization is cancelled and awaited before scene
restoration; no new robot return move is issued. If a child fails to acknowledge
cancellation within `service_timeout`, the batch aborts and preserves its
reservation and setup state. Stop the child and inspect the recorded state before
restoring the scene or clearing `batch_directory` manually.

`continue_on_failure: true` retains failed runs and continues after ordinary
per-run setup/optimization errors. Unacknowledged cancellation stops the batch.
Configuration errors stop before experiment setup. Cleanup failures make the batch
result unsuccessful and are recorded explicitly.

## Model and seeds

The model retains the working branch's lateral Gaussian, angular Gaussian, and
axial Lorentzian coupling terms. Roll about the optical axis is unobservable.
Supported modes are `5D`, `2D_TRANS`, `2D_ROT`, and `3D_TRANS`, or an explicit
`enabled_joints` mapping using X/Y/Z/A/B. The generic optimizer still supports other
robot joints; this restriction belongs to the optical experiment generator.

Seeds are generated inside the enabled-DOF search box with the requested clean
power. Disabled axes remain zero. Impossible powers are rejected rather than
silently replaced; exact zero coupling is unreachable with this finite model.
A group with `num_seeds: 0` is skipped, so a disabled 0% group is permitted.

The publisher's `optical` policy follows the working branch: `/distance` is the
physical distance in metres, `/angle_deviation` is tilt, and `/signal` is noisy
coupling. It also publishes clean efficiencies, percentages, and the complete
rotation on `/angle`. Frame pairs, model, noise coefficients, and publish rate
come from YAML. Frame pairs and publish rate are fixed at startup. Policy, model,
and noise settings can change at runtime and are restored after a batch.

## Editing or selecting the measurement policy

The node only reads transforms, calls a policy, and publishes its returned values.
Policy calculations are separate files:

- `active_alignment_skills/active_alignment_skills/simulation/policies/optical.py`:
  edit `CouplingModel.evaluate` to change the optical equations. `OpticalPolicy`
  applies measurement noise while retaining `eta_clean` for verification.
- `active_alignment_skills/active_alignment_skills/simulation/policies/geometric.py`:
  the original main-branch distance × 10 and full-rotation policy.
- `simulation/policies/__init__.py`: select the policy with an explicit factory.

Choose the policy in the simulation YAML:

```yaml
model:
  policy: optical  # or geometric
  legacy_distance_scale: 10.0  # used only by geometric
```

Or change it on a running standalone publisher:

```bash
ros2 param set /multi_frame_deviation_node policy geometric
ros2 param set /multi_frame_deviation_node policy optical
```

`geometric` publishes only `/distance` and `/angle`. The optical signal and
efficiency topics stop updating while that policy is active. Optical batches
select `optical` for their duration and restore the previous policy afterward.
A custom policy can implement the same `evaluate(translation, quaternion)` method
and return a dictionary of topic suffixes to numeric values. Add any new topic
suffixes to the publisher's topic list as well.

To override shared execution settings for one task:

```yaml
tasks:
  - algorithm: nelder_mead
    iterations: 5
    general:
      max_evaluations: 120
      measurement_timeout: 3.0
      wait_for_settle: 0.0
    params:
      simplex_step: 0.05
```

The node parameter names remain `general.max_evaluations`,
`general.measurement_timeout`, and `nelder_mead.simplex_step`. Per-task settings
are written to the goal's parameter YAML before the optimization starts.

## Results

```text
<simulation YAML directory>/simulation_results/<timestamp>_batch_<ID>/
  simulation_parameter.yaml
  seed_poses.yaml
  initial_frames.json
  batch_summary.json
  run_00001_nelder_mead_seed_42_rep_1/
    optimization_parameter.yaml
    result.json
    optimization_logs/<timestamp>_nelder_mead_<ID>/
      optimization_parameter.yaml
      history.csv
      metadata.json
      summary.txt
```

The summary is updated after each run. Failures retain their error and partial
history; missing data is not padded with fabricated zero measurements. Each
successful optimization log identifies the best measured row, which can differ
from the final row. Batch logs automatically include the optical diagnostics from
the signal-verification topic's prefix unless a task supplies another
`general.metrics_topic_prefix`. The batch result reports successful and failed run counts
and the absolute results directory.

Batch helpers (`config.py`, `scene_frames.py`, `seed_generator.py`, `observations.py`)
live under `active_alignment_simulation/active_alignment_simulation/helpers/`;
only the batch node and `__init__.py` remain at that package root.
