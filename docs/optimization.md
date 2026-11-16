# Optimization actions and parameters

The server exposes one `active_alignment_interfaces/action/ActiveAlign` action
per algorithm. The existing endpoint names retain their original algorithm:

| Algorithm ID / endpoint suffix | Implementation | Parameter namespace |
| --- | --- | --- |
| `hill_climb` | Original stochastic hill climbing | `hill_climb.*` |
| `hill_climb_manual` | Working-branch coordinate/simultaneous hill climbing | `hill_climb_manual.*` |
| `adaptive_hill_climb` | Working-branch adaptive coordinate hill climbing | `adaptive_hill_climb.*` |
| `nelder_mead` | SciPy `minimize(method="Nelder-Mead")` | `nelder_mead.*` |
| `nelder_mead_manual` | Classical handwritten Nelder–Mead | `nelder_mead_manual.*` |
| `pybobyqa` | Optional Py-BOBYQA library | `pybobyqa.*` |
| `random_search` | Global or local uniform random search | `random_search.*` |
| `simple_adaptive_random_search` | Uniform random search with a sliding-window 1/5 success rule | `simple_adaptive_random_search.*` |

Each action is named `/active_alignment_server/exec_active_alignment_<ID>`.
The algorithms live in separate modules under
`active_alignment_skills/active_alignment_skills/optimization/` and share bounds,
evaluation accounting, objective measurement, recording, and cancellation.
An accepted goal reserves the robot across all eight endpoints until cleanup.
Competing goals are rejected.

## Build and startup

The action definition changed. Rebuild the interface and its clients, and restart
nodes using the old generated interface:

```bash
cd ~/ros2_ws
colcon build --packages-select active_alignment_interfaces active_alignment_skills \
  active_alignment_simulation active_alignment_bringup
source install/setup.bash
ros2 launch active_alignment_bringup active_alignment.launch.py
```

The launch file loads the installed
`active_alignment_skills/config/optimization_parameter.yaml` at startup. The
server also loads these defaults when started directly:

```bash
ros2 run active_alignment_skills active_alignment_skill_node
```

An alternative startup file may use either compact YAML or ordinary ROS YAML:

```bash
ros2 run active_alignment_skills active_alignment_skill_node --ros-args \
  -p initial_parameter_file:=/absolute/path/optimization_parameter.yaml
```

Ordinary ROS parameter overrides and `--params-file` are supported too. An explicit
`initial_parameter_file` is applied after declared startup overrides. Files are
partial overrides: unspecified settings retain their current values. The installed
default file is the complete parameter reference, including numeric types.

Py-BOBYQA is optional and is imported only when selected. Install `Py-BOBYQA`
in the Python environment used by ROS to enable that endpoint. Verification was
performed with Py-BOBYQA 1.5.0. Other algorithms work without it.

Py-BOBYQA uses seeded random interpolation directions by default
(`pybobyqa.random_initial_directions: true`) to reduce singular geometry failures
around narrow, noisy peaks. `general.random_seed` keeps those directions reproducible.
Its trust-region radii use normalized joint ranges: with a 20 mm search interval,
`initial_tr_radius: 0.1` means 2 mm. For micrometre-scale signals, a starting radius
of `0.0001` corresponds to 2 micrometres; keep `final_tr_radius` smaller than it.

If a singular-matrix/geometry error recurs, the terminal error includes these
troubleshooting hints and the run's results directory:

1. Keep `pybobyqa.random_initial_directions: true` and retry with another
   `general.random_seed` to change the interpolation geometry.
2. Inspect `history.csv`. If most raw signal values are flat or zero, reduce
   `pybobyqa.initial_tr_radius` to sample within the signal's useful width.
3. Keep `0 < final_tr_radius < initial_tr_radius`; avoid physical steps below
   the robot's motion resolution or the sensor's ability to distinguish positions.
4. If it persists, try `nelder_mead` or `adaptive_hill_climb`. Keep `metadata.json`,
   `history.csv`, and `optimization_parameter.yaml` from the failed run for diagnosis.

## Goal fields and path precedence

`ActiveAlign.Goal` contains:

```text
JointAlign[] active_joints
AlignTopic[] alignment_topics
string parameter_file
string rsap_path
```

Parameter lookup follows these rules:

1. A nonempty `parameter_file` selects that YAML. A relative path is resolved
   against the RSAP directory if supplied, otherwise against the server's working
   directory. An invalid explicit path fails the goal; it does not fall back.
2. Otherwise a nonempty `rsap_path` selects
   `<RSAP directory>/optimization_parameter.yaml` if it exists. If it is missing,
   use the node's current parameter values (initialized from the packaged defaults).
3. With both fields empty, use the node's current parameter values.

`rsap_path` can be the RSAP JSON file or its containing directory. A `.json` or
`.rsap` filename is treated as a file even if it has not yet been written. The
RSAP file is never opened or modified.

With `rsap_path`, logs go to
`<RSAP directory>/optimization_logs/<UTC timestamp>_<algorithm>_<unique ID>/`.
Without it, the parent is the node's `results_directory` parameter, which defaults
to `~/.ros/active_alignment`. The directories must be writable by the server.

## Example optimization_parameter.yaml

The compact form is convenient beside an RSAP program:

```yaml
general:
  max_evaluations: 80
  movement_time: 0.1
  wait_for_settle: 0.0
  measurement_timeout: 5.0
  trajectory_timeout: 20.0
  joint_tolerance: 0.000001
  random_seed: 1000
  stop_on_objective_threshold: false
  objective_threshold: 0.0

nelder_mead:
  max_iterations: 200
  simplex_step: 0.05
  xatol: 0.00001
  fatol: 0.00001
  adaptive: false

nelder_mead_manual:
  alpha: 1.0  # reflection
  beta: 2.0   # expansion
  gamma: 0.5  # contraction
  delta: 0.5  # shrink

adaptive_hill_climb:
  step_size: 0.1
  adaptive_step: true
  step_increase_factor: 1.3
  step_decrease_factor: 0.3
  min_step: 0.001
  max_step: 0.5
  coordinate_wise: true

simple_adaptive_random_search:
  sigma: 0.25
  alpha: 1.1
  gamma: 0.9
  min_sigma: 0.01
  max_sigma: 1.0
  window_size: 10
```

The `general` section is shared by every optimizer. Budgets, timing, joint tolerance,
random seeds, optional diagnostics, and objective stopping live there once. Algorithm
sections contain only their own search settings. For example,
`nelder_mead.xatol` controls simplex convergence while
`general.measurement_timeout` controls waiting for sensor data for every algorithm.

Previous files must move shared settings out of algorithm sections. The earlier
`target_enabled` / `target_value` pair is now
`general.stop_on_objective_threshold` / `general.objective_threshold`.

The corresponding ROS format wraps the same settings:

```yaml
active_alignment_server:
  ros__parameters:
    general:
      max_evaluations: 80
      wait_for_settle: 0.0
    nelder_mead:
      xatol: 0.00001
```

Flat dotted keys and the `/**` ROS wildcard are also accepted. Unknown keys,
incorrect types, invalid factors, and non-finite values fail before movement.
Settings are applied atomically and remain visible through ROS parameters:

```bash
ros2 param get /active_alignment_server general.max_evaluations
ros2 param set /active_alignment_server general.max_evaluations 100
```

Each run uses an immutable snapshot. Changes made during a run affect subsequent
goals. Loading an action's YAML changes node parameters persistently; it is not
an ephemeral override. Batch testing restores the pre-batch snapshot at the end.

## Example action goal

```bash
ros2 action send_goal /active_alignment_server/exec_active_alignment_nelder_mead \
  active_alignment_interfaces/action/ActiveAlign \
  "{active_joints: [
      {joint_name: SP_X_Joint, has_constraint: true, lower_limit: -0.00001, upper_limit: 0.00001},
      {joint_name: SP_Y_Joint, has_constraint: true, lower_limit: -0.00001, upper_limit: 0.00001}],
    alignment_topics: [
      {topic_name: /Test1/signal, minimize_not_maximize: false, weight_factor: 1.0}],
    parameter_file: /home/niklas/experiments/optimization_parameter.yaml,
    rsap_path: /home/niklas/experiments/alignment.rsap.json}" --feedback
```

Joint constraints retain the existing relative-to-start semantics: the lower and
upper values describe permitted displacement from the measured initial position,
intersected with finite URDF limits. The lower bound accepts a negative value or a
positive magnitude, as in existing RSAP descriptions. Normalized `[0,1]` positions
are internal. Returned joint values and logged commanded/measured positions use
physical units: metres for prismatic joints and radians for revolute joints.

## Objective, budgets, and results

All strategies minimize the same weighted objective:

```text
cost = sum(weight * signed_topic_value) / sum(weight)
signed_topic_value = value for minimize, -value for maximize
```

For a single maximized topic with a positive value of `0.8`, the internal
objective is `-0.8`, while the reported `final_eval_value` is the raw value `0.8`.
For multiple topics, `final_eval_value` is their weighted average without direction
sign changes: `sum(weight * raw_topic_value) / sum(weight)`. The signed objective
and individual raw topic values are recorded separately in `history.csv`.

Directions come from `AlignTopic.minimize_not_maximize`; there is no second
`maximize` switch in the optimizer. To stop at 99% coupling, maximize `/Test1/signal`
and set `general.stop_on_objective_threshold: true` and
`general.objective_threshold: -0.99`. The threshold always compares against the
signed weighted objective (`cost <= objective_threshold`), including goals that
mix minimization and maximization topics. The response still reports the raw
weighted topic value in `final_eval_value`. This is an optional early-stop threshold,
not an estimate of the unknown optimum. Leave it disabled for measurements whose
useful range is unknown. Diagnostic metrics never replace the requested objective.

`general.max_evaluations` counts physical measurements after the initial measurement.
An 80-evaluation budget creates at most 81 history rows. A budget of zero records
only the initial point. `max_iterations` is a separate simplex-loop limit for
the two Nelder–Mead implementations. The initial measurement is cached for the
library's first objective request; later repeat requests are genuine measurements.
`num_of_iterations` in the action result reports post-initial evaluations across
all algorithms for consistent comparisons; actual solver iterations are in metadata.

Every evaluation waits for trajectory success, measured joint positions within
`general.joint_tolerance`, optional settling, and a new receipt from **every** requested topic.
`general.wait_for_settle: 0.0` is supported: after the movement finishes,
the next evaluation waits for a newly received sample from each topic. A nonzero
settling time moves that freshness boundary later. `general.measurement_timeout`
is the maximum wait, not a sleep before reading a cached value.
Timeouts and non-finite observations abort the action. Float32 messages do not
carry acquisition timestamps, so freshness means local receipt after settling;
sensors with internal buffering should publish appropriately or use a future
stamped interface for stronger acquisition-time guarantees.

On success, the robot is moved back to the best observed position and that movement
is verified. `final_eval_value` uses the raw topic measurements from the historical
point selected by the signed objective, not a fresh remeasurement after restoring
the point. With noise, the best observed value can
be optimistic; the full history remains available for analysis. Cancellation
cancels pending trajectory motion and does not command a new return movement.
Action success means the run completed normally; reaching a budget or convergence
without reaching the optional objective threshold is still a successful run with its stop reason recorded.

Every algorithm logs execution start, robot initialization, the initial measurement,
and progress every 10 evaluations or after two seconds between measurement updates.
Progress includes the evaluation count, current and best raw topic values, and
elapsed time. The terminal also reports restoration of the best position,
cancellation requests, and completion/cancellation/failure with the stop reason.

Each result directory contains:

- `optimization_parameter.yaml`: full resolved parameter snapshot.
- `history.csv`: initial row followed by each measured candidate, canonical cost,
  raw requested topics, normalized coordinates, and physical commanded/measured joints.
- `metadata.json`: goal, outcome, stop reason, best history-row index, physical
  result, evaluation counts, and solver details.
- `summary.txt`: readable result summary.

History is flushed after each measurement, and failed/cancelled runs retain their
partial results. If `general.metrics_topic_prefix` is nonempty, the server additionally
records fresh coupling efficiencies and distance/angle diagnostic topics from
that prefix. An empty prefix, the default, creates no diagnostic subscriptions.

## Source layout

Only `active_alignment_skill_node.py`, `dummy_value_publisher.py`, and `__init__.py`
are at the Python package root. Supporting code is organized under `helpers/`,
`optimization/`, `py_modules/`, and `simulation/`. The node comments describe the
move → optional settle → fresh measurement → record sequence. Each optimizer
comments its proposal, acceptance, adaptation, and stopping steps.
