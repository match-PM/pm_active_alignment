# Algorithm port and correctness review

Source comparison: `main` at `701c136` and the working branch
`origin/leonard_ba` at `016437b`. Implementations were reworked into separate
classes rather than merged into the ROS node. Both stochastic/main hill climbing
and working-branch coordinate hill climbing are retained, as are SciPy and
handwritten Nelder–Mead. Py-BOBYQA remains an optional library implementation.

## Corrections

- The requested weighted objective is used by every optimizer. The working
  branch's simulation `signal` no longer bypasses requested topics or their
  minimize/maximize flags. Direction is converted once to a minimization cost.
- All optimizers return physical joint positions through ROS. Normalized values
  remain internal, and inactive controller joints retain their initial values.
- Finalization restores and verifies the best observed position for every
  strategy, including both Nelder–Mead implementations and Py-BOBYQA.
- Algorithms no longer assume physical zero equals normalized 0.5 or recenter
  the robot before an ordinary alignment. Experiment resets are separate.
- Evaluation budgets count actual measurements consistently, excluding exactly
  one initial measurement. Initial library requests reuse that recorded sample.
- Both simplex implementations initialize distinct vertices at upper bounds.
  The manual implementation includes outside contraction, inside contraction,
  expansion, and shrink; the retained reflected point is tracked even if the
  budget expires during an expansion or contraction evaluation.
- A failed outside contraction falls through to shrink. The manual and SciPy
  classical algorithms agree numerically on an interior quadratic benchmark.
- Adaptive hill climbing stops after a failed sweep at minimum step size. It no
  longer relies on `step < min_step` after clamping the step to `min_step`.
- Random search and adaptive random search always compare against the best
  measured point and record a threshold-reaching final candidate consistently.
- Py-BOBYQA validates interpolation-set requirements, radii, and solver errors.
  The initial radius cannot exceed half the normalized bound width, and its
  final radius must be strictly smaller. Other algorithms import successfully
  without Py-BOBYQA installed.
  Its internal RNG is seeded per run and the previous NumPy global state is
  restored afterward. The library's built-in absolute objective threshold is
  disabled so arbitrarily scaled negative costs use the shared stopping policy.
- Solver, movement, non-finite measurement, and configuration errors abort
  actions. Cancellation has one terminal-state owner. Neither error is silently
  reported as optimization success.
- Every objective source must publish after motion and settling. Extra metrics
  also require fresh receipts and remain independent of the objective.
- Every per-run subscription is destroyed on success, failure, and cancellation.
  No private ROS goal-handle dictionary manipulation or client-reset workaround
  is used. Service, trajectory, measurement, and batch waits are bounded.
- Target-power seed generation keeps disabled coordinates zero while solving
  the requested coupling, instead of discarding components afterward. Impossible
  powers are errors rather than silently substituted problems.

## Deliberate behavior

Fixed-step coordinate hill climbing stops when a sweep finds no improvement;
repeating identical deterministic sweeps would consume the remaining measurement
budget without changing its search. Stochastic hill climbing retains its original
random perturbation behavior, with adaptation optional.

All strategies optimize a black-box scalar objective. The code does not claim a
global optimum, and noisy best-observed values are not unbiased estimates. Repeated
observations are retained. Comparisons should use the same physical starting pose,
problem seeds, stopping rules, and evaluation budgets.

## Verification

Pure numerical tests cover all eight strategies, known optima, bound starts,
evaluation budgets, initial-point targets, cancellation, non-finite values,
random-seed reproducibility, parameter precedence/validation, result-row
consistency, and target-power seeds for single-axis and mixed-axis modes.

ROS integration tests use fake trajectory controllers, joint states, objective
publishers, and assembly-manager services in an isolated domain. They exercise
physical result conversion, final positioning, inactive joints, concurrent goal
rejection, cancellation, controller failure, atomic YAML updates, the full batch
loop, frame creation/restoration/deletion, parameter restoration, and setup failure.
They do not move real hardware or validate the physical optical model.

Run after building the interfaces, with the two Python packages importable:

```bash
export ROS_DOMAIN_ID=187
export ROS_LOCALHOST_ONLY=1
export PYTHONDONTWRITEBYTECODE=1
python3 -m pytest -q \
  src/pm_active_alignment/active_alignment_skills/test/test_optimizers.py \
  src/pm_active_alignment/active_alignment_skills/test/test_parameters_and_recording.py \
  src/pm_active_alignment/active_alignment_skills/test/test_ros_execution.py \
  src/pm_active_alignment/active_alignment_simulation/test
```

Use a ROS domain not used by the robot environment. Optional Py-BOBYQA tests skip
when the dependency is unavailable. Existing repository-wide lint checks also
inspect legacy modules which were not part of this port.
