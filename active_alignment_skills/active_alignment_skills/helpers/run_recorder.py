"""Incremental, per-run result storage in writable user-selected locations."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import yaml


def unique_run_directory(base, label):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = Path(base) / f'{stamp}_{label}_{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True)
    return path


class RunRecorder:
    """Write every observation immediately; retain partial failed/cancelled runs."""

    def __init__(self, directory, algorithm, parameters, goal):
        self.directory = Path(directory)
        self.metadata = {'algorithm': algorithm, 'status': 'running', 'goal': goal}
        self.stream = (self.directory / 'history.csv').open('w', newline='')
        self.writer = None
        self.count = 0
        with (self.directory / 'optimization_parameter.yaml').open('w') as stream:
            yaml.safe_dump(parameters, stream, sort_keys=False)
        self._save_metadata()

    def record(self, measurement):
        row = {'evaluation': self.count, 'initial': self.count == 0,
               'elapsed_seconds': measurement.elapsed, 'objective': measurement.cost}
        row.update({f'normalized_{i}': float(x) for i, x in enumerate(measurement.position)})
        row.update(measurement.metrics)
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        self.count += 1

    def finish(self, status, session=None, message='', joint_results=None):
        self.metadata.update(status=status, message=message, joint_results=joint_results or {})
        if session is not None:
            self.metadata.update(
                stop_reason=session.stop_reason, optimization_evaluations=session.evaluations,
                total_evaluations=len(session.history), iterations=session.iterations,
                solver_details=session.solver_details,
                best_objective=session.best.cost if session.best is not None else None,
                best_history_index=(next(i for i, record in enumerate(session.history)
                                         if record is session.best)
                                    if session.best is not None else None))
        self.stream.close()
        self._save_metadata()
        with (self.directory / 'summary.txt').open('w') as stream:
            stream.write(json.dumps(self.metadata, indent=2) + '\n')

    def _save_metadata(self):
        with (self.directory / 'metadata.json').open('w') as stream:
            json.dump(self.metadata, stream, indent=2)
