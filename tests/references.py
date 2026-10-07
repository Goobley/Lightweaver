'''
Stored reference values for the smoke tests.

The references in `data/references.npz` are produced by the current code and
compared against with a tight relative tolerance, so that unintended changes
to the results are caught. When a change to the physics or numerics is
intended, regenerate them with

    LW_REGEN_REFERENCES=1 pytest tests

and commit the updated `data/references.npz` alongside the change.

To print the largest relative difference from each reference without
failing (e.g. to compare iteration schemes), set LW_REFERENCE_REPORT=1.
'''
import os
from pathlib import Path

import numpy as np

ReferencePath = Path(__file__).parent / 'data' / 'references.npz'
# Set from the spread of these quantities across the scalar, SSE2 and
# AVX2FMA iteration schemes, with a safety margin.
DefaultRtol = 1e-5


def env_flag(name: str) -> bool:
    return os.environ.get(name, '0') not in ('', '0')


def regenerating() -> bool:
    return env_flag('LW_REGEN_REFERENCES')


class ReferenceStore:
    def __init__(self, path: Path = ReferencePath):
        self.path = path
        self.regen = regenerating()
        self.report = env_flag('LW_REFERENCE_REPORT')
        self.new = {}
        if path.exists():
            with np.load(path) as data:
                self.stored = {k: data[k] for k in data.files}
        else:
            self.stored = {}

    def check(self, name: str, value, rtol: float = DefaultRtol, atol: float = 0.0):
        '''
        Compare `value` against the stored reference `name` (or record it if
        regenerating).
        '''
        value = np.asarray(value, dtype=np.float64)
        if self.regen:
            self.new[name] = value
            return

        if name not in self.stored:
            raise KeyError(f'No stored reference "{name}"; regenerate with LW_REGEN_REFERENCES=1.')
        if self.report:
            ref = self.stored[name]
            rel = np.max(np.abs(value - ref) / np.maximum(np.abs(ref), np.finfo(np.float64).tiny))
            print(f'\nREFERENCE {name}: max rel diff {rel:.3e}')
            return
        np.testing.assert_allclose(value, self.stored[name], rtol=rtol, atol=atol,
                                   err_msg=f'Reference "{name}" changed')

    def save(self):
        if not self.regen or len(self.new) == 0:
            return
        merged = dict(self.stored)
        merged.update(self.new)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **merged)
