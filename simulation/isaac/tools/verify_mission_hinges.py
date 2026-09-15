"""Verify real recorded fin-link poses rotate about each radial body axis."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def verify(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('hinge_layout') != 'radial_span_v1':
        raise ValueError('This recording predates radial hinge physics')
    frames = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
    def rotations(values):
        q = np.array(values)
        return Rotation.from_quat(q[:, [1, 2, 3, 0]])
    bodies = rotations([f['quaternion'] for f in frames])
    # Isaac body axes: forward +X, right -Y, aft -X, left +Y.
    axes = np.array([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]])
    angles = np.array([f['fin_angles'] for f in frames])
    results = []
    for i, name in enumerate(('FWD', 'RIGHT', 'AFT', 'LEFT')):
        fins = rotations([f['fin_quaternions'][i] for f in frames])
        relative = bodies.inv() * fins
        expected = Rotation.from_rotvec(angles[:, i, None] * axes[i])
        error = np.degrees((expected.inv() * relative).magnitude()).max()
        assert error < .02, f'{name}: fin pose differs from radial hinge by {error} degrees'
        # An axial chord vector must sweep tangentially, never along its span.
        edge = relative.apply(np.tile([0, 0, -.078], (len(frames), 1)))
        span_motion = np.abs(edge @ axes[i]).max()
        assert span_motion < .00003, f'{name}: trailing edge moved along radial span'
        results.append(dict(fin=name, max_angle_deg=float(np.abs(np.degrees(angles[:, i])).max()),
                            max_pose_error_deg=float(error), max_span_motion_m=float(span_motion)))
    return dict(mission=folder.name, samples=len(frames), passed=True, fins=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = verify(args.folder)
    (args.folder / 'hinge_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
