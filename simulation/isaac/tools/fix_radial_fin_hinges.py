"""Make USD joint axes follow the radial vane span confirmed by the builder.

The 2026-09-14 camera review exposed tangential joints rotating vanes in their
own planes. Preserve neutral poses and geometry; change joint frames only.
"""
from pathlib import Path
import shutil
from pxr import Usd, UsdPhysics, Gf

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'assets/usd/drone_v2_physics.usd'
    backup = path.with_name('drone_v2_tangential_hinges_legacy.usd')
    if not backup.exists():
        shutil.copy2(path, backup)
    stage = Usd.Stage.Open(str(path))
    for name, axis, reversed_axis in [('FwdFin', 'X', False), ('RightFin', 'Y', True),
                                      ('AftFin', 'X', True), ('LeftFin', 'Y', False)]:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Drone/Body/joint_' + name))
        joint.GetAxisAttr().Set(axis)
        rotation = Gf.Quatf(0, 0, 0, 1) if reversed_axis else Gf.Quatf(1, 0, 0, 0)
        joint.GetLocalRot0Attr().Set(rotation)
        joint.GetLocalRot1Attr().Set(rotation)
    stage.GetRootLayer().Save()
    print('Radial fin hinges saved; old asset preserved at', backup)


if __name__ == '__main__':
    main()
