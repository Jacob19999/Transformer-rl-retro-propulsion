"""Check physical joints against mesh span, independently of control code."""
from pathlib import Path
import numpy as np
import pytest
from tvc_env.asset.usd_loader import load_asset_metadata


def test_usd_hinges_follow_fin_span_and_match_force_metadata():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    root = Path(__file__).resolve().parents[2]
    metadata = load_asset_metadata(root / 'assets/metadata/edf_drone_v2.asset.yaml')
    stage = Usd.Stage.Open(str(root / 'assets/usd/drone_v2_physics.usd'))
    for name, expected in zip(metadata['fin_link_names'], metadata['hinge_axes']):
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Drone/Body/joint_'+name))
        axis = np.eye(3)['XYZ'.index(joint.GetAxisAttr().Get())]
        actual = np.array(Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(Gf.Vec3d(*axis)))
        np.testing.assert_allclose(actual * [1,-1,-1], expected, atol=1e-6)
        points = np.array(UsdGeom.Mesh(stage.GetPrimAtPath('/Drone/'+name+'/'+name)).GetPointsAttr().Get())
        extent = np.ptp(points, axis=0)
        span = np.zeros(3);span[np.argmax(extent[:2])] = 1
        normal = np.zeros(3);normal[np.argmin(extent)] = 1
        assert abs(actual @ span) > .999
        assert abs(actual @ normal) < 1e-6
        assert joint.GetLocalRot0Attr().Get() == joint.GetLocalRot1Attr().Get()
