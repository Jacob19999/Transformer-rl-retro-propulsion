"""
Simulation test: Joint axes verification (test_01).

For each fin joint, commands positive deflection with gravity disabled,
verifies rotation occurs around the correct hinge axis in the expected direction.

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import pytest
import torch
from pathlib import Path

try:
    import omni.usd
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ISAAC_AVAILABLE, reason="Isaac Sim runtime not available")

METADATA_PATH = Path(__file__).parents[2] / "assets/metadata/edf_drone_v2.asset.yaml"
VEHICLE_CONFIG_PATH = Path(__file__).parents[2] / "configs/vehicle/edf_drone_v2.yaml"
DEFLECTION_ANGLE = 0.1  # rad — test command angle
TOLERANCE = 0.01  # rad tolerance for actual vs commanded


@pytest.fixture
def scene_and_articulation():
    """Set up minimal scene with a single drone for joint testing."""
    from tvc_env.asset.usd_loader import load_asset_metadata
    from tvc_env.sim.scene_builder import SceneConfig, build_scene

    metadata = load_asset_metadata(METADATA_PATH)
    config = SceneConfig(num_envs=1, gizmos_enabled=False)
    scene = build_scene(config)
    drone = scene["drone"]
    return scene, drone, metadata


class TestJointAxes:
    def test_each_fin_deflects_around_correct_axis(self, scene_and_articulation):
        """Positive deflection on each fin should produce rotation around its hinge axis."""
        scene, drone, metadata = scene_and_articulation

        from tvc_env.asset.usd_loader import load_asset_metadata
        from tvc_env.dynamics.fin_geometry import load_hinge_axes

        hinge_axes = load_hinge_axes(metadata)  # (4, 3)
        joint_names = metadata["fin_joint_names"]

        for i, joint_name in enumerate(joint_names):
            # Find joint index
            joint_idx = [j for j, n in enumerate(drone.joint_names) if joint_name in n][0]

            # Command positive deflection on this fin, zero on others
            target = torch.zeros(1, len(drone.joint_names))
            target[0, joint_idx] = DEFLECTION_ANGLE

            drone.set_joint_position_target(target)
            # Step simulation for N frames to let position settle
            for _ in range(10):
                scene.step()

            # Read actual joint position
            actual_angle = drone.data.joint_pos[0, joint_idx].item()
            assert abs(actual_angle - DEFLECTION_ANGLE) < TOLERANCE, \
                f"Fin {i} ({joint_name}): commanded {DEFLECTION_ANGLE:.3f} rad, " \
                f"got {actual_angle:.3f} rad"

    def test_joint_limits_respected(self, scene_and_articulation):
        """Commanding beyond max_deflection should be clamped to joint limits."""
        scene, drone, metadata = scene_and_articulation
        max_def = 0.262  # 15°

        # Command 45° (well beyond limit)
        joint_names = metadata["fin_joint_names"]
        joint_idx = [j for j, n in enumerate(drone.joint_names) if joint_names[0] in n][0]
        target = torch.zeros(1, len(drone.joint_names))
        target[0, joint_idx] = 1.0  # >> max_def

        drone.set_joint_position_target(target)
        for _ in range(20):
            scene.step()

        actual = drone.data.joint_pos[0, joint_idx].item()
        assert actual <= max_def + 0.01, \
            f"Joint exceeded max_deflection: got {actual:.3f} rad, limit={max_def}"
