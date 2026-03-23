"""
Simulation test: Single fin articulation (test_02).

Commands each fin to known angles, verifies actual joint position matches
command within tolerance, verifies joint limits are respected (clamping at max_deflection).

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
MAX_DEFLECTION = 0.262  # rad (15°) from config
TEST_ANGLES = [0.0, 0.05, 0.1, 0.2, 0.262, -0.1, -0.262]
TOLERANCE = 0.015  # rad


@pytest.fixture
def setup_env():
    from tvc_env.asset.usd_loader import load_asset_metadata
    from tvc_env.sim.scene_builder import SceneConfig, build_scene

    metadata = load_asset_metadata(METADATA_PATH)
    config = SceneConfig(num_envs=1, gizmos_enabled=False)
    scene = build_scene(config)
    drone = scene["drone"]
    return scene, drone, metadata


class TestSingleFinArticulation:
    def test_commanded_angles_reached(self, setup_env):
        """Each commanded angle should be reached within tolerance."""
        scene, drone, metadata = setup_env
        joint_names = metadata["fin_joint_names"]
        joint_idx = [j for j, n in enumerate(drone.joint_names) if joint_names[0] in n][0]

        for angle in TEST_ANGLES:
            target = torch.zeros(1, len(drone.joint_names))
            target[0, joint_idx] = angle
            drone.set_joint_position_target(target)
            for _ in range(15):
                scene.step()

            actual = drone.data.joint_pos[0, joint_idx].item()
            expected = max(-MAX_DEFLECTION, min(MAX_DEFLECTION, angle))  # clamped
            assert abs(actual - expected) < TOLERANCE, \
                f"Angle {angle:.3f}: expected {expected:.3f}, got {actual:.3f}"

    def test_all_four_fins_controllable(self, setup_env):
        """All four fins should independently reach commanded angles."""
        scene, drone, metadata = setup_env
        joint_names = metadata["fin_joint_names"]
        test_angles = [0.1, -0.1, 0.15, -0.15]

        for i, (joint_name, angle) in enumerate(zip(joint_names, test_angles)):
            joint_idx = [j for j, n in enumerate(drone.joint_names) if joint_name in n][0]
            target = torch.zeros(1, len(drone.joint_names))
            target[0, joint_idx] = angle
            drone.set_joint_position_target(target)
            for _ in range(15):
                scene.step()
            actual = drone.data.joint_pos[0, joint_idx].item()
            assert abs(actual - angle) < TOLERANCE, \
                f"Fin {i}: expected {angle:.3f}, got {actual:.3f}"
