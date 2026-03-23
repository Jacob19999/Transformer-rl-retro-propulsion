"""
Simulation test: Asset validation (test_00).

Tests:
  - Valid asset metadata passes all structural checks
  - Missing fin link name causes diagnostic failure
  - Undefined joint axis causes diagnostic failure

This test requires Isaac Sim runtime for full USD checks.
The metadata-only portion can be run offline.
"""

from __future__ import annotations
import pytest
import copy
from pathlib import Path
from tvc_env.asset.usd_loader import load_asset_metadata
from tvc_env.asset.mass_properties import load_vehicle_config
from tvc_env.asset.asset_validator import validate_asset, AssetValidationError

# Paths relative to the simulation/isaac/ directory
METADATA_PATH = Path(__file__).parents[2] / "assets/metadata/edf_drone_v2.asset.yaml"
VEHICLE_CONFIG_PATH = Path(__file__).parents[2] / "configs/vehicle/edf_drone_v2.yaml"


@pytest.fixture
def metadata():
    return load_asset_metadata(METADATA_PATH)


@pytest.fixture
def vehicle_config():
    return load_vehicle_config(VEHICLE_CONFIG_PATH)


class TestAssetValidationOffline:
    """Offline metadata-only tests (no Isaac Sim required)."""

    def test_valid_metadata_passes(self, metadata, vehicle_config):
        """Valid metadata should pass all offline checks without raising."""
        diagnostics = validate_asset(metadata, vehicle_config)
        # Diagnostics may have warnings, but no exception
        assert isinstance(diagnostics, list)

    def test_four_fin_links_required(self, metadata, vehicle_config):
        """Missing fin link should raise AssetValidationError."""
        bad_metadata = copy.deepcopy(metadata)
        bad_metadata["fin_link_names"] = bad_metadata["fin_link_names"][:3]  # Remove one
        with pytest.raises(AssetValidationError, match="4 fin link names"):
            validate_asset(bad_metadata, vehicle_config)

    def test_four_fin_joints_required(self, metadata, vehicle_config):
        """Missing fin joint should raise AssetValidationError."""
        bad_metadata = copy.deepcopy(metadata)
        bad_metadata["fin_joint_names"] = bad_metadata["fin_joint_names"][:2]
        with pytest.raises(AssetValidationError, match="4 fin joint names"):
            validate_asset(bad_metadata, vehicle_config)

    def test_non_unit_hinge_axis_raises(self, metadata, vehicle_config):
        """Non-unit hinge axis should raise AssetValidationError."""
        bad_metadata = copy.deepcopy(metadata)
        bad_metadata["hinge_axes"][0] = [2.0, 0.0, 0.0]  # Not unit
        with pytest.raises(AssetValidationError, match="unit vector"):
            validate_asset(bad_metadata, vehicle_config)

    def test_missing_body_link_name_raises(self, metadata, vehicle_config):
        """Metadata without body_link_name should raise AssetValidationError."""
        bad_metadata = copy.deepcopy(metadata)
        del bad_metadata["body_link_name"]
        with pytest.raises(AssetValidationError, match="Missing required keys"):
            validate_asset(bad_metadata, vehicle_config)

    def test_correct_fin_count(self, metadata, vehicle_config):
        """Metadata should specify exactly 4 fins."""
        assert len(metadata["fin_link_names"]) == 4
        assert len(metadata["fin_joint_names"]) == 4
        assert len(metadata["hinge_axes"]) == 4
        assert len(metadata["fin_cop_positions"]) == 4


# NOTE: Isaac Sim tests below require the sim runtime.
# They will be skipped automatically if Isaac Sim is not available.
try:
    import omni.usd  # noqa: F401
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False


@pytest.mark.skipif(not ISAAC_AVAILABLE, reason="Isaac Sim runtime not available")
class TestAssetValidationWithIsaac:
    """Full USD structural validation tests (require Isaac Sim runtime)."""

    def test_valid_usd_passes_all_checks(self, metadata, vehicle_config):
        """Valid USD scene should pass all structural checks."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        diagnostics = validate_asset(metadata, vehicle_config, stage=stage)
        # Should not raise; any diagnostics are warnings
        assert isinstance(diagnostics, list)

    def test_missing_fin_link_raises(self, metadata, vehicle_config):
        """USD scene missing a fin link prim should raise AssetValidationError."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        bad_metadata = copy.deepcopy(metadata)
        bad_metadata["fin_link_names"][0] = "nonexistent_fin"
        with pytest.raises(AssetValidationError, match="not found in USD"):
            validate_asset(bad_metadata, vehicle_config, stage=stage)
