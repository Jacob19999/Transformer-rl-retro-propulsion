"""
Unit tests for fin spatial layout and COP position computation.

Tests COP positions for each fin, fin-local-to-body transforms, and
fin ordering consistency.
"""

import pytest
import torch
from pathlib import Path
from tvc_env.dynamics.fin_geometry import (
    load_cop_positions,
    load_hinge_axes,
    compute_fin_body_transforms,
    validate_fin_ordering,
    FIN_LABELS,
    NUM_FINS,
)
from tvc_env.asset.usd_loader import load_asset_metadata

METADATA_PATH = Path(__file__).parents[2] / "assets/metadata/edf_drone_v2.asset.yaml"


@pytest.fixture
def metadata():
    return load_asset_metadata(METADATA_PATH)


class TestCopPositions:
    def test_load_returns_four_positions(self, metadata):
        """Should return exactly 4 COP positions."""
        cops = load_cop_positions(metadata)
        assert cops.shape == (4, 3)

    def test_cop_positions_are_finite(self, metadata):
        """All COP positions should be finite (no NaN/inf)."""
        cops = load_cop_positions(metadata)
        assert torch.all(torch.isfinite(cops))

    def test_fin_ordering_px_positive_x(self, metadata):
        """First fin (+X) should have positive x COP component."""
        cops = load_cop_positions(metadata)
        assert cops[0, 0] > 0.0, f"+X fin COP x-component should be positive, got {cops[0, 0]}"

    def test_fin_ordering_py_positive_y(self, metadata):
        """Second fin (+Y) should have positive y COP component."""
        cops = load_cop_positions(metadata)
        assert cops[1, 1] > 0.0, f"+Y fin COP y-component should be positive, got {cops[1, 1]}"

    def test_fin_ordering_mx_negative_x(self, metadata):
        """Third fin (-X) should have negative x COP component."""
        cops = load_cop_positions(metadata)
        assert cops[2, 0] < 0.0, f"-X fin COP x-component should be negative, got {cops[2, 0]}"

    def test_fin_ordering_my_negative_y(self, metadata):
        """Fourth fin (-Y) should have negative y COP component."""
        cops = load_cop_positions(metadata)
        assert cops[3, 1] < 0.0, f"-Y fin COP y-component should be negative, got {cops[3, 1]}"


class TestHingeAxes:
    def test_load_returns_four_axes(self, metadata):
        """Should return exactly 4 hinge axes."""
        axes = load_hinge_axes(metadata)
        assert axes.shape == (4, 3)

    def test_all_axes_are_unit_vectors(self, metadata):
        """All hinge axes should be unit vectors."""
        axes = load_hinge_axes(metadata)
        norms = axes.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Norms: {norms}"


class TestFinBodyTransforms:
    def test_zero_deflection_identity_quat(self, metadata):
        """At zero deflection, fin-local-to-body transform should be identity quaternion."""
        axes = load_hinge_axes(metadata)
        deflections = torch.zeros(1, 4)
        quats = compute_fin_body_transforms(axes, deflections)
        assert quats.shape == (1, 4, 4)
        # Identity quaternion: (1, 0, 0, 0)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for i in range(4):
            assert torch.allclose(quats[0, i], expected, atol=1e-5), \
                f"Fin {i} at zero deflection: got {quats[0, i]}"

    def test_batch_shape(self, metadata):
        """Transform computation should handle batch dimension correctly."""
        axes = load_hinge_axes(metadata)
        deflections = torch.randn(32, 4) * 0.1
        quats = compute_fin_body_transforms(axes, deflections)
        assert quats.shape == (32, 4, 4)

    def test_quaternion_is_unit(self, metadata):
        """All output quaternions should have unit norm."""
        axes = load_hinge_axes(metadata)
        deflections = torch.randn(16, 4) * 0.2
        quats = compute_fin_body_transforms(axes, deflections)
        norms = quats.norm(dim=-1)  # (16, 4)
        assert torch.allclose(norms, torch.ones(16, 4), atol=1e-5)


class TestFinOrdering:
    def test_valid_metadata_passes(self, metadata):
        """Valid metadata should pass ordering check without raising."""
        validate_fin_ordering(metadata)  # Should not raise

    def test_wrong_ordering_raises(self):
        """Incorrect fin ordering should raise ValueError."""
        bad_metadata = {
            "fin_cop_positions": [
                [-0.04, 0.0, 0.10],  # Wrong: -X where +X should be
                [0.0, 0.04, 0.10],
                [0.04, 0.0, 0.10],
                [0.0, -0.04, 0.10],
            ]
        }
        with pytest.raises(ValueError, match="ordering"):
            validate_fin_ordering(bad_metadata)

    def test_label_count(self):
        """Should have exactly 4 canonical fin labels."""
        assert len(FIN_LABELS) == NUM_FINS == 4
