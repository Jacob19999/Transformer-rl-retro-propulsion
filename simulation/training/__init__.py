"""Training subpackage (Gym env, controllers, scripts)."""

__all__: list[str] = []

# Keep package import lightweight: Gymnasium may not be installed in all dev envs.
try:  # pragma: no cover
    from simulation.training.edf_landing_env import EDFLandingEnv

    __all__.append("EDFLandingEnv")
except ImportError:
    # Allow importing `simulation.training` without Gymnasium; users can install
    # training deps when needed (requirements.txt includes gymnasium>=0.29).
    pass

