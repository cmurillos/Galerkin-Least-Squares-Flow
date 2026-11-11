"""Fit a polynomial reduced field and integrate its local flow."""

import torch
from ngfield import FunctionalFlow, GalerkinField, integrate_field

from ._polynomials import _BallPolynomials
from ._sampling import sample_unit_ball
from ._validation import compute_device, floating_dtype, positive_integer, positive_real


def _least_squares(blocks):
    """Compress each block with QR before solving the remaining small system."""
    triangular = transformed = None
    for features, targets in blocks:
        if triangular is not None:
            features = torch.cat((triangular, features), dim=0)
            targets = torch.cat((transformed, targets), dim=0)
        orthogonal, triangular = torch.linalg.qr(features, mode="reduced")
        transformed = orthogonal.mT @ targets
    result = torch.linalg.lstsq(triangular.cpu(), transformed.cpu(), driver="gelsd")
    if result.rank.item() < triangular.shape[-1]:
        raise ValueError("The sampled polynomial basis is rank deficient. Increase samples.")
    if not bool(torch.isfinite(result.solution).all()):
        raise FloatingPointError("The least-squares coefficients must be finite.")
    return result.solution.to(triangular.device)


class _PolynomialField:
    def __init__(self, features, coefficients, radius):
        self.features = features
        self.coefficients = coefficients
        self.dimension = features.dimension
        self.radius = radius
        self.device = coefficients.device
        self.dtype = coefficients.dtype

    def _states(self, states):
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch tensor.")
        if states.ndim < 1 or states.shape[-1] != self.dimension:
            raise ValueError(f"states must have shape [...,{self.dimension}].")
        if states.dtype != self.dtype or states.device != self.device:
            raise ValueError("states and field must share dtype and device.")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("states must be finite.")
        if not bool((torch.linalg.vector_norm(states, dim=-1) < self.radius).all()):
            raise ValueError("states must lie strictly inside the open ball.")

    def __call__(self, states):
        self._states(states)
        return self.radius * (self.features(states / self.radius) @ self.coefficients)


class Flow(FunctionalFlow):
    """Fitted field in Galerkin coordinates and its local function-valued flow."""

    @property
    def feature_count(self):
        return self._dynamics.features.size

    @property
    def degree(self):
        return self._dynamics.features.degree

    def velocity(self, states):
        """Return the fitted velocity at reduced coordinates of shape [...,N]."""
        return self._dynamics(states)

    def solve(self, initial, times, *, step=None, tolerance=None, order=4):
        """Integrate reduced coordinates with the shared NGF Taylor method."""
        return integrate_field(
            self._dynamics,
            initial,
            times,
            step=step,
            tolerance=tolerance,
            order=order,
            radius=self.radius,
        )

    @torch.no_grad()
    def field_rmse(self, states):
        """Return the RMS Euclidean field error on held-out reduced coordinates."""
        predicted = self.velocity(states)
        if states.numel() == 0:
            raise ValueError("Provide at least one held-out state.")
        return ((predicted - self._field(states)).square().sum(dim=-1).mean()).sqrt()


class System:
    """Fix a Galerkin problem and the radius used for fitting and local evolution."""

    def __init__(self, *, basis, weak, radius, quadrature=None):
        self.basis = basis
        self.weak = weak
        self.radius = positive_real(radius, "radius")
        self.quadrature = quadrature

    @torch.no_grad()
    def fit(self, *, degree=2, samples=10_000, batch_size=4096, seed=0, device="cpu", dtype=None):
        """Fit NGF velocities at sampled states by blockwise least squares."""
        samples = positive_integer(samples, "samples")
        batch_size = positive_integer(batch_size, "batch_size")
        seed = positive_integer(seed, "seed", minimum=0)
        device, dtype = compute_device(device), floating_dtype(dtype)
        reference = GalerkinField(
            basis=self.basis, weak=self.weak, quadrature=self.quadrature, device=device, dtype=dtype
        )
        features = _BallPolynomials(
            reference.dimension, degree, device=reference.device, dtype=dtype
        )
        if samples < features.size:
            raise ValueError(f"Use at least {features.size} samples for this polynomial basis.")
        generator = torch.Generator(device=reference.device).manual_seed(seed)

        def blocks():
            for start in range(0, samples, batch_size):
                unit_states = sample_unit_ball(
                    min(batch_size, samples - start),
                    reference.dimension,
                    generator=generator,
                    device=reference.device,
                    dtype=dtype,
                    radius=self.radius,
                )
                targets = reference(self.radius * unit_states) / self.radius
                if not bool(torch.isfinite(targets).all()):
                    raise FloatingPointError("The reference field returned nonfinite targets.")
                yield features(unit_states), targets

        field = _PolynomialField(features, _least_squares(blocks()), self.radius)
        return Flow(reference, field, radius=self.radius)
