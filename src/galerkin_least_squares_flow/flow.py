"""Fit a polynomial reduced field and integrate its local flow."""

from math import comb

import numpy as np
import torch
from ngfield import FunctionalFlow, GalerkinField, integrate_field
from scipy.linalg.lapack import get_lapack_funcs
from torch.nn.functional import linear

from ._polynomials import _BallPolynomials
from ._sampling import sample_unit_ball
from ._validation import compute_device, floating_dtype, positive_integer, positive_real


def _least_squares(blocks):
    """Compress each block with QR before solving the remaining small system."""
    triangular = transformed = None
    for features, targets in blocks:
        if (
            triangular is not None
            and triangular.device.type == "cpu"
            and triangular.shape[0] == triangular.shape[1] >= 512
        ):
            merged = _merge_cpu_qr(triangular, transformed, features, targets)
            if merged is not None:
                triangular, transformed = merged
                continue
        if triangular is not None:
            features = torch.cat((triangular, features), dim=0)
            targets = torch.cat((transformed, targets), dim=0)
        reflectors, factors = torch.geqrf(features)
        count = len(factors)
        triangular = reflectors[:count].triu()
        transformed = torch.ormqr(reflectors, factors, targets, transpose=True)[:count]
    solution = _solve_reduced(triangular, transformed)
    if not bool(torch.isfinite(solution).all()):
        raise FloatingPointError("The least-squares coefficients must be finite.")
    return solution


def _merge_cpu_qr(triangular, transformed, features, targets):
    """Update a square triangular factor without refactorizing its zero entries."""
    try:
        factor, apply = get_lapack_funcs(("tpqrt", "tpmqrt"), (triangular.numpy(),))
    except ValueError:
        return None
    upper = np.asfortranarray(triangular.numpy())
    top = np.asfortranarray(transformed.numpy())
    bottom = np.array(features.numpy(), order="F", copy=True)
    rhs = np.array(targets.numpy(), order="F", copy=True)
    upper, reflectors, factors, info = factor(0, 64, upper, bottom, overwrite_a=1, overwrite_b=1)
    if info:
        raise RuntimeError(f"The block QR update failed with LAPACK status {info}.")
    top, _, info = apply(0, reflectors, factors, top, rhs, trans="T", overwrite_a=1, overwrite_b=1)
    if info:
        raise RuntimeError(f"The block QR transform failed with LAPACK status {info}.")
    return torch.from_numpy(upper), torch.from_numpy(top)


def _norm_bound(absolute):
    return absolute.sum(dim=0).amax().sqrt() * absolute.sum(dim=1).amax().sqrt()


def _verified_inverse_bound(triangular):
    # If ||I - R X|| < 1, then R^-1 = X (R X)^-1 and the Neumann bound applies.
    identity = torch.eye(len(triangular), dtype=triangular.dtype, device=triangular.device)
    inverse = torch.linalg.solve_triangular(triangular, identity, upper=True)
    residual = _norm_bound((identity - triangular @ inverse).abs())
    return _norm_bound(inverse.abs()) / torch.where(residual < 0.5, 1 - residual, torch.nan)


def _solve_reduced(triangular, targets):
    # The triangular comparison matrix bounds |R^-1| entrywise by its inverse.
    # Row and column sums give a conservative lower bound for 1/cond_2(R).
    absolute = triangular.abs()
    comparison = -absolute
    comparison.diagonal().neg_()
    ones = torch.ones_like(triangular[:, :1])
    inverse_inf = torch.linalg.solve_triangular(comparison, ones, upper=True).abs().amax()
    inverse_one = torch.linalg.solve_triangular(comparison.T, ones, upper=False).abs().amax()
    norm = _norm_bound(absolute)
    condition_bound = norm * inverse_inf.sqrt() * inverse_one.sqrt()
    margin = 32 * torch.finfo(triangular.dtype).eps * len(triangular)
    if bool(condition_bound * margin < 1):
        return torch.linalg.solve_triangular(triangular, targets, upper=True)
    if triangular.device.type == "cpu" and len(triangular) >= 512:
        condition_bound = norm * _verified_inverse_bound(triangular)
        if bool(condition_bound * margin < 1):
            return torch.linalg.solve_triangular(triangular, targets, upper=True)
    result = torch.linalg.lstsq(triangular.cpu(), targets.cpu(), driver="gelsd")
    if result.rank.item() < triangular.shape[-1]:
        raise ValueError("The sampled polynomial basis is rank deficient. Increase samples.")
    return result.solution.to(triangular.device)


class _PolynomialField:
    def __init__(self, features, coefficients, radius):
        self.features = features
        self.dimension = features.dimension
        self.radius = radius
        self.device = coefficients.device
        self.dtype = coefficients.dtype
        self._bias, self._weight = features.monomial_coefficients(coefficients)
        self._orthogonal_coefficients = None
        if not bool(torch.isfinite(self._bias).all() & torch.isfinite(self._weight).all()):
            self._orthogonal_coefficients = coefficients
            self._bias = self._weight = None

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
        return self.radius * self._unit_velocity(states / self.radius)

    def _quadratics(self, left, right):
        first, second = self.features._pairs
        mixed = left.index_select(-1, first) * right.index_select(-1, second)
        if left is not right:
            mixed = (mixed + right.index_select(-1, first) * left.index_select(-1, second)) * 0.5
        return torch.cat((mixed, left * right), dim=-1)

    def _unit_velocity(self, state):
        if self._orthogonal_coefficients is not None:
            return self.features(state) @ self._orthogonal_coefficients
        features = state
        if self.features.degree == 2:
            features = torch.cat((state, self._quadratics(state, state)), dim=-1)
        return linear(features, self._weight, self._bias)

    def _taylor_jets(self, states, highest):
        """Compute derivatives in time by the product rule for the fitted polynomial."""
        if self._orthogonal_coefficients is None:
            return self._polynomial_jets(states, highest)
        return None

    def _polynomial_jets(self, states, highest):
        self._states(states)
        unit = states / self.radius
        velocity = self._unit_velocity(unit)
        yield self.radius * velocity
        if self.features.degree == 1:
            for _ in range(1, highest):
                velocity = linear(velocity, self._weight)
                yield self.radius * velocity
            return

        derivatives = [unit, velocity]
        for degree in range(1, highest):
            # The Leibniz rule pairs derivatives with indices summing to degree.
            quadratic = 2 * self._quadratics(unit, derivatives[-1])
            for index in range(1, degree // 2 + 1):
                other = degree - index
                multiplicity = comb(degree, index) * (1 if other == index else 2)
                quadratic = quadratic + multiplicity * self._quadratics(
                    derivatives[index], derivatives[other]
                )
            velocity = linear(torch.cat((derivatives[-1], quadratic), dim=-1), self._weight)
            derivatives.append(velocity)
            yield self.radius * velocity


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
        typed_radius = torch.tensor(self.radius, dtype=dtype, device="cpu")
        if not bool(torch.isfinite(typed_radius) & (typed_radius > 0)):
            raise ValueError("radius must be representable as a positive finite number in dtype.")
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
