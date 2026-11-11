from math import sqrt

import pytest
import torch
from ngfield import DomainExitError, GalerkinField, integrate_field
from ngfield.evolution import _autodiff_taylor_jets

from galerkin_least_squares_flow import Geometry, Space, System
from galerkin_least_squares_flow._polynomials import _BallPolynomials
from galerkin_least_squares_flow.flow import _least_squares, _PolynomialField


def test_quadratic_features_are_orthonormal_on_the_disk():
    dtype = torch.float64
    angles = torch.arange(8, dtype=dtype) * (torch.pi / 4)
    directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
    squared_radii = torch.tensor([1 - sqrt(3 / 5), 1, 1 + sqrt(3 / 5)], dtype=dtype) / 2
    points = (squared_radii.sqrt()[:, None, None] * directions).reshape(-1, 2)
    weights = torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=dtype).repeat_interleave(8) / 8
    basis = _BallPolynomials(2, 2, device="cpu", dtype=dtype)
    values = basis(points)
    torch.testing.assert_close(values.T @ (weights[:, None] * values), torch.eye(6, dtype=dtype))


def test_blockwise_least_squares_matches_one_dense_fit():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(41, 9, generator=generator, dtype=torch.float64)
    targets = torch.randn(41, 3, generator=generator, dtype=torch.float64)
    fitted = _least_squares(zip(features.split(5), targets.split(5)))
    expected = torch.linalg.lstsq(features, targets, driver="gelsd").solution
    torch.testing.assert_close(fitted, expected, atol=1e-12, rtol=1e-12)


def test_blockwise_fit_rejects_a_rank_deficient_design():
    features = torch.ones(12, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="rank deficient"):
        _least_squares([(features, features[:, :1])])


def test_large_cpu_blocks_preserve_the_dense_least_squares_solution():
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(645, 512, generator=generator, dtype=torch.float64)
    targets = torch.randn(645, 4, generator=generator, dtype=torch.float64)
    fitted = _least_squares(zip(features.split(520), targets.split(520)))
    expected = torch.linalg.lstsq(features, targets, driver="gelsd").solution
    torch.testing.assert_close(fitted, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("degree", [1, 2])
def test_monomial_velocity_and_taylor_jets_match_original_field(degree):
    generator = torch.Generator().manual_seed(27)
    features = _BallPolynomials(3, degree, device="cpu", dtype=torch.float64)
    coefficients = torch.randn(features.size, 3, dtype=torch.float64, generator=generator)
    field = _PolynomialField(features, coefficients, radius=1.7)
    states = torch.randn(2, 2, 3, dtype=torch.float64, generator=generator) * 0.12
    states.requires_grad_()

    def original(z):
        return field.radius * (features(z / field.radius) @ coefficients)

    torch.testing.assert_close(field(states), original(states))
    actual = tuple(field._taylor_jets(states, 5))
    expected = tuple(_autodiff_taylor_jets(original, states, 5))
    for computed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(computed, reference, atol=2e-10, rtol=2e-12)
    actual_grad = torch.autograd.grad(actual[-1].sum(), states)[0]
    expected_grad = torch.autograd.grad(expected[-1].sum(), states)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_specialized_taylor_path_preserves_boundary_contact():
    features = _BallPolynomials(1, 1, device="cpu", dtype=torch.float64)
    field = _PolynomialField(features, torch.tensor([[1.0], [0.0]], dtype=torch.float64), 1.0)
    initial = torch.tensor([0.9], dtype=torch.float64)
    times = torch.tensor([0.0, 0.05, 0.2], dtype=torch.float64)
    with pytest.raises(DomainExitError) as caught:
        integrate_field(field, initial, times, step=0.2, radius=field.radius)
    assert caught.value.exit_time == pytest.approx(0.1)
    torch.testing.assert_close(caught.value.completed_times, times[:2])


@pytest.mark.parametrize("degree, power", [(1, 1), (2, 2)])
def test_fit_matches_galerkin_field_and_flow(degree, power):
    geometry = Geometry(vertices=[[0.0], [0.5], [1.0]], simplices=[[0, 1], [1, 2]])
    basis = Space(geometry=geometry, components=1).basis("polynomial", size=2)

    def weak(u, v, dx, ds):
        return -(u[0] ** power) * v[0] * dx

    problem = System(basis=basis, weak=weak, radius=2.5, quadrature=6)
    flow = problem.fit(degree=degree, samples=40, batch_size=7, seed=12)
    reference = GalerkinField(basis=basis, weak=weak, quadrature=6)
    z = torch.tensor([[[0.2, -0.1], [-0.4, 0.3]]], dtype=flow.dtype)
    torch.testing.assert_close(flow.velocity(z), reference(z), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.func.jacrev(flow.velocity)(z[0, 0]),
        torch.func.jacrev(reference)(z[0, 0]),
        atol=2e-12,
        rtol=2e-12,
    )
    times = torch.tensor([0.0, 0.03, 0.07], dtype=flow.dtype)
    path = flow.evolve(flow.from_coefficients(z), times, step=0.02)
    expected = reference.solve(z, times, step=0.02, radius=flow.radius)
    torch.testing.assert_close(path.coefficients(), expected, atol=2e-12, rtol=2e-12)
    assert flow.field_rmse(z).item() < 2e-12
    with pytest.raises(ValueError, match="open ball"):
        flow.velocity(torch.tensor([2.5, 0.0], dtype=flow.dtype))
    with pytest.raises(ValueError, match="samples"):
        problem.fit(degree=degree, samples=2)
