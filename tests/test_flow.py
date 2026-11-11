from math import sqrt

import pytest
import torch
from ngfield import GalerkinField

from galerkin_least_squares_flow import Geometry, Space, System
from galerkin_least_squares_flow._polynomials import _BallPolynomials
from galerkin_least_squares_flow.flow import _least_squares


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
