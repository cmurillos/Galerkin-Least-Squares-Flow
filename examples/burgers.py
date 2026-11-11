"""Fit and evolve a reduced viscous Burgers field."""

import argparse

import torch
from ngfield import GalerkinField

from galerkin_least_squares_flow import Geometry, Space, System, ZeroTrace, grad, inner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(4, 16, 32), default=16)
    dimension = parser.parse_args().dimension
    cells = max(16, 2 * dimension)
    vertices = [[i / cells] for i in range(cells + 1)]
    simplices = [[i, i + 1] for i in range(cells)]
    geometry = Geometry(vertices=vertices, simplices=simplices)
    space = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = space.basis("laplacian", size=dimension)

    def weak(u, v, dx, ds):
        return (-0.01 * inner(grad(u[0]), grad(v[0])) + 0.5 * u[0] ** 2 * grad(v[0])[0]) * dx

    problem = System(basis=basis, weak=weak, radius=1.0, quadrature=6)
    features = (dimension + 1) * (dimension + 2) // 2
    flow = problem.fit(
        degree=2, samples=max(512, 4 * features), batch_size=max(128, features), seed=7
    )
    initial = flow.state(lambda x: 0.2 * torch.sin(torch.pi * x[:, :1]))
    times = torch.linspace(0, 0.2, 11, dtype=flow.dtype, device=flow.device)
    path = flow.evolve(initial, times, step=0.01)

    reference = GalerkinField(basis=basis, weak=weak, quadrature=6, dtype=flow.dtype)
    z0 = initial.coefficients()
    reference_path = reference.solve(z0, times, step=0.01, radius=flow.radius)
    error = torch.linalg.vector_norm(path.coefficients() - reference_path, dim=-1).max()
    generator = torch.Generator().manual_seed(17)
    held_out = torch.randn(16, dimension, dtype=flow.dtype, generator=generator)
    held_out = 0.35 * held_out / torch.linalg.vector_norm(held_out, dim=-1, keepdim=True)
    print(f"Reduced dimension: {dimension}")
    print(f"Polynomial features per component: {flow.feature_count}")
    print(f"Held-out field RMS error: {flow.field_rmse(held_out):.3e}")
    print(f"Maximum trajectory error: {error:.3e}")


if __name__ == "__main__":
    main()
