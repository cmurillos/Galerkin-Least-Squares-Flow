# SPDX-License-Identifier: BSD-3-Clause
"""Uniform-volume state sampling inherited from Galerkin Neural Semigroup."""

import torch


def sample_unit_ball(count, dimension, *, generator, device, dtype, radius=1.0):
    """Sample normalized Lebesgue volume on the open unit ball."""
    states = torch.empty(count, dimension, device=device, dtype=dtype)
    pending = torch.arange(count, device=device)
    while len(pending):
        directions = torch.randn(
            len(pending), dimension, generator=generator, device=device, dtype=dtype
        )
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        directions = directions / norms.clamp_min(torch.finfo(dtype).tiny)
        radii = torch.rand(len(pending), 1, generator=generator, device=device, dtype=dtype)
        candidates = directions * radii.pow(1.0 / dimension)
        valid = (
            (norms[:, 0] > 0)
            & (torch.linalg.vector_norm(candidates, dim=-1) < 1)
            & (torch.linalg.vector_norm(radius * candidates, dim=-1) < radius)
        )
        states[pending[valid]] = candidates[valid]
        pending = pending[~valid]
    return states
