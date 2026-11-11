# SPDX-License-Identifier: BSD-3-Clause
"""Uniform-volume state sampling inherited from Galerkin Neural Semigroup."""

import torch


def sample_unit_ball(count, dimension, *, generator, device, dtype, radius=1.0):
    """Sample normalized Lebesgue volume on the open unit ball."""
    pending = None
    while True:
        size = count if pending is None else len(pending)
        directions = torch.randn(size, dimension, generator=generator, device=device, dtype=dtype)
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        directions = directions / norms.clamp_min(torch.finfo(dtype).tiny)
        radii = torch.rand(size, 1, generator=generator, device=device, dtype=dtype)
        candidates = directions * radii.pow(1.0 / dimension)
        valid = (
            (norms[:, 0] > 0)
            & (torch.linalg.vector_norm(candidates, dim=-1) < 1)
            & (torch.linalg.vector_norm(radius * candidates, dim=-1) < radius)
        )
        invalid = (~valid).nonzero(as_tuple=True)[0]
        if pending is None:
            states = candidates
            pending = invalid
        else:
            states[pending] = candidates
            pending = pending.index_select(0, invalid)
        if not len(pending):
            return states
