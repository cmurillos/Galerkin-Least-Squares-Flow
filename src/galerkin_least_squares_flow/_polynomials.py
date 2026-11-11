"""Polynomial basis orthonormal for normalized volume on the unit ball."""

from math import sqrt

import torch

from ._validation import positive_integer


class _BallPolynomials:
    def __init__(self, dimension, degree, *, device, dtype):
        self.dimension = dimension
        self.degree = positive_integer(degree, "degree")
        if self.degree not in (1, 2):
            raise ValueError("degree must be 1 or 2.")
        self.size = dimension + 1
        self._linear_scale = sqrt(dimension + 2)
        if self.degree == 2:
            self.size += dimension * (dimension + 1) // 2
            self._pairs = torch.triu_indices(dimension, dimension, offset=1, device=device)
            self._square_index = torch.arange(1, dimension, device=device, dtype=dtype)
            fourth_moment = (dimension + 2) * (dimension + 4)
            self._mixed_scale = sqrt(fourth_moment)
            self._contrast_scale = torch.sqrt(
                fourth_moment / (2 * self._square_index * (self._square_index + 1))
            )
            self._radial_center = dimension / (dimension + 2)
            self._radial_scale = (dimension + 2) * sqrt((dimension + 4) / dimension) / 2

    def __call__(self, states):
        constant = torch.ones_like(states[..., :1])
        linear = self._linear_scale * states
        if self.degree == 1:
            return torch.cat((constant, linear), dim=-1)

        mixed = self._mixed_scale * states[..., self._pairs[0]] * states[..., self._pairs[1]]
        squares = states.square()
        # The diagonal quadratics split into trace-free contrasts and a radial term.
        contrasts = self._contrast_scale * (
            squares[..., :-1].cumsum(dim=-1) - self._square_index * squares[..., 1:]
        )
        radial = self._radial_scale * (squares.sum(dim=-1, keepdim=True) - self._radial_center)
        return torch.cat((constant, linear, mixed, contrasts, radial), dim=-1)
