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

        mixed = (self._mixed_scale * states).index_select(-1, self._pairs[0]) * states.index_select(
            -1, self._pairs[1]
        )
        squares = states.square()
        # The diagonal quadratics split into trace-free contrasts and a radial term.
        contrasts = self._contrast_scale * (
            squares[..., :-1].cumsum(dim=-1) - self._square_index * squares[..., 1:]
        )
        radial = self._radial_scale * (squares.sum(dim=-1, keepdim=True) - self._radial_center)
        return torch.cat((constant, linear, mixed, contrasts, radial), dim=-1)

    def monomial_coefficients(self, coefficients):
        """Convert fitted coefficients once, preserving the polynomial in unit coordinates."""
        bias = coefficients[0]
        linear = self._linear_scale * coefficients[1 : self.dimension + 1]
        if self.degree == 1:
            return bias.contiguous(), linear.T.contiguous()

        mixed_end = self.dimension + 1 + self._pairs.shape[-1]
        mixed = self._mixed_scale * coefficients[self.dimension + 1 : mixed_end]
        contrast = self._contrast_scale[:, None] * coefficients[mixed_end:-1]
        radial = self._radial_scale * coefficients[-1]
        zero = torch.zeros_like(bias)[None]
        diagonal = (
            radial
            + torch.cat((contrast.flip(0).cumsum(0).flip(0), zero))
            - torch.cat((zero, self._square_index[:, None] * contrast))
        )
        bias = bias - self._radial_center * radial
        return bias.contiguous(), torch.cat((linear, mixed, diagonal)).T.contiguous()
