"""Galerkin-supervised local flows fitted by polynomial least squares."""

from ngfield import Geometry, Solution, Space, State, ZeroTrace, grad, inner

from .flow import Flow, System

__version__ = "0.1.0"
__all__ = ["Flow", "Geometry", "Solution", "Space", "State", "System", "ZeroTrace", "grad", "inner"]
