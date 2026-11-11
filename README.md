# Galerkin Least-Squares Flow

GLSF fits a reduced Galerkin vector field from evaluations at sampled states and
integrates the fitted field to obtain a local flow. Numerical Galerkin Field
(NGF) supplies the reference velocities, the physical basis and the Taylor
integrator.

## Installation

```bash
python -m pip install -e '.[dev]'
python examples/burgers.py
python examples/burgers.py --dimension 32
python -m pytest
```

The [Burgers example](examples/burgers.py) builds a weak form, fits a quadratic
field, and compares its held-out velocities and trajectories with NGF.

## Method

For a reduced state \(z\) in the open ball of radius \(R\), let \(x=z/R\).
The reference target is \(G_{N,Q}(Rx)/R\). GLSF fits its components in a common
basis of polynomials of total degree at most one or two. Each scalar basis
function multiplied by a coordinate vector gives one vector-valued basis
function. The basis is orthonormal for normalized volume on the unit ball.
The sampled problem is solved by blockwise QR with a shared design for all
output components. The QR transforms the targets using Householder reflectors.
Large CPU blocks can update the existing triangular factor directly.
A triangular solve is used when a condition bound supports it, with a
rank-revealing solve for the other cases. The fitted velocity in the original
coordinates is scaled by \(R\). Training requires state and velocity pairs.

`System(basis=..., weak=..., radius=...)` fixes the Galerkin problem.
`System.fit(degree=2, samples=..., batch_size=..., seed=...)` returns a fitted
`Flow`. Use `velocity(z)` or `solve(z, times)` for reduced coordinates and
`state(u0)` with `evolve(state, times)` for physical functions.

The sample count must be at least the number of scalar basis functions.
Larger sample sets can be processed in blocks. For degree two, the number
of functions per output is \((N+1)(N+2)/2\). Blockwise QR stores the triangular
factor and the current block. Its cost still increases with the basis size.

The basis includes a constant term. The fitted field defines an autonomous
local flow within the open ball. Its derivatives in time are computed directly
from the fitted polynomial and supplied to the shared NGF Taylor integrator.
NGF controls step acceptance and reports numerical contact with the boundary.
CPU is the default device.

## Provenance

The unit-ball sampler and project layout are adapted from [Galerkin Neural
Semigroup](https://github.com/cmurillos/Galerkin-Neural-Semigroup/tree/225018362787e12a0d5bafbcd49c145682721991).
GLSF uses the [Apache 2.0 license](LICENSE). Adapted GNS material retains its
[BSD 3-Clause notice](LICENSE-GNS).
