# Galerkin Least-Squares Flow

Keep the operational Galerkin basis and weak form in NGF. GLSF fits the
normalized velocity G(R x)/R on the open unit ball and evaluates the fitted
physical velocity as R f(z/R). NGF owns projection, reconstruction, and
Taylor integration, including numerical detection of domain exit.

The current fitted spaces are complete polynomials of degree one or two,
with an orthonormal basis for normalized volume on the unit ball. Preserve
arbitrary leading batch axes and gradients of fitted velocities with respect
to states. Keep the sampled regression as blockwise least squares and avoid
forming normal equations.

Run the following from the repository root after changes.

```bash
python -m pytest
python examples/burgers.py
ruff check .
ruff format --check .
python -m build
```
