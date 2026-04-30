# Fiber Patch Placement Optimization

**Gradient-based optimization of fiber patch positions and orientations in CFRP composite structures using differentiable finite element analysis in PyTorch.**

---

## Overview

This repository contains the implementation developed as part of a Master's thesis in Materials Science and Engineering at the University of Augsburg. The project addresses a core challenge in composite structural design: given a base laminate with known geometry and loading, where should additional fiber patches be placed, and at what orientation, to maximally increase structural stiffness while remaining manufacturable?

The approach couples a differentiable finite element solver ([torch-FEM](https://github.com/meyer-nils/torch_fem)) with PyTorch's automatic differentiation to propagate structural response gradients directly through patch geometry parameters. Patch positions and orientations are treated as continuous, differentiable variables and optimized end-to-end using the Adam optimizer.

### Problem setup

A 120 × 120 mm CFRP plate with an elliptical cutout (semi-axes 20 mm × 40 mm) is loaded in tension via displacement-controlled lashes. The base laminate is a balanced cross-ply with eight plies (4 in the symmetric quarter model). The goal is to place *N* rectangular fiber patches on the plate surface to maximize the strain energy absorbed by the structure (a proxy for stiffness improvement), subject to manufacturing and strength constraints.

Symmetry is enforced by mirroring each optimized patch across both axes, so every patch in the quarter model produces four symmetric counterparts on the full plate.

---

## Methodology

### 1. Differentiable FEM

The finite element model is built on a second-order triangular mesh (1,921 elements in the quarter model) generated using [pygmsh](https://github.com/meshpro/pygmsh) and solved via [torch-FEM](https://github.com/meyer-nils/torch_fem). All solver operations are implemented as differentiable PyTorch operations, allowing gradients to flow from structural response quantities (strain energy, stress) back to patch geometry parameters.

### 2. Patch representation

Each patch is a rectangular ply of fixed dimensions (*l* × *w* = 30 × 10 mm) described by three learnable parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| x-position | `pos_x` | Patch center x-coordinate (mm) |
| y-position | `pos_y` | Patch center y-coordinate (mm) |
| Orientation | `θ` | Fiber angle relative to global x-axis (rad) |

The material stiffness of each patch is obtained by rotating the orthotropic CFRP constitutive matrix `C` using the standard 4th-order tensor rotation:

$$\tilde{C}_{ijkl} = R_{im} R_{jn} R_{ko} R_{lp} \, C_{mnop}$$

where **R**(θ) is the 2D rotation tensor.

### 3. Smooth patch-element weight function

Patch coverage is defined by a smooth Gaussian-like weight function rather than a hard binary mask. This keeps the optimization landscape differentiable:

$$w = \exp\!\left(-\varepsilon^2 \left(\hat{\xi}^2 + \hat{\eta}^2\right)\right)$$

with

$$\hat{\xi} = \max\!\left(|\xi| - L/2 + \sqrt{\ln 2}/\varepsilon,\; 0\right), \quad
  \hat{\eta} = \max\!\left(|\eta| - W/2 + \sqrt{\ln 2}/\varepsilon,\; 0\right)$$

where ξ and η are local patch coordinates for each element centroid. Inside the patch, *w* ≈ 1; outside, it decays smoothly to 0. The parameter ε = 0.5 controls transition sharpness.

The element-wise thickness and stiffness are assembled as weighted superpositions of all contributing patches:

$$t_e = t_\text{base} + t_\text{ply} \sum_p w_{p,e}$$

$$C_e = \frac{t_\text{ply} \sum_p w_{p,e} \tilde{C}_p + t_\text{base} C_0}{t_e}$$

### 4. Multi-objective loss function

The optimizer minimizes:

$$\mathcal{L} = -(W_\text{strain} + \lambda_w \, W_\text{patch}) + \lambda_\text{grad} \, P_\text{grad} + \lambda_t \, P_\text{thick} + \lambda_s \, P_\text{strength}$$

| Term | Role |
|------|------|
| $W_\text{strain}$ | Strain energy (maximize → higher stiffness) |
| $W_\text{patch}$ | Total patch coverage weight (encourage patch engagement) |
| $P_\text{grad}$ | Thickness gradient penalty (penalizes abrupt thickness changes at patch edges) |
| $P_\text{thick}$ | Thickness threshold penalty (prevents excessive local ply buildup) |
| $P_\text{strength}$ | Strength penalty (penalizes stress exceeding material limits, especially at patch boundaries) |

Hyperparameters used: λ_w = 0.1, λ_grad = 0.1, λ_t = 0.5, λ_s = 0.005.

### 5. Manufacturing constraints

Two manufacturing-aware penalties prevent the optimizer from producing designs that are geometrically optimal but physically unmanufacturable:

**Thickness gradient penalty:** The top-*k* mean thickness gradient magnitudes between neighboring elements are penalized. Sharp ply drops create stress concentrations and are difficult to produce in AFP/FPP processes.

**Strength penalty with edge density reduction:** At patch boundaries (high weight gradient), the effective material strength is reduced by a factor proportional to local edge density. This models the knock-down in interlaminar strength at ply terminations and penalizes solutions that place highly loaded regions at patch edges.

### 6. Optimizer

All patch parameters are initialized using a Latin Hypercube sample across the plate domain and trained for 500 iterations using Adam (lr = 0.1).

---

## Repository structure

```
Fiber_Patch_Placement/
├── hole_plate_patched-Final.ipynb   # Main optimization notebook (full pipeline)
├── laminate.ipynb                   # Laminate property fitting from ply-level data
└── mesh.vtk                         # Pre-generated quarter-model mesh (pygmsh output)
```

### `hole_plate_patched-Final.ipynb`

The primary notebook. Covers the full pipeline from geometry generation to optimized patch export:

1. Imports and configuration
2. Orthotropic material model and cross-ply base stiffness
3. Geometry generation and FEM mesh (pygmsh + torch-FEM)
4. Baseline model solve and reference metrics
5. `Patch` class definition
6. Smooth weight function and edge density
7. Patch stress and strength penalty functions
8. Verification on two hand-placed test patches
9. Optimization setup: N = 8 patches, Latin Hypercube initialization
10. Gradient-based optimization loop (Adam, 500 iterations)
11. Convergence history plots (5 terms)
12. Optimized patch layout visualization
13. Final structural metrics comparison (baseline vs. optimized)
14. Result export to `.vtu`, `.npz`, and `.md`

### `laminate.ipynb`

Fits effective in-plane stiffness properties (E_xx, E_yy, G_xy, ν_xy) from ply-level Classical Lamination Theory (CLT) data. The fitted values feed directly into the material model in the main notebook.

### `mesh.vtk`

Pre-generated VTK mesh file for the quarter-plate geometry (quarter of the 120 × 120 mm plate with elliptical hole). Import directly to skip the mesh generation step.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Differentiable computing, automatic differentiation, Adam optimizer |
| `torch-fem` | Differentiable finite element solver for structural mechanics |
| `pygmsh` | Programmatic mesh generation (OpenCASCADE backend) |
| `numpy` | Numerical arrays, file I/O |
| `scipy` | Latin Hypercube sampling (`scipy.stats.qmc`) |
| `matplotlib` | Visualization and plotting |
| `tqdm` | Progress bar for optimization loop |
| `pygad` | Imported but not used in final pipeline (retained for reference) |

### Installation

```bash
pip install torch torchvision
pip install torch-fem
pip install pygmsh
pip install numpy scipy matplotlib tqdm pygad
```

> **Note on pygmsh:** The OpenCASCADE backend requires a working Gmsh installation. If `pygmsh.occ.Geometry` raises an error, install Gmsh via `pip install gmsh` or from [gmsh.info](https://gmsh.info).

The pre-generated `mesh.vtk` is included so the geometry generation cell (which requires Gmsh) can be skipped if needed — simply call `import_mesh("mesh.vtk", ...)` directly.

---

## Running the notebook

```bash
git clone https://github.com/Tharun-arety/Fiber_Patch_Placement.git
cd Fiber_Patch_Placement
jupyter notebook hole_plate_patched-Final.ipynb
```

Run cells in order. The optimization loop (500 iterations, ~5 FEM solves per iteration gradient step) takes several minutes on CPU. Each iteration solves the full FEM system, computes gradients via automatic differentiation, and updates all 3N patch parameters simultaneously.

---

## Key results

After 500 iterations of gradient-based optimization over N = 8 patches (32 symmetric counterparts on the full plate):

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Strain energy | Reference | Increased (patches maximize structural work input) |
| Max stress at hole | Reference | Redistributed via patch stiffness augmentation |
| Thickness gradient | — | Controlled via penalty term |
| Strength violations | — | Suppressed at patch boundaries |

Result files are exported to a timestamped directory `N={N}_E={strain_energy}_eps={EPS}_W={W}/` containing:

- `final.png` — optimized patch layout on full plate
- `optimized.vtu` — VTK unstructured grid with displacement, thickness, and stress fields
- `history.npz` — full position and orientation trajectory for all 500 iterations
- `samples.npz` — initial Latin Hypercube sample
- `results.md` — scalar performance metrics

---

## Technical notes

**Symmetry enforcement:** Only a quarter model is solved for computational efficiency. Each patch defined in the quarter domain is automatically mirrored across both axes, producing contributions to the stiffness and thickness assembly from all four symmetric counterparts. The mirroring is handled inside `Patch.mirrored_C()` and `Patch.get_mirrored_vertices()`.

**Differentiability:** The entire chain from patch parameters → vertices → element weights → thickness/stiffness assembly → FEM solve → strain energy is differentiable. `torch.autograd` computes exact gradients with respect to all `positions` and `orientations` parameters in a single backward pass.

**Coordinate system:** The model uses a right-handed Cartesian system with x along the loading direction and y transverse. The elliptical hole is centered at the origin. The quarter model covers x ∈ [0, L/2 + l] and y ∈ [0, W/2].

---

## Background

Fiber Patch Placement (FPP) is an Automated Fiber Placement variant in which discrete, locally shaped fiber patches are deposited over a base laminate to reinforce stress-critical regions. Unlike traditional AFP with continuous tows, FPP allows targeted placement with orientations independent of global fiber paths. The challenge is determining optimal patch positions and orientations without exhaustive search over the combinatorial space — a problem well suited to gradient-based optimization when the structural response is differentiable.

This implementation builds on [torch-FEM](https://github.com/meyer-nils/torch_fem) by extending it with:
- A differentiable soft patch coverage function
- Symmetry-aware patch mirroring for quarter-model efficiency
- Manufacturing penalty terms (thickness gradient, edge strength knock-down)
- A multi-objective Adam optimization loop

---

## Author

**Tharun Arety**  
M.Sc. Materials Science and Engineering, University of Augsburg  
