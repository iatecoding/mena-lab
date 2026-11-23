<img src="mena_app/static/icon.svg" alt="Mena-Lab" width="64" height="64">

# Mena-Lab

Fast machine-learned interatomic potentials for molecules, crystals, and surfaces, with a local web app, command line,
and Python API that run on your own computer (Windows, macOS, Linux; CPU, NVIDIA CUDA GPU, or Apple Silicon MPS GPU).

A single command opens the browser interface so you can inspect structures, preview runtime cost and memory estimates,
relax geometries, run molecular dynamics, compute IR/Raman spectra or phonons, and find reaction barriers without boilerplate.

Mena-Lab is an independent project developed and trained with personal resources. It ships three compact models (~26–35 MB):
**Mena-LAM-6** (`lam6-v1`, default for crystals & surfaces, with physics interpretation heads) and **Mena-LAM-s** (`lam5-v1`,
default for molecules, IR & Raman, and large biomolecular scaling), plus **Mena-LAM-preview** (`lam-400k`) kept for comparison.
Fused Metal/Triton/CPU kernels and low-memory modes bring high-throughput molecular and materials simulation to laptops and workstations;
NVIDIA CUDA runs use Triton Dual-CSR / fused kernels when available and fall back cleanly to PyTorch without requiring a local CUDA build toolchain.

## Install

```bash
pip install git+https://github.com/iatecoding/mena-lab
```

Python 3.10 or newer. For an NVIDIA GPU, install the CUDA build of [PyTorch](https://pytorch.org/get-started/locally/) first.

## Use

```bash
mena webapp
```

Opens the app in your browser. Drop in a CIF, XYZ, or POSCAR file (or pick an example), choose a calculation, and press **Run**.
Before running, the app shows the model, device, level of theory, expected time, and memory. Every result comes with plots,
an interactive 3D view, downloadable files, and the CLI/Python/JSON script that reproduces it.

The same from the command line:

```bash
mena sp caffeine --explain                    # energy, forces, charges, dipole & physics heads
mena relax structure.cif --fmax 0.01          # relax atoms and cell
mena md water_box.extxyz --temperature 300 --steps 10000
mena vib aspirin.xyz                          # frequencies, IR and Raman spectra
mena phonons si --supercell 3                 # band structure and density of states
mena neb reactant.xyz product.xyz --images 9  # reaction barrier
mena <task> --help                            # every option
```

A structure can be a file, `smiles:CCO`, or an example name. Each run writes a folder with the results and a `job.json`
you can repeat with `mena run job.json`.

From Python, as an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator or one-call helpers:

```python
from mena_app import calculator, singlepoint

res = singlepoint("caffeine", device="auto")      # picks lam5-v1 for molecules, lam6-v1 for crystals/surfaces
atoms.calc = calculator("lam6-v1")                # or "lam5-v1" / "lam-400k"; device="cuda" / "mps" for a GPU
```

The model weights (`34 MB`, `35 MB`, and `26 MB`) download automatically the first time they are needed (and are verified by SHA-256).

### Tips for SMILES & Molecular Dynamics

- **Relax SMILES structures first:** Entering a SMILES string builds one approximate 3D force-field conformer (not a DFT/MLIP minimum, and not a full conformer search). Run `mena relax` first before comparing single-point energies/dipoles, or turn on **Relax first** (`--prerelax`) when starting MD directly from SMILES so stored ring/steric strain does not spike the temperature.
- **MD time step & thermostat friction:** `1.0 fs` works well for relaxed organic molecules at `300 K`; use `0.5 fs` above `500 K` or for stiff/rough geometries, and `2.0 fs` only for heavy-atom systems without hydrogen. Default friction `0.01 /fs` (`τ = 100 fs`) suits production NVT; increase to `0.05–0.1 /fs` (`τ = 10–20 fs`) to damp initial strain faster during equilibration, or use **NVE** to check energy conservation.
- **Verify charge & spin on edge cases:** Formal charges and simple organic radicals (`[O-]`, `[NH4+]`, `[CH3]`) are inferred automatically from SMILES, but check **Charge** and **Spin multiplicity** explicitly for diradicals (e.g. triplet `O₂`), carbenes, or transition-metal complexes. Multi-fragment SMILES (e.g. `A.B`) run in vacuum without a confining box.
- **Model selection:** Keep **Auto** (`Mena-LAM-s` / `lam5-v1` for molecules; `Mena-LAM-6` / `lam6-v1` for crystals & surfaces). Pick `lam6-v1` explicitly on molecules if you want its extra physics-head readouts (`--explain`: HOMO–LUMO gap, Pauli overlap, unpaired-electron index).

## Models

| Model | Used for | Highlights |
|---|---|---|
| **Mena-LAM-6** (`lam6-v1`) | crystals and surfaces | Default materials model; Matbench Discovery CPS 0.874, F1 0.904, κ<sub>SRME</sub> 0.182; OMol25 val energy 2.00, forces 0.40 kcal/mol/Å; OC20, OC25, MAD-1.6, OMC25, SPICE 2; physics heads (`--explain`: HOMO–LUMO gap, unpaired-electron index, Pauli overlap, spin exchange) |
| **Mena-LAM-s** (`lam5-v1`) | molecules | Default molecular model; OMol25 held-out energy 1.64 kcal/mol, forces 0.34 kcal/mol/Å; PLF547 protein–ligand 0.297 kcal/mol; partial charges, dipole, polarizability, IR and Raman |
| **Mena-LAM-preview** (`lam-400k`) | crystals and surfaces | Earlier preview model kept for comparison; Matbench Discovery F1 0.905, CPS 0.849, κ<sub>SRME</sub> 0.307; DynaMat MD CMDS 0.787 |

Automatic picks the model by structure type (`lam6-v1` for crystals/surfaces, `lam5-v1` for molecules). To override the default for a domain, open **Settings** in the web app, or:

```bash
mena models default crystal lam5-v1   # "auto" restores the recommended model (lam6-v1)
```

Each model carries its own learned energy reference offsets: compare relative/total energies computed with the same model and level of theory.

`mena models add` loads a checkpoint file with PyTorch: add only files you trust.

## Benchmarks

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/molecules-dark.svg">
  <img alt="Mena-LAM-s on molecules: protein–ligand interaction error 0.297 kcal/mol on PLF547; 29,808 atoms per second and 2.69 GB peak GPU memory (1.28 GB low-memory) at 8,192 atoms on an Apple M3 Max laptop" src="docs/molecules-light.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/materials-dark.svg">
  <img alt="Mena-LAM-6 on crystals and surfaces: CPS 0.874, 555.3 atoms/ms on RTX PRO 6000 (98.6 on M3 Max), DynaMat MD in 43.6 min and 0.46 GB VRAM" src="docs/materials-light.svg" width="100%">
</picture>

The web app's **Performance** tab shows detailed benchmarks and lets you run a live 3D simulation or scaling test on your own computer.

## License

Code: [Apache-2.0](LICENSE). Model weights: [CC BY 4.0](LICENSE-WEIGHTS). Third-party material and training data: [NOTICE](NOTICE).

## Citation

If you use Mena-Lab or any of the Mena-LAM weights in academic or industrial work, please cite this repository and name the model (`Mena-LAM-6`, `Mena-LAM-s`, or `Mena-LAM-preview`) and level of theory used (see `CITATION.cff`).
