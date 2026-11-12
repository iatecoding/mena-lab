# Mena-Lab

Fast machine-learned interatomic potentials for molecules, crystals and surfaces, with a web app and a command line
that run on your own computer (Windows, macOS, Linux; CPU, NVIDIA GPU or Apple GPU).

Relax a structure, run molecular dynamics, compute vibrations with IR and Raman spectra, phonons or a reaction barrier,
without writing code.

Mena-Lab is an independent project: the models were developed and trained by Francesco Viola with personal resources and in personal time.

## Install

```bash
pip install git+https://github.com/iatecoding/mena-lab
```

Python 3.10 or newer. For an NVIDIA GPU, install the CUDA build of [PyTorch](https://pytorch.org/get-started/locally/) first.

## Use

```bash
mena webapp
```

Opens the app in your browser. Drop in a CIF, XYZ or POSCAR file (or pick an example), choose a calculation, press **Run**.
Before running, the app shows the model, device, expected time and memory. Every result comes with plots, a 3D view,
downloadable files, and the command or Python script that repeats it.

The same from the command line:

```bash
mena sp caffeine                              # energy, forces, charges, dipole
mena relax structure.cif --fmax 0.01          # relax atoms and cell
mena md water_box.extxyz --temperature 300 --steps 10000
mena vib aspirin.xyz                          # frequencies, IR and Raman spectra
mena phonons si --supercell 3                 # band structure and density of states
mena neb reactant.xyz product.xyz --images 9  # reaction barrier
mena <task> --help                            # every option
```

A structure can be a file, `smiles:CCO`, or an example name. Each run writes a folder with the results and a `job.json`
you can repeat with `mena run job.json`.

From Python, as an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator:

```python
from mena_app import calculator
atoms.calc = calculator("lam5-v1")    # or "lam-400k"; device="cuda" / "mps" for a GPU
```

The model weights (35 MB and 26 MB) download automatically the first time they are needed.

## Models

| Model | Used for | Highlights |
|---|---|---|
| **Mena-LAM-s** (`lam5-v1`) | molecules | OMol25 held-out: energy 1.64 kcal/mol, forces 0.34 kcal/mol/Å; PLF547 protein–ligand 0.297 kcal/mol; charges, dipole, IR and Raman |
| **Mena-LAM-preview** (`lam-400k`) | crystals and surfaces | Matbench Discovery: F1 0.905, CPS 0.849, κ<sub>SRME</sub> 0.307 |

Automatic picks the model by structure type. To use another model for a type, open **Settings** in the web app, or:

```bash
mena models default crystal lam5-v1   # "auto" restores the recommended model
```

The two models have different energy references: compare energies computed with the same model.

`mena models add` loads a model file with PyTorch, which can run code stored in the file: add only files you trust.

## Benchmarks

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/accuracy-dark.svg">
  <img alt="Accuracy: protein–ligand interaction error on PLF547 (Mena-LAM-s 0.297 kcal/mol) and DynaMat CMDS on Matbench Discovery (Mena-LAM-preview 0.787)" src="docs/accuracy-light.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/speed-memory-dark.svg">
  <img alt="Speed and memory in DynaMat molecular dynamics: Mena-LAM-preview 2.4 h and 0.72 GB on one A100" src="docs/speed-memory-light.svg" width="100%">
</picture>

The web app's **Performance** tab has more results and a live speed test on your own computer.

## License

Code: [Apache-2.0](LICENSE). Model weights: [CC BY 4.0](LICENSE-WEIGHTS). Third-party material and training data: [NOTICE](NOTICE).

## Citation

A paper is in preparation. Until then, please cite this repository (Mena-Lab) and name the model you used.
