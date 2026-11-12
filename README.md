# Mena-Lab

Fast MLIP for molecules, crystals and surfaces, with a web app and a command line
that run on your own computer (Windows, macOS, Linux; CPU, NVIDIA GPU or Apple GPU).

The core idea is that bu running a single command, it should open the web interface where you can start running simulations and tests without friction

The project is mainly optimized to run on CPU, and MPS (GPU acceleration of Macs). CUDA optimization hasn't been tested extensively due to missing resources, but I believe there is a 3-5x speed improvement at least.

Mena-Lab is an independent project: the models were developed and trained independently with personal resources a lot of effort. 
The calculator uses 2 Mena models for now, one for materials, and another for molecules. Especially the molecules ones is very fast and efficient, I can say that datacenter gpu speed is achieved on a laptop, paying only partially for accuracy, and in some cases not even paying for it.

Relax a structure, run molecular dynamics, compute vibrations with IR and Raman spectra, phonons or a reaction barrier,
without writing code.



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

## License

Code: [Apache-2.0](LICENSE). Model weights: [CC BY 4.0](LICENSE-WEIGHTS). Third-party material and training data: [NOTICE](NOTICE).

## Citation

A paper hasn't been published, being alone I didn't want to publish ai slop, but maybe a technical paper on the model architecture and characteristics would be interesting. Until then, please cite this repository (Mena-Lab) and name the model you used, or contact me for further information.
