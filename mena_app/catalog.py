"""The tasks and their settings, as data. The web page draws its forms from this, the command line builds its flags from it,
and validation.py checks every request against it, so the three cannot disagree. Text here is read by chemists: plain words."""

from __future__ import annotations

from copy import deepcopy


def _p(name, label, type, default, help, unit="", **kw):
    return {"name": name, "label": label, "type": type, "default": default, "help": help, "unit": unit, **kw}


FMAX = _p("fmax", "Force target", "float", 0.02, "Stop when the largest force on any atom is below this. 0.05 is a quick look, 0.01 is publication quality.",
          "eV/Å", min=1e-4, max=1.0)
MAX_STEPS = _p("max_steps", "Maximum steps", "int", 500, "Stop here even if the force target was not reached.", min=1, max=20000)
OPT = _p("optimizer", "Method", "choice", "LBFGS", "LBFGS is usually fastest. FIRE is slower but copes better with a bad starting structure.",
         options=["LBFGS", "FIRE"], advanced=True)
PRE = _p("prerelax", "Relax first", "bool", True, "Vibrations are only meaningful at a relaxed structure. Keep this on unless your input is already relaxed.")

TASKS = [
    {"id": "singlepoint", "title": "Energy & forces", "icon": "point", "domains": ["molecule", "crystal", "slab"],
     "summary": "Energy, physical decomposition, forces, dipole, polarizability and charges of the structure as given.",
     "params": [_p("interpret", "Interpretation", "bool", True,
                   "Also read the model's physics heads: charges, unpaired electrons, Pauli repulsion and the HOMO–LUMO gap, "
                   "each with how far to trust it. One extra model call; fullest with Mena-LAM-6.", cli_alias="--explain")]},
    {"id": "relax", "title": "Relax structure", "icon": "down", "domains": ["molecule", "crystal", "slab"],
     "summary": "Moves the atoms downhill in energy to the nearest stable structure.",
     "params": [FMAX, MAX_STEPS, OPT,
                _p("relax_cell", "Also relax the unit cell", "bool", True, "Lets the cell shape and volume change too. Turn off to keep the lattice fixed.",
                   domains=["crystal"])]},
    {"id": "md", "title": "Molecular dynamics", "icon": "wave", "domains": ["molecule", "crystal", "slab"],
     "summary": "Follows the atoms in time at a chosen temperature.",
     "params": [_p("ensemble", "Conditions", "choice", "nvt", "Constant temperature is the usual choice. Constant pressure lets the box breathe. Constant energy is for checking stability.",
                   options=["nvt", "npt", "nve"], labels={"nvt": "Constant temperature (NVT)", "npt": "Constant temperature and pressure (NPT)", "nve": "Constant energy (NVE)"}),
                _p("temperature", "Temperature", "float", 300.0, "", "K", min=1, max=6000),
                _p("pressure", "Pressure", "float", 1.0, "Only used at constant pressure.", "bar", min=0, max=1e7, show_if={"ensemble": ["npt"]}, domains=["crystal"]),
                _p("timestep", "Time step", "float", 1.0, "1 fs is safe with hydrogen, 2 fs is fine for heavy atoms.", "fs", min=0.05, max=10),
                _p("steps", "Steps", "int", 1000, "Simulated time is steps × time step.", min=1, max=10_000_000),
                _p("friction", "Thermostat friction", "float", 0.01, "How strongly the thermostat holds the temperature.", "1/fs", min=1e-4, max=1, advanced=True),
                _p("save_every", "Save a frame every", "int", 10, "", "steps", min=1, max=100000, advanced=True),
                _p("seed", "Random seed", "int", 0, "Same seed, same start velocities.", min=0, max=2**31 - 1, advanced=True)]},
    {"id": "vibrations", "title": "Vibrations, IR & Raman", "icon": "bars", "domains": ["molecule"],
     "summary": "Vibrational frequencies, IR & Raman spectra, zero-point energy and free-energy correction of a molecule.",
     "params": [PRE, _p("temperature", "Temperature", "float", 298.15, "For the thermal corrections.", "K", min=1, max=5000),
                _p("displacement", "Displacement", "float", 0.01, "How far each atom is moved to measure the force constants.", "Å", min=0.001, max=0.1, advanced=True),
                {**FMAX, "default": 0.005, "show_if": {"prerelax": [True]}}]},
    {"id": "phonons", "title": "Phonons", "icon": "bars", "domains": ["crystal"],
     "summary": "Phonon band structure, density of states and dynamic stability of a crystal.",
     "params": [PRE, _p("supercell", "Supercell", "int", 0, "Repeat of the cell in each direction. 0 picks it for you: an odd repeat at least 9 Å across. Odd sizes avoid artefacts near Γ. Larger is more accurate and costs much more.", min=0, max=8),
                _p("temperature", "Temperature", "float", 300.0, "For the vibrational free energy.", "K", min=1, max=5000),
                _p("displacement", "Displacement", "float", 0.03, "How far each atom is moved to measure the force constants.", "Å", min=0.001, max=0.2, advanced=True),
                _p("kpoints", "DOS k-point mesh", "int", 20, "", min=4, max=60, advanced=True),
                {**FMAX, "default": 0.005, "show_if": {"prerelax": [True]}}]},
    {"id": "neb", "title": "Reaction barrier", "icon": "hill", "domains": ["molecule", "slab", "crystal"], "needs_second": True,
     "summary": "Finds the energy barrier between two structures with a nudged elastic band.",
     "params": [_p("images", "Intermediate images", "int", 7, "Points along the path between the two ends.", min=3, max=31),
                _p("relax_ends", "Relax both ends first", "bool", True, "Barriers are measured from relaxed end points."),
                {**FMAX, "default": 0.05}, {**MAX_STEPS, "default": 300},
                _p("climb", "Climbing image", "bool", True, "Pushes the highest point onto the true saddle.", advanced=True)]},
    {"id": "benchmark", "title": "Speed & memory benchmark", "icon": "gauge", "domains": ["crystal", "molecule"], "hidden": True,
     "summary": "Runs a live 3D molecular or crystal simulation or grows a system-size ladder on this computer, measuring real speed (atoms/s and atoms/ms) and peak memory.",
     "params": [_p("mode", "Simulation mode", "choice", "ladder",
                   "Live visual MD runs real-time 3D dynamics on a molecular droplet or crystal supercell; Scaling ladder grows the system in separate processes to measure peak memory across sizes.",
                   options=["visual_md", "ladder"],
                   labels={"visual_md": "Live visual MD (3D dynamics)", "ladder": "Scaling ladder (grow system)"}),
                _p("max_atoms", "Target size / largest system", "int", 10000, "In Live visual MD, builds a molecular cluster or crystal supercell near this size; in Scaling ladder, doubles the system until it passes this size.", "atoms", min=16, max=100000),
                _p("compare_low_memory", "Also measure low-memory mode", "bool", True, "Runs every size a second time with low-memory mode.", show_if={"mode": ["ladder"]}),
                _p("steps", "Steps", "int", 5, "Timed steps per size in Scaling ladder, or MD steps in Live visual MD.", min=2, max=500, advanced=True),
                _p("temperature_K", "Pulse temperature", "float", 600.0, "Initial thermal temperature in Live visual MD.", "K", min=50.0, max=4000.0, advanced=True)]},
]

RUNTIME = [
    _p("model", "Model", "choice", "auto", "Auto picks the best installed model for your structure."),
    _p("lot", "Level of theory", "choice", "auto", "The DFT method the model reproduces. Auto picks it from the structure."),
    _p("device", "Device", "choice", "auto", "Auto compares the devices for this exact job and takes the fastest."),
    _p("dtype", "Precision", "choice", "auto", "Auto uses double precision for vibrations and phonons, where it costs little.", options=["auto", "float32", "float64"]),
    _p("precision", "Precision mode", "choice", "auto", "Auto lets the router choose; Strict FP32 (IEEE) disables TF32, and Double (FP64 exact) enforces full double precision.",
       options=["auto", "fp32_ieee", "exact"]),
    _p("low_memory", "Low-memory mode", "choice", "auto", "Auto turns this on only if the job would not fit in free memory otherwise.", options=["auto", "on", "off"]),
    _p("compile", "Compiled kernels", "choice", "auto", "Faster model calls after a one-time warm-up of 5-30 s. Auto turns this on only when the job is long enough to earn the warm-up back. The first call is checked against the uncompiled model.", options=["auto", "on", "off"]),
    _p("skin", "Neighbor skin", "float", None, "Verlet neighbor-list skin buffer in Å (leave empty for automatic: 1.0 Å MD, 0.5 Å relax, 0.0 Å single-point).", "Å", min=0.0, max=3.0, nullable=True, advanced=True),
    _p("edge_mode", "Edge kernel", "choice", "auto", "Auto uses fused Metal/Triton Dual-CSR kernels on GPU. Choose ref or stream to force pure PyTorch.", options=["auto", "csr", "ref", "stream"], advanced=True),
    _p("add_dispersion", "D3 dispersion", "choice", "auto", "Auto applies DFT-D3 add-back whenever the level of theory requires it (e.g. ωB97M-V, OMC25).", options=["auto", "on", "off"], advanced=True),
    _p("u_flag", "Hubbard +U", "choice", "auto", "Auto applies MP2020 +U on transition-metal oxides/fluorides.", options=["auto", "1", "0"], labels={"auto": "Auto (MP2020)", "1": "Force +U on", "0": "Force +U off"}, advanced=True),
    _p("threads", "CPU threads", "int", None, "Leave empty for the default.", min=1, max=512, nullable=True, advanced=True),
]

SYSTEM = [
    _p("charge", "Total charge", "int", 0, "Net charge of the molecule, in units of one electron.", min=-10, max=10, nullable=True, domains=["molecule"]),
    _p("spin", "Spin multiplicity", "int", 1, "2S+1: 1 for closed shell, 2 for a radical, 3 for a triplet.", min=1, max=11, nullable=True, domains=["molecule"]),
]


LOT_LABELS = {
    "mp_pbe": "PBE (Materials Project, PBE_52)", "mp_pbeU": "PBE+U (Materials Project, PBE_52)",
    "omat_pbe": "PBE (OMat24, PBE_54)", "omat_pbeU": "PBE+U (OMat24, PBE_54)",
    "matpes_pbe": "PBE (MatPES, PBE_64)", "matpes_r2scan": "r²SCAN (MatPES, PBE_64)",
    "oc20_rpbe": "RPBE (OC20 catalysis)", "oc22_pbeU": "PBE+U (OC22 oxide catalysis)",
    "oc25_rpbe": "RPBE (OC25 catalysis)",
    "mad16_r2scan": "r²SCAN (MAD-1.6 aims_tight)", "mad16_pbe": "PBE (MAD-1.6 aims_tight)",
    "omc25_pbe": "PBE+D3 (OMC25 molecular crystals)",
    "omol_wb97mv": "ωB97M-V/def2-TZVPD (OMol25)",
    "opoly26_wb97mv": "ωB97M-V/def2-TZVPD (OPoly26)",
    "openrxn_wb97xd": "ωB97X-D3/def2-TZVPPD (OpenRxn)",
    "spice_wb97md3": "ωB97M-D3(BJ)/def2-TZVPPD (SPICE)",
}


def catalog() -> dict:
    return deepcopy({"tasks": TASKS, "runtime": RUNTIME, "system": SYSTEM, "lot_labels": LOT_LABELS})


def task(task_id: str) -> dict | None:
    return next((t for t in TASKS if t["id"] == task_id), None)
