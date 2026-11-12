"""The same job as a command line, a Python script, or a config file, so a run started in the browser can be repeated,
scripted, or put in a methods section. Only non-default values are written."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

from mena_app import catalog, routing, structures


def _nondefault(schema: list[dict], values: dict) -> dict:
    return {p["name"]: values[p["name"]] for p in schema if values.get(p["name"]) is not None and values.get(p["name"]) != p["default"]}


def _structure_ref(s: dict) -> str:
    if s.get("format") == "builtin":
        return s["text"]
    if s.get("format") == "smiles":
        return s["text"] if s["text"].startswith("smiles:") else "smiles:" + s["text"]
    if s.get("kind") == "path" and s.get("source"):
        return Path(s["source"]).name
    if "source" in s and s["source"]:
        return str(s["source"])
    return _default_name(s)


def save_inputs(spec: dict, folder) -> tuple[str | None, str | None]:
    """Copy file-based input structures into a run folder (input.<ext>, input_end.<ext>) so its job.json re-runs from
    there. Returns the file names, or None for examples and SMILES (they need no file)."""
    names = []
    for key, stem in (("structure", "input"), ("structure2", "input_end")):
        s = spec.get(key)
        if not s or s.get("format") not in ("xyz", "extxyz", "cif", "vasp"):
            names.append(None)
            continue
        name = f"{stem}.{s['format']}"
        (Path(folder) / name).write_text(s["text"], encoding="utf-8")
        names.append(name)
    return names[0], names[1]


def to_config(spec: dict, path: str | None = None, path2: str | None = None) -> dict:
    t = catalog.task(spec["task"])
    cfg = {"task": spec["task"], "structure": path or _structure_ref(spec["structure"]),
           **_nondefault(t["params"], spec["params"]),
           **_nondefault(catalog.RUNTIME, spec["runtime"]), **_nondefault(catalog.SYSTEM, spec["system"])}
    if "structure2" in spec:
        cfg["structure2"] = path2 or _structure_ref(spec["structure2"])
    return cfg


def to_cli(spec: dict, path: str | None = None) -> str:
    cfg = to_config(spec, path)
    args = ["mena", cfg.pop("task"), cfg.pop("structure")]
    if "structure2" in cfg:
        args.append(str(cfg.pop("structure2")))
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            args.append(flag if v else "--no-" + k.replace("_", "-"))
        else:
            args += [flag, str(v)]
    if sys.platform == "win32":  # cmd.exe / PowerShell quoting, not POSIX single quotes
        return subprocess.list2cmdline(args)
    return " ".join(shlex.quote(a) for a in args)


def _py_load_expr(s: dict, override_path: str | None = None) -> tuple[bool, bool, str]:
    if override_path is not None:
        if override_path.startswith("smiles:") or override_path in structures.BUILTINS:
            return False, True, f"load_structure({override_path!r})"
        return True, False, f"read({override_path!r})"
    if s["format"] == "builtin":
        return False, True, f"load_structure({s['text']!r})"
    if s["format"] == "smiles":
        ref = s["text"] if s["text"].startswith("smiles:") else "smiles:" + s["text"]
        return False, True, f"load_structure({ref!r})"
    return True, False, f"read({_structure_ref(s)!r})"


def _md_body(p: dict) -> str:
    ens = p.get("ensemble", "nvt")
    if ens == "nve":
        return (
            "from ase import units\n"
            "from ase.md.velocitydistribution import MaxwellBoltzmannDistribution\n"
            "from ase.md.verlet import VelocityVerlet\n"
            f"MaxwellBoltzmannDistribution(atoms, temperature_K={p['temperature']})\n"
            f"md = VelocityVerlet(atoms, timestep={p['timestep']} * units.fs, trajectory='md.traj')\n"
            f"md.run({p['steps']})\n"
        )
    if ens == "npt":
        return (
            "from ase import units\n"
            "from ase.md.velocitydistribution import MaxwellBoltzmannDistribution\n"
            f"MaxwellBoltzmannDistribution(atoms, temperature_K={p['temperature']})\n"
            "try:\n"
            "    from ase.md.nose_hoover_chain import IsotropicMTKNPT\n"
            f"    md = IsotropicMTKNPT(atoms, timestep={p['timestep']} * units.fs, temperature_K={p['temperature']}, "
            f"pressure_au={p['pressure']} * units.bar, tdamp=100 * units.fs, pdamp=1000 * units.fs, trajectory='md.traj')\n"
            "except ImportError:\n"
            "    from ase.md.nptberendsen import NPTBerendsen\n"
            f"    md = NPTBerendsen(atoms, timestep={p['timestep']} * units.fs, temperature_K={p['temperature']}, "
            f"pressure_au={p['pressure']} * units.bar, taut=100 * units.fs, taup=1000 * units.fs, "
            "compressibility_au=4.57e-5 / units.bar, trajectory='md.traj')\n"
            f"md.run({p['steps']})\n"
        )
    return (
        "from ase import units\n"
        "from ase.md.langevin import Langevin\n"
        "from ase.md.velocitydistribution import MaxwellBoltzmannDistribution\n"
        f"MaxwellBoltzmannDistribution(atoms, temperature_K={p['temperature']})\n"
        f"md = Langevin(atoms, timestep={p['timestep']} * units.fs, temperature_K={p['temperature']}, "
        f"friction={p['friction']} / units.fs, trajectory='md.traj')\n"
        f"md.run({p['steps']})\n"
    )


def to_python(spec: dict, plan: dict, version: str, path: str | None = None) -> str:
    d, p, m, analysis = plan["decision"], spec["params"], plan["model"], plan["analysis"]
    backend = m.get("backend", "lam")
    is_mol = analysis["domain"] == "molecule"
    q = int(spec["system"].get("charge") or 0)
    spn = int(spec["system"].get("spin") or 1)

    use_read1, use_load1, expr1 = _py_load_expr(spec["structure"], path)
    use_read2, use_load2, expr2 = (False, False, "")
    if "structure2" in spec:
        use_read2, use_load2, expr2 = _py_load_expr(spec["structure2"])

    imports: list[str] = []
    if m.get("code_root"):
        imports.append(f"import sys\nsys.path.insert(0, {m['code_root']!r})")
    if use_read1 or use_read2:
        imports.append("from ase.io import read")
    if use_load1 or use_load2:
        imports.append("from mena_app.structures import load as load_structure")
    if backend == "lam4":
        imports.append("from lam4 import MenaCalculator")
        calc = (
            f"MenaCalculator.from_checkpoint({m.get('path')!r}, device={d['device']!r}, dtype={d['dtype']!r}, lot={d['lot']!r}"
            + (f", skin={d['skin']}" if d["skin"] else "")
            + (", low_memory=True" if d["low_memory"] else "")
            + (", compile=True" if d.get("compile") else "")
            + ")"
        )
    else:
        imports.append("from lam.models.mena_lam import MenaLAMCalculator")
        calc = (
            f"MenaLAMCalculator.from_checkpoint({m.get('path')!r}, device={d['device']!r}, dtype={d['dtype']!r}, lot={d['lot']!r}"
            + (f", skin={d['skin']}" if d["skin"] else "")
            + (", low_memory=True" if d["low_memory"] else "")
            + (", compile=True" if d.get("compile") else "")
            + ")"
        )

    setup_lines = [f"atoms = {expr1}"]
    if use_read1:
        setup_lines.append('atoms.info.pop("lot", None)  # the level of theory is set on the calculator below')
    if spec["task"] == "md":
        r = routing.md_repeat(analysis)
        if r > 1:
            setup_lines.append(f"atoms = atoms.repeat({r})")
    if is_mol:
        setup_lines.append(f'atoms.info["charge"] = {q}')
        setup_lines.append(f'atoms.info["spin"] = {spn}')
    setup_lines.append(f"calc = {calc}")
    setup_lines.append("atoms.calc = calc")

    task = spec["task"]
    if task == "singlepoint":
        body = "print('energy (eV):', atoms.get_potential_energy())\nprint('forces (eV/Å):', atoms.get_forces())\n"
    elif task == "relax":
        cell = bool(p.get("relax_cell")) and analysis["domain"] == "crystal"
        filt = "from ase.filters import FrechetCellFilter\n" if cell else ""
        target = "FrechetCellFilter(atoms)" if cell else "atoms"
        ext = "cif" if analysis["domain"] == "crystal" else ("extxyz" if analysis["domain"] == "slab" else "xyz")
        body = f"from ase.optimize import {p['optimizer']}\n{filt}opt = {p['optimizer']}({target})\nopt.run(fmax={p['fmax']}, steps={p['max_steps']})\natoms.write('relaxed.{ext}')\n"
    elif task == "md":
        body = _md_body(p)
    elif task == "vibrations":
        pre = f"from ase.optimize import LBFGS\nLBFGS(atoms).run(fmax={p['fmax']}, steps=1000)\n" if p.get("prerelax") else ""
        body = f"{pre}from ase.vibrations import Vibrations\nvib = Vibrations(atoms, delta={p['displacement']})\nvib.run()\nvib.summary()\nvib.clean()\n"
    elif task == "phonons":
        k = routing.phonon_supercell(spec, analysis)
        pre = f"from ase.filters import FrechetCellFilter\nfrom ase.optimize import LBFGS\nLBFGS(FrechetCellFilter(atoms)).run(fmax={p['fmax']}, steps=1000)\n" if p.get("prerelax") else ""
        body = (
            f"{pre}from ase.phonons import Phonons\n"
            f"ph = Phonons(atoms, calc, supercell=({k}, {k}, {k}), delta={p['displacement']})\n"
            f"ph.run()\nph.read(acoustic=True)\n"
            f"dos = ph.get_dos(kpts=({p['kpoints']}, {p['kpoints']}, {p['kpoints']})).sample_grid(npts=200, width=2e-4)\n"
            "ph.clean()\n"
        )
    elif task == "neb":
        end_lines = [f"end_atoms = {expr2}"]
        if is_mol:
            end_lines.append(f'end_atoms.info["charge"] = {q}')
            end_lines.append(f'end_atoms.info["spin"] = {spn}')
        end_lines.append("from mena_app.backends import clone_calculator")
        end_lines.append("end_atoms.calc = clone_calculator(calc)")
        if p.get("relax_ends"):
            ef = min(p["fmax"], 0.02)
            end_lines.append(f"from ase.optimize import LBFGS\nLBFGS(atoms).run(fmax={ef}, steps=500)\nLBFGS(end_atoms).run(fmax={ef}, steps=500)")
        body = (
            "\n".join(end_lines) + "\n"
            "from ase.mep import NEB\nfrom ase.optimize import FIRE\n"
            f"images = [atoms] + [atoms.copy() for _ in range({p['images']})] + [end_atoms]\n"
            "for im in images[1:-1]:\n    im.calc = clone_calculator(calc)\n"
            f"neb = NEB(images, climb={bool(p['climb'])}, method='improvedtangent')\n"
            "neb.interpolate('idpp')\n"
            f"FIRE(neb).run(fmax={p['fmax']}, steps={p['max_steps']})\n"
        )
    else:
        body = f"# The {task} task: run `{to_cli(spec, path)}` for the full workflow.\n"

    title = catalog.task(task)["title"]
    header = f'"""{title} with {m.get("title")} — exported from mena {version}."""\n'
    return header + "\n".join(imports) + "\n\n" + "\n".join(setup_lines) + "\n" + body


def to_json(spec: dict, path: str | None = None, path2: str | None = None) -> str:
    return json.dumps(to_config(spec, path, path2), indent=2, ensure_ascii=False)


def _default_name(s: dict) -> str:
    if s["format"] == "builtin":
        return s["text"]
    if s["format"] == "smiles":
        return "smiles:" + s["text"]
    ext = {"xyz": "xyz", "extxyz": "extxyz", "cif": "cif", "vasp": "vasp"}[s["format"]]
    raw = str(s.get("name") or "structure")
    stem = Path(raw).stem if Path(raw).suffix.lower() in structures.FILE_FORMATS else raw
    name = "".join(c for c in stem if c.isalnum() or c in "-_") or "structure"
    return f"{name}.{ext}"
