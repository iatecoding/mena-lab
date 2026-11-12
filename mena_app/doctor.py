"""`mena doctor`: what is installed, what is missing, and whether each model loads and gives forces."""

from __future__ import annotations

import importlib.util
import json
import sys

from mena_app import __version__, costmodel, machine, models, runner

CHECK = ("import json,sys;from ase.build import molecule,bulk;from mena_app import backends;"
         "m=json.loads(sys.argv[1]);a=(molecule('H2O') if 'molecule' in m['domains'] else bulk('Si'));a.info.update(charge=0,spin=1);"
         "a.calc=backends.build(m,{'device':'cpu','dtype':'float32','lot':'auto','skin':0.0,'low_memory':False});"
         "f=a.get_forces();e=a.get_potential_energy();"
         "assert f.shape==(len(a),3);"
         "print(json.dumps({'e':float(e),'fmax':float((f**2).sum(axis=1).max()**0.5)}))")


def main() -> int:
    ok = True
    print(f"mena {__version__}, Python {sys.version.split()[0]}")
    for mod in ("torch", "numpy", "ase", "vesin"):
        have = importlib.util.find_spec(mod) is not None
        ok &= have
        print(f"  {mod:8} {'ok' if have else 'MISSING (required)'}")
    try:
        from ase.build import bulk
        from mena_app import crystal_symmetry, smiles, yaml_lite

        mol = smiles.smiles_to_atoms("CCO")
        si = bulk("Si", cubic=False)
        sym = crystal_symmetry.find_crystal_symmetry(si.cell.array, si.get_scaled_positions(wrap=False), si.numbers)
        ym = yaml_lite.safe_load("a: 1\n")
        builtin_ok = len(mol) == 9 and sym is not None and len(sym["rotations"]) == 48 and ym == {"a": 1}
    except Exception:  # noqa: BLE001
        builtin_ok = False
    ok &= builtin_ok
    print(f"  builtins {'ok (smiles, symmetry, yaml)' if builtin_ok else 'FAILED'}")
    m = machine.snapshot()
    avail_mem = m["available_gb"] if m["available_gb"] is not None else "?"
    print(f"  devices  {', '.join(m['devices'])}; free memory {avail_mem} of {m['total_gb']} GB")
    print(f"  timing   {costmodel.load()['source']}")
    n_installed = 0
    for e in models.all_models():
        if not e["available"]:
            if e.get("unsupported"):
                print(f"  model {e['id']:12} {e['unsupported']}")
            elif e.get("downloadable"):
                n_installed += 1
                print(f"  model {e['id']:12} not downloaded yet ({e.get('size_mb', '?')} MB, on first use or `mena models download`)")
            else:
                print(f"  model {e['id']:12} file missing: {e.get('path')}")
            continue
        n_installed += 1
        r = runner.isolated_python(e, ["-c", CHECK, json.dumps(e)], timeout=300)
        good = r.returncode == 0
        ok &= good
        print(f"  model {e['id']:12} {'loads and computes' if good else 'FAILED: ' + (r.stderr.strip().splitlines() or ['?'])[-1][:120]}")
    if n_installed == 0:
        ok = False
        print("  models   NONE INSTALLED — run `mena models download` or `mena models add NAME PATH`")
    return 0 if ok else 1
