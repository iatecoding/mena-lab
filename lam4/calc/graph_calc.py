"""GraphPointCalculator: single-point E/F of fixed-size periodic structures as one CUDA-graph replay (diatomics sweeps, relaxations of one cell).

Built on MDEngine's static padded batch + static Dual-CSR plan: per call the host builds the neighbour list (skin 0) and copies positions, edges,
plan buffers and the composition tensors (z, u_flag, token, ref_id) into the static buffers, then replays ONE graph holding the fixed-shape CSR
forward + backward (5.6k kernel launches become one launch). Same atom count, cell and pbc required; the composition may change between calls
(a diatomics sweep over elements uses one graph). Energy = (E_net + E0) / CRYSTAL_EV exactly as `MenaCalculator.calculate`. No stress.
"""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from lam4.calc.md_engine import MDEngine, _dense_nlh


class GraphPointCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calc, **kwargs) -> None:
        super().__init__()
        self.calc, self.kwargs = calc, kwargs
        self.eng: MDEngine | None = None
        self.graph = None
        self.graph_error: str | None = None
        self._key = None
        self._regrows = 0

    def _setup(self, atoms) -> None:
        self.eng = self.calc.md_engine(atoms, temperature_K=300.0, maxwell=False, use_graph=False, skin=0.0, **self.kwargs)
        n = len(atoms)
        sdt = self.eng.state_dt
        self.e_buf = torch.zeros((), dtype=sdt, device=self.eng.dev)
        self.f_buf = torch.zeros((n, 3), dtype=sdt, device=self.eng.dev)
        self.graph, self._regrows = None, 0
        self._key = (n, atoms.cell.array.tobytes(), tuple(atoms.get_pbc()))

    def _body(self) -> None:
        e, f, _ = self.eng._forces(False)
        self.e_buf.copy_(e)
        self.f_buf.copy_(f)

    def _capture(self) -> None:
        eng = self.eng
        eng._q_cur = eng.st.q
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with _dense_nlh():
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._body()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                self._body()
        self.graph = g

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        at = self.atoms
        if self.eng is None or self._key != (len(at), at.cell.array.tobytes(), tuple(at.get_pbc())):
            self._setup(at)
        eng = self.eng
        pos = np.ascontiguousarray(at.get_positions(), dtype=np.float64)
        eng.retarget(at)
        eng.st.q.copy_(torch.from_numpy(pos).to(dtype=eng.state_dt))
        with torch.no_grad():
            eng._rebuild(q_np=pos)
            if eng.stats["regrows"] != self._regrows:
                self._regrows, self.graph = eng.stats["regrows"], None
            if self.graph is None and self.graph_error is None and eng.dev.type == "cuda" and eng.d3_mode is None:
                try:
                    self._capture()
                except Exception as exc:   # eager fallback, same static buffers
                    self.graph_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if self.graph is not None:
                self.graph.replay()
            else:
                eng._q_cur = eng.st.q
                self._body()
            packed = torch.cat([self.e_buf.view(1), self.f_buf.reshape(-1)]).cpu().numpy()
        e_ev = (float(packed[0]) + eng.e_ref_kcal) * eng.kcal_to_ev
        self.results = {"energy": e_ev, "free_energy": e_ev, "forces": packed[1:].reshape(-1, 3)}
