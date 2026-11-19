"""Write the printable parts for real trials into assets/, from the same numbers as the simulated ones: the blade pattern's master (print it, mould it in RTV
silicone, pour patterns in blue injection wax), an open box to pour that mould in, and the knife's rest with its blade guard. Sizes in millimetres.

    PYTHONPATH=<yam-isaac-lab-arena> python3 make_assets.py
"""

from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import cluster as c  # noqa: E402
import mechkit as m  # noqa: E402
import task  # noqa: E402


def write_stl(path: pathlib.Path, mesh: m.Mesh, scale: float = 1000.0) -> None:
    v = mesh.v * scale
    with open(path, "wb") as f:
        f.write(b"weld-blade-waxes".ljust(80, b" "))
        f.write(struct.pack("<I", len(mesh.f)))
        for a, b, cc in mesh.f:
            n = np.cross(v[b] - v[a], v[cc] - v[a])
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *v[a], *v[b], *v[cc], 0))


def master() -> m.Mesh:
    return m.join(*(mesh for _, mesh, _ in task.pattern_parts()))


def mould_box() -> m.Mesh:
    """An open box 10 mm clear of the master all round, 3 mm walls: the master stands in it on its gate end, the gate's face on the floor."""
    inner = np.array([c.CHORD + 0.02, c.PLATFORM[2] + 0.02, c.LENGTH + 0.01])
    t = 0.003
    walls = [m.box((inner[0] + 2 * t, inner[1] + 2 * t, t), (0, 0, t / 2))]
    for sx in (-1, 1):
        walls.append(m.box((t, inner[1] + 2 * t, inner[2]), (sx * (inner[0] + t) / 2, 0, t + inner[2] / 2)))
    for sy in (-1, 1):
        walls.append(m.box((inner[0], t, inner[2]), (0, sy * (inner[1] + t) / 2, t + inner[2] / 2)))
    return m.join(*walls)


def knife_rest() -> m.Mesh:
    """The bench part's cradles, rails and guard, in the rest's own frame (the blade's centre at the origin, handle toward -x, table at z = -rest height)."""
    parts = [mesh for name, mesh, _ in task._bench()[0] if name.startswith(("cradle", "guard"))]
    out = m.join(*parts)
    return m.Mesh(out.v - np.array([c.KNIFE_REST[0], c.KNIFE_REST[1], 0.0]), out.f)


if __name__ == "__main__":
    d = HERE / "assets"
    d.mkdir(exist_ok=True)
    write_stl(d / "blade_pattern_master.stl", master())
    write_stl(d / "mould_box.stl", mould_box())
    write_stl(d / "knife_rest.stl", knife_rest())
    for p in sorted(d.glob("*.stl")):
        print(p.name, p.stat().st_size, "bytes")
