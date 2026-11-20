"""Writes the real-trial files from the task's own numbers (ring_geometry.py), so that they cannot drift apart:

    wax_ring.stl             the wax ring pattern as moulded: shank, head with its bearing, four claws standing open (mm)
    wax_ring_drawing.svg     the same, dimensioned
    blocks_and_stand.svg     the V-block, the rest, the stone's block, the tool block and the pen's stand, dimensioned

    PYTHONPATH=<yam-isaac-lab-arena> python3 assets/make_ring.py        (from the task's folder)
"""

from __future__ import annotations

import math
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import mechkit as m  # noqa: E402
import ring_geometry as g  # noqa: E402

MM = 1000.0


def ring_meshes() -> list:
    e = 0.0003
    loop = lambda prof, n: m.lathe(list(prof) + [prof[0]], n)  # noqa: E731
    shank = loop([(g.RING_IN_R, -g.SHANK_W / 2), (g.RING_OUT_R - e, -g.SHANK_W / 2), (g.RING_OUT_R, -g.SHANK_W / 2 + e), (g.RING_OUT_R, g.SHANK_W / 2 - e),
                  (g.RING_OUT_R - e, g.SHANK_W / 2), (g.RING_IN_R, g.SHANK_W / 2)], 96).aimed("+Y")
    out = [shank, loop(g.CUP_SECTION, 64)]
    for i in range(g.N_CLAWS):
        centre, rot, half = g.claw_box(i, 0.0)
        rot = rot if np.linalg.det(rot) > 0 else rot * np.array([1.0, -1.0, 1.0])  # a rotation, so that the box keeps its outside out
        box = m.box(tuple(2 * half + np.array([0.0, 0.0, 0.0003])))  # 0.6 mm into the rim, so that the claw and the head print as one piece
        out.append(m.Mesh(box.v @ rot.T + centre - rot[:, 2] * 0.0003, box.f))
    return out


def write_stl(path: pathlib.Path, meshes) -> int:
    tris = np.vstack([mesh.v[mesh.f] for mesh in meshes]) * MM
    with open(path, "wb") as f:
        f.write(b"wax ring pattern, mm".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            f.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))
    return len(tris)


class Svg:
    def __init__(self, w, h, scale):
        self.w, self.h, self.k = w, h, scale
        self.out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif" font-size="11">',
                    f'<rect width="{w}" height="{h}" fill="white"/>']

    def line(self, a, b, colour="black", width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{colour}" stroke-width="{width}"{d}/>')

    def poly(self, pts, fill="none", colour="black", width=1.0):
        self.out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" stroke="{colour}" stroke-width="{width}"/>')

    def circle(self, c, r, fill="none", colour="black", width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r:.1f}" fill="{fill}" stroke="{colour}" stroke-width="{width}"{d}/>')

    def text(self, p, s, size=11, anchor="start", colour="black"):
        self.out.append(f'<text x="{p[0]:.1f}" y="{p[1]:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}">{s}</text>')

    def dim(self, a, b, label, off=14, side=1):
        """A dimension line between two points, drawn `off` px to one side."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-9) * off * side
        self.line(a, a + n * 1.15, "#888", 0.6)
        self.line(b, b + n * 1.15, "#888", 0.6)
        self.line(a + n, b + n, "#0a5", 0.9)
        mid = (a + b) / 2 + n * 1.5
        self.text(mid + [0, 4], label, 10, "middle", "#063")

    def save(self, path):
        path.write_text("\n".join(self.out + ["</svg>"]) + "\n")


def ring_drawing(path: pathlib.Path) -> None:
    k = 20.0  # px per mm
    s = Svg(1200, 900, k)
    s.text((20, 26), "Wax ring pattern for wax setting: solitaire, US size 7, four-claw head for an 8.0 mm round stone. Millimetres. Injection wax.", 14)
    s.text((20, 46), "Left: section through two opposite claws, as moulded (claws open), with the stone pressed home drawn dashed. Right: top view.", 11)
    s.text((20, 70), f"Claws: {g.CLAW_W * MM:.1f} x {g.CLAW_W * MM:.1f}, {g.CLAW_H * MM:.1f} above the seat line, leaning out {g.CLAW_LEAN_DEG:.0f} degrees, on the diagonals (45 degrees to the shank),", 11)
    s.text((20, 86), f"their inside faces {2 * g.CLAW_RH * MM:.1f} apart at the seat line; they stand {(g.CLAW_RH + g.CLAW_W - g.CUP_R1) * MM:.1f} proud of the head's rim.", 11)
    s.text((20, 104), f"Stone: {g.STONE_D * MM:.1f} round brilliant; table {2 * g.TABLE_R * MM:.2f}, crown {g.CROWN_H * MM:.2f}, girdle {g.GIRDLE_H * MM:.2f}, pavilion {g.PAV_H * MM:.2f}.", 11, "start", "#900")
    s.text((20, 122), f"Head: {2 * g.CUP_R0 * MM:.1f} across where it meets the shank ({g.CUP_Z0 * MM:.1f} above the ring's axis), {2 * g.CUP_R1 * MM:.1f} at the rim; a {2 * g.BEAR_R * MM:.1f} hole through the middle, opening to", 11)
    s.text((20, 138), f"{g.STONE_D * MM:.1f} at the rim: that cone is the bearing the stone's pavilion sits on, its lower edge {(g.Z_S - g.BEAR_Z) * MM:.2f} under the seat line.", 11)
    ox, oy = 330.0, 670.0  # the ring's axis in the section
    P = lambda r, z: (ox + r * MM * k, oy - z * MM * k)  # noqa: E731
    s.circle(P(0, 0), g.RING_OUT_R * MM * k, "#dff4f2")
    s.circle(P(0, 0), g.RING_IN_R * MM * k, "white")
    lean = math.radians(g.CLAW_LEAN_DEG)
    for sgn in (-1, 1):
        s.poly([P(sgn * r, z) for r, z in g.CUP_SECTION], "#bfe9e6")
        hinge = np.array([sgn * g.CLAW_RH, g.Z_S])
        out, up = np.array([sgn * math.cos(lean), -math.sin(lean)]), np.array([sgn * math.sin(lean), math.cos(lean)])
        s.poly([P(*q) for q in (hinge, hinge + out * g.CLAW_W, hinge + out * g.CLAW_W + up * g.CLAW_H, hinge + up * g.CLAW_H)], "#8fd6d2")
    stone = [(r, z + g.Z_S) for r, z in g.STONE_PROFILE] + [(-r, z + g.Z_S) for r, z in reversed(g.STONE_PROFILE[1:-1])]
    s.out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in (P(*q) for q in stone))}" fill="none" stroke="#c00" stroke-width="1" stroke-dasharray="5 3"/>')
    s.line(P(-0.0125, g.Z_S), P(0.0125, g.Z_S), "#888", 0.6, "2 3")
    s.text(P(-0.0128, g.Z_S - 0.0002), "seat line", 10, "end", "#555")
    top = g.Z_S + g.CLAW_H * math.cos(lean)
    tip = g.CLAW_RH + g.CLAW_H * math.sin(lean)
    s.dim(P(-tip, top), P(tip, top), f"{2 * tip * MM:.1f} between open tips", off=-18)
    s.dim(P(-g.CLAW_RH, top), P(g.CLAW_RH, top), f"{2 * g.CLAW_RH * MM:.1f} between the claws at the seat line", off=-52)
    s.dim(P(-g.CUP_R1, top), P(g.CUP_R1, top), f"head {2 * g.CUP_R1 * MM:.1f} across at the rim", off=-86)
    s.dim(P(-g.RING_IN_R, 0), P(g.RING_IN_R, 0), f"inside {2 * g.RING_IN_R * MM:.1f}", off=10)
    s.dim(P(g.RING_OUT_R, -0.0062), P(g.RING_IN_R, -0.0062), f"shank {g.SHANK_T * MM:.1f} thick, {g.SHANK_W * MM:.1f} wide", off=14)
    s.dim(P(0.0135, 0), P(0.0135, g.Z_S), f"seat line {g.Z_S * MM:.1f}", off=16, side=-1)
    s.text(P(0.0146, g.Z_S / 2 - 0.0008), "above the ring's axis", 10, "start", "#063")
    tx, ty = 900.0, 460.0  # top view
    T = lambda x, y: (tx + x * MM * k, ty - y * MM * k)  # noqa: E731
    s.poly([T(-0.0125, -g.SHANK_W / 2), T(0.0125, -g.SHANK_W / 2), T(0.0125, g.SHANK_W / 2), T(-0.0125, g.SHANK_W / 2)], "#eef8f7", "#999", 0.6)
    s.circle(T(0, 0), g.CUP_R1 * MM * k, "#bfe9e6")
    s.circle(T(0, 0), g.STONE_R * MM * k, "white", "#c00", 1, "5 3")
    s.circle(T(0, 0), g.BEAR_R * MM * k, "white")
    for i in range(g.N_CLAWS):
        centre, rot, half = g.claw_box(i, 0.0)
        s.poly([T(c[0], c[1]) for c in (centre + rot @ (np.array([sx, sy, 1.0]) * half) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))], "#8fd6d2")
    s.dim(T(-g.CUP_R1, 0.0085), T(g.CUP_R1, 0.0085), f"{2 * g.CUP_R1 * MM:.1f}", off=-8)
    s.text(T(-0.0125, -0.0100), "Top view. The shank (pale band) runs across; the mandrel runs up the page.", 10)
    s.text(T(-0.0125, -0.0112), "Dashed: the stone's girdle. White: the hole under the stone.", 10, "start", "#666")
    s.save(path)


def blocks_drawing(path: pathlib.Path) -> None:
    k = 5.0  # px per mm
    s = Svg(1240, 900, k)
    s.text((20, 26), "Bench fixtures for the wax-setting task. Millimetres. Top views, heights in the notes; where each stands is in the README's layout table.", 14)
    s.text((20, 44), f"Funnel mouths (grey): steel plates {g.MOUTH_W * MM:.0f} wide and {g.MOUTH_T * MM:.1f} thick at 45 degrees, their lower edges on the opening's edges, standing {g.MOUTH_RISE * MM:.1f} above it.", 11)

    def rect(x, y, w, h, fill="#ead9b8"):
        s.poly([(x, y), (x + w * k, y), (x + w * k, y + h * k), (x, y + h * k)], fill)

    def mouth(cx, cy, ox, oy, length, sides):
        """Plates round an opening ox by oy (page x, y) centred at cx, cy."""
        reach = g.MOUTH_RISE * MM * k
        for dx, dy in sides:
            if dx:
                x0 = cx + dx * ox * k / 2
                s.poly([(x0, cy - length * k / 2), (x0 + dx * reach, cy - length * k / 2), (x0 + dx * reach, cy + length * k / 2), (x0, cy + length * k / 2)], "#c9ccd3")
            else:
                y0 = cy + dy * oy * k / 2
                s.poly([(cx - length * k / 2, y0), (cx + length * k / 2, y0), (cx + length * k / 2, y0 + dy * reach), (cx - length * k / 2, y0 + dy * reach)], "#c9ccd3")

    x, y = 40, 110  # stone block
    rect(x, y, g.STONE_BLOCK[0] * MM, g.STONE_BLOCK[1] * MM)
    h = g.STONE_HOLE * MM * k / 2
    cx, cy = x + g.STONE_BLOCK[0] * MM * k / 2, y + g.STONE_BLOCK[1] * MM * k / 2
    s.poly([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)], "white")
    s.text((x, y - 22), f"Stone block: beech {g.STONE_BLOCK[0] * MM:.0f} x {g.STONE_BLOCK[1] * MM:.0f} x {g.STONE_BLOCK[2] * MM:.0f} high")
    s.text((x, y - 8), f"square hole {g.STONE_HOLE * MM:.1f} across, 6 deep, in the middle")
    x, y = 330, 110  # V-block
    rect(x, y, g.V_SIZE[0] * MM, g.V_SIZE[1] * MM, "#d7d9de")
    s.line((x + g.V_SIZE[0] * MM * k / 2, y), (x + g.V_SIZE[0] * MM * k / 2, y + g.V_SIZE[1] * MM * k), "#666", 0.8, "4 3")
    s.text((x, y - 22), f"V-block: steel {g.V_SIZE[0] * MM:.0f} x {g.V_SIZE[1] * MM:.0f} x {g.V_SIZE[2] * MM:.0f} high, 90 degree V")
    s.text((x, y - 8), f"it carries the stick's axis {g.AXIS_H * MM:.0f} above the table, {g.V_Y * MM:.0f} from the ring")
    x, y = 640, 110  # rest
    rect(x, y, g.REST_SIZE[0] * MM, g.REST_SIZE[1] * MM)
    s.text((x, y - 22), f"Rest: beech {g.REST_SIZE[0] * MM:.0f} x {g.REST_SIZE[1] * MM:.0f} x {g.REST_SIZE[2] * MM:.1f} high")
    s.text((x, y - 8), f"{g.REST_Y * MM:.0f} from the ring, on the same side as the V-block")
    x, y = 40, 470  # tool block: across the page is the station's x
    rect(x, y, g.TOOL_BLOCK[0] * MM, g.TOOL_BLOCK[1] * MM)
    x_low = g.TOOL_BLOCK_XY[0] - g.TOOL_BLOCK[0] / 2
    mid = y + g.TOOL_BLOCK[1] * MM * k / 2
    tcx = x + (g.TWEEZER_XY[0] - x_low) * MM * k
    mouth(tcx, mid, g.TWEEZER_SLOT[0] * MM, g.TROUGH_TOP * MM, (g.TROUGH_TOP + 0.002) * MM, ((1, 0), (-1, 0)))
    rect(tcx - g.TWEEZER_SLOT[0] * MM * k / 2, mid - g.TROUGH_TOP * MM * k / 2, g.TWEEZER_SLOT[0] * MM, g.TROUGH_TOP * MM, "#d9c49a")  # the leaning walls, seen from above
    rect(tcx - g.TWEEZER_SLOT[0] * MM * k / 2, mid - g.TWEEZER_SLOT[1] * MM * k / 2, g.TWEEZER_SLOT[0] * MM, g.TWEEZER_SLOT[1] * MM, "white")
    pcx = x + (g.PUSHER_XY[0] - x_low) * MM * k
    mouth(pcx, mid, g.PUSHER_HOLE[0] * MM, g.PUSHER_HOLE[1] * MM, 16.0, ((1, 0), (-1, 0), (0, 1), (0, -1)))
    rect(pcx - g.PUSHER_HOLE[0] * MM * k / 2, mid - g.PUSHER_HOLE[1] * MM * k / 2, g.PUSHER_HOLE[0] * MM, g.PUSHER_HOLE[1] * MM, "white")
    s.text((tcx, y + g.TOOL_BLOCK[1] * MM * k + 16), "tweezers", 10, "middle")
    s.text((pcx, y + g.TOOL_BLOCK[1] * MM * k + 16), "pusher", 10, "middle")
    s.text((x, y - 78), f"Tool block: beech {g.TOOL_BLOCK[0] * MM:.0f} long x {g.TOOL_BLOCK[1] * MM:.0f} wide x {g.TOOL_BLOCK[2] * MM:.0f} high, its length across the bench (along x)")
    s.text((x, y - 64), f"Tweezer trough: {g.TWEEZER_SLOT[0] * MM:.1f} long, {g.TWEEZER_SLOT[2] * MM:.0f} deep, its floor {g.TWEEZER_SLOT[1] * MM:.1f} wide (white); its two long walls lean apart to "
                        f"{g.TROUGH_TOP * MM:.1f} at the block's top")
    s.text((x, y - 50), f"({math.degrees(math.atan(g.TROUGH_RUN)):.1f} degrees from upright), and it has a mouth at each end. The tips stand on the floor; the walls keep the legs upright.")
    s.text((x, y - 36), f"Pusher hole {g.PUSHER_HOLE[0] * MM:.0f} x {g.PUSHER_HOLE[1] * MM:.0f}, {g.PUSHER_HOLE[2] * MM:.0f} deep, with a mouth all round.")
    s.text((x, y - 22), f"Centres {abs(g.TWEEZER_XY[0] - g.PUSHER_XY[0]) * MM:.0f} apart, each {(g.TOOL_BLOCK[0] / 2 - abs(g.TWEEZER_XY[0] - g.TOOL_BLOCK_XY[0])) * MM:.0f} from an end, on the block's centre line.")
    x, y = 880, 110  # pen stand, top view
    rect(x, y, g.PEN_BASE[0] * MM, g.PEN_BASE[1] * MM, "#d7d9de")
    off = (g.PEN_BASE[0] - g.PEN_TUBE[0]) / 2 * MM * k
    rect(x + off, y + off, g.PEN_TUBE[0] * MM, g.PEN_TUBE[1] * MM, "#b9bcc4")
    pc = (x + g.PEN_BASE[0] * MM * k / 2, y + g.PEN_BASE[1] * MM * k / 2)
    mouth(pc[0], pc[1], g.PEN_BORE * MM, g.PEN_BORE * MM, g.PEN_TUBE[0] * MM, ((1, 0), (-1, 0), (0, 1), (0, -1)))
    b = g.PEN_BORE * MM * k / 2
    s.poly([(pc[0] - b, pc[1] - b), (pc[0] + b, pc[1] - b), (pc[0] + b, pc[1] + b), (pc[0] - b, pc[1] + b)], "white")
    s.text((x, y - 22), f"Pen stand: steel base {g.PEN_BASE[0] * MM:.0f} x {g.PEN_BASE[1] * MM:.0f} x {g.PEN_BASE[2] * MM:.0f}, square tube {g.PEN_TUBE[0] * MM:.0f} outside,")
    s.text((x, y - 8), f"its top {g.PEN_TUBE[2] * MM:.0f} above the table, with a mouth all round")
    sx, sy, kk = 990, 860, 1.6  # the stand's section
    S = lambda a, z: (sx + a * MM * k * kk, sy - z * MM * k * kk)  # noqa: E731
    half_t, half_b, half_s = g.PEN_TUBE[0] / 2, g.PEN_BORE / 2, g.PEN_TIP_BORE / 2
    s.poly([S(-g.PEN_BASE[0] / 2, 0), S(g.PEN_BASE[0] / 2, 0), S(g.PEN_BASE[0] / 2, g.PEN_BASE[2]), S(-g.PEN_BASE[0] / 2, g.PEN_BASE[2])], "#d7d9de")
    for sgn in (-1, 1):
        s.poly([S(sgn * half_t, g.PEN_BASE[2]), S(sgn * half_t, g.PEN_TUBE[2]), S(sgn * half_b, g.PEN_TUBE[2]), S(sgn * half_b, g.PEN_LEDGE_H), S(sgn * half_s, g.PEN_LEDGE_H), S(sgn * half_s, g.PEN_BASE[2])], "#b9bcc4")
        s.line(S(sgn * half_b, g.PEN_TUBE[2]), S(sgn * (half_b + g.MOUTH_RISE), g.PEN_TUBE[2] + g.MOUTH_RISE), "black", 1.5)
    s.text((sx - 170, sy - g.PEN_TUBE[2] * MM * k * kk - 92), "Section through the pen stand:", 11)
    s.text((sx - 170, sy - g.PEN_TUBE[2] * MM * k * kk - 76), f"bore {g.PEN_BORE * MM:.0f} square down to a ledge {g.PEN_LEDGE_H * MM:.0f} above the table,", 10)
    s.text((sx - 170, sy - g.PEN_TUBE[2] * MM * k * kk - 62), f"then {g.PEN_TIP_BORE * MM:.0f} square for the tip, which hangs clear of the base", 10)
    s.save(path)


if __name__ == "__main__":
    n = write_stl(HERE / "wax_ring.stl", ring_meshes())
    ring_drawing(HERE / "wax_ring_drawing.svg")
    blocks_drawing(HERE / "blocks_and_stand.svg")
    print(f"wax_ring.stl: {n} triangles; drawings written")
