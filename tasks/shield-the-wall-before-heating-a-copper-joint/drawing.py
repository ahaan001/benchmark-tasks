"""Writes wall_section.svg, the build drawing of the wall section and the bench layout, from layout.py:

    python drawing.py

Two views, in centimetres: the wall from the front (as the arms see it) and the bench from above.
"""

from pathlib import Path

import layout as L

S = 9.0  # pixels per cm
CM = 100.0


def rect(x, y, w, h, fill, stroke="#333", extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1" {extra}/>'


def text(x, y, s, size=12, anchor="middle"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="#111">{s}</text>'


def dim(x0, y0, x1, y1, label, off=14):
    """A dimension line between two points, labelled at its middle."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    vertical = abs(x1 - x0) < 1e-6
    tick = (f'<line x1="{x0 - 4:.1f}" y1="{y0:.1f}" x2="{x0 + 4:.1f}" y2="{y0:.1f}" stroke="#06c"/><line x1="{x1 - 4:.1f}" y1="{y1:.1f}" x2="{x1 + 4:.1f}" y2="{y1:.1f}" stroke="#06c"/>'
            if vertical else
            f'<line x1="{x0:.1f}" y1="{y0 - 4:.1f}" x2="{x0:.1f}" y2="{y0 + 4:.1f}" stroke="#06c"/><line x1="{x1:.1f}" y1="{y1 - 4:.1f}" x2="{x1:.1f}" y2="{y1 + 4:.1f}" stroke="#06c"/>')
    where = text(mx + off, my + 4, label, 11, "start") if vertical else text(mx, my - 5, label, 11)
    return f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#06c"/>' + tick + where


def front_view(ox, oy):
    """Seen from the arms: y to the left, z up. The viewer's left is +y."""
    def X(y):
        return ox + (L.WALL_Y1 - y) * CM * S

    def Z(z):
        return oy - z * CM * S

    out = [text(ox + (L.WALL_Y1 - L.WALL_Y0) * CM * S / 2, Z(L.WALL_H) - 40, "Wall section from the front (cm)", 15)]
    out.append(rect(X(L.WALL_Y1), Z(L.WALL_H), (L.WALL_Y1 - L.WALL_Y0) * CM * S, L.WALL_H * CM * S, "#e2c58a"))
    for y0, y1 in ((L.STUD_L_Y0, L.WALL_Y1), (L.WALL_Y0, L.STUD_R_Y1)):
        out.append(rect(X(y1), Z(L.WALL_H), (y1 - y0) * CM * S, L.WALL_H * CM * S, "#ecd9a8"))
    out.append(rect(X(L.STUD_L_Y0), Z(L.BLOCK_TOP), L.BAY * CM * S, L.STUD_T * CM * S, "#ecd9a8"))
    py = L.PIPE_XY[1]
    out.append(rect(X(py + L.PIPE_D / 2), Z(L.WALL_H), L.PIPE_D * CM * S, L.WALL_H * CM * S, "#c9793a"))
    out.append(rect(X(py + L.COUPLING_D / 2), Z(L.COUPLING_Z + L.COUPLING_L / 2), L.COUPLING_D * CM * S, L.COUPLING_L * CM * S, "#e39a55"))
    gy, gz = L.SHIELD_GOAL[1], L.BLOCK_TOP
    out.append(rect(X(gy + L.PANEL_W / 2), Z(gz + L.SHEET), L.PANEL_W * CM * S, L.SHEET * CM * S, "none", "#888", 'stroke-dasharray="6 4"'))
    out.append(text(X(gy) - 4, Z(gz + L.SHEET) + 16, "shield, in place", 11))
    base = Z(0) + 22
    out.append(dim(X(L.WALL_Y1), base, X(L.WALL_Y0), base, f"{(L.WALL_Y1 - L.WALL_Y0) * CM:.1f} overall"))
    out.append(dim(X(L.STUD_L_Y0), base + 26, X(L.STUD_R_Y1), base + 26, f"{L.BAY * CM:.1f} between studs (24 in on centre)"))
    out.append(dim(X(py), base + 52, X(L.STUD_R_Y1), base + 52, f"{(py - L.STUD_R_Y1) * CM:.1f} riser to stud"))
    right = X(L.WALL_Y0) + 18
    out.append(dim(right, Z(0), right, Z(L.WALL_H), f"{L.WALL_H * CM:.0f}"))
    out.append(dim(right + 48, Z(0), right + 48, Z(L.BLOCK_TOP), f"{L.BLOCK_TOP * CM:.1f} block top"))
    out.append(dim(right + 150, Z(0), right + 150, Z(L.COUPLING_Z), f"{L.COUPLING_Z * CM:.1f} coupling centre"))
    out.append(text(X(L.WALL_Y1) + 4, Z(L.WALL_H) - 10, f"2x4 studs and block: {L.STUD_T * CM:.1f} x {L.STUD_D * CM:.1f}. Sheathing: 7/16 in OSB. Riser: 1/2 in copper tube, {L.PIPE_D * CM:.2f} across.", 11, "start"))
    return out


def top_view(ox, oy):
    """Seen from above: x away from the arms (up the page), y to the left."""
    def X(y):
        return ox + (L.WALL_Y1 - y) * CM * S

    def Y(x):
        return oy - x * CM * S

    out = [text(ox + (L.WALL_Y1 - L.WALL_Y0) * CM * S / 2, Y(L.OSB_X + L.OSB_T) - 30, "Bench from above (cm): x from the line through the arm bases, y from their midpoint", 15)]
    out.append(rect(X(L.WALL_Y1), Y(L.OSB_X + L.OSB_T), (L.WALL_Y1 - L.WALL_Y0) * CM * S, L.OSB_T * CM * S, "#e2c58a"))
    for y0, y1 in ((L.STUD_L_Y0, L.WALL_Y1), (L.WALL_Y0, L.STUD_R_Y1)):
        out.append(rect(X(y1), Y(L.OSB_X), (y1 - y0) * CM * S, L.STUD_D * CM * S, "#ecd9a8"))
    out.append(rect(X(L.STUD_L_Y0), Y(L.OSB_X), L.BAY * CM * S, L.STUD_D * CM * S, "#f3e6c4"))
    px, py = L.PIPE_XY
    out.append(f'<circle cx="{X(py):.1f}" cy="{Y(px):.1f}" r="{L.PIPE_D / 2 * CM * S:.1f}" fill="#c9793a" stroke="#333"/>')
    sx, sy = L.SHIELD_START[0], L.SHIELD_START[1]
    out.append(rect(X(sy + L.PANEL_W / 2), Y(sx + L.SHIELD_T / 2), L.PANEL_W * CM * S, L.SHIELD_T * CM * S, "#fff"))
    out.append(rect(X(sy + L.PANEL_W / 2), Y(sx - L.SHIELD_T / 2), L.SHIELD_T * CM * S, L.WING * CM * S, "#fff"))
    out.append(text(X(sy) + 10, Y(sx - 0.05), "shield at the start", 11))
    tx, ty = L.TORCH_HOME
    out.append(f'<circle cx="{X(ty):.1f}" cy="{Y(tx):.1f}" r="{L.MARK_D / 2 * CM * S:.1f}" fill="#fff" stroke="#333"/>')
    out.append(f'<circle cx="{X(ty):.1f}" cy="{Y(tx):.1f}" r="{L.CYL_D / 2 * CM * S:.1f}" fill="#2a56b0" stroke="#333"/>')
    out.append(text(X(ty), Y(tx - 0.075), "torch on its mark", 11))
    for y in (0.305, -0.305):
        out.append(f'<circle cx="{X(y):.1f}" cy="{Y(0):.1f}" r="12" fill="#222"/>')
    out.append(text(X(0), Y(0) + 5, "arm bases", 11))
    left = X(L.WALL_Y1) - 20
    out.append(dim(left, Y(0), left, Y(L.OSB_X), f"{L.OSB_X * CM:.1f} sheathing face", -110))
    out.append(dim(X(py) + 40, Y(px), X(py) + 40, Y(L.OSB_X), f"{(L.OSB_X - px) * CM:.2f}"))
    out.append(dim(X(ty) + 60, Y(0), X(ty) + 60, Y(tx), f"{tx * CM:.1f}"))
    out.append(dim(X(0), Y(px) + 18, X(py), Y(px) + 18, f"{-py * CM:.1f}"))
    out.append(dim(X(0), Y(tx) + 60, X(ty), Y(tx) + 60, f"{-ty * CM:.1f}"))
    out.append(dim(X(sy + L.PANEL_W / 2), Y(sx) - 16, X(sy - L.PANEL_W / 2), Y(sx) - 16, f"panel {L.PANEL_W * CM:.1f}, y {(sy - L.PANEL_W / 2) * CM:.1f} to {(sy + L.PANEL_W / 2) * CM:.1f}, face at x {(sx - L.SHIELD_T / 2) * CM:.1f}"))
    out.append(dim(X(sy + L.PANEL_W / 2) - 16, Y(sx - L.SHIELD_T / 2), X(sy + L.PANEL_W / 2) - 16, Y(sx - L.SHIELD_T / 2 - L.WING), f"wing {L.WING * CM:.1f}", -62))
    return out


def main():
    width = (L.WALL_Y1 - L.WALL_Y0) * CM * S + 420
    front_h = L.WALL_H * CM * S + 150
    top_h = (L.OSB_X + L.OSB_T) * CM * S + 110
    body = front_view(60, 60 + L.WALL_H * CM * S) + top_view(60, front_h + 60 + (L.OSB_X + L.OSB_T) * CM * S)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{front_h + top_h + 40:.0f}" viewBox="0 0 {width:.0f} {front_h + top_h + 40:.0f}">'
           f'<rect width="100%" height="100%" fill="#fff"/>' + "".join(body) + "</svg>\n")
    Path(__file__).with_name("wall_section.svg").write_text(svg)


if __name__ == "__main__":
    main()
