"""Writes the camera bracket's drawing (camera_bracket.svg) and its printable model (camera_bracket.scad) from the sizes in tslot_geometry.py.

    python3 drawing.py
"""
import pathlib

import tslot_geometry as g

HERE = pathlib.Path(__file__).resolve().parent
MM = 1000.0
S = 3.0  # drawing scale: pixels a millimetre


def scad() -> str:
    fx, fy, fz = (v * MM for v in g.FOOT)
    return f"""// Camera bracket for a 40 mm T-slot bar. Units: mm. Print in PETG, solid (100 % infill), foot down, supports under the arm and the side plate.
// Origin: the middle of the foot's underside. x along the foot's long side, y along the arm, z up.
foot = [{fx:g}, {fy:g}, {fz:g}];
hole_d = {g.HOLE_D * MM:g};        // clearance for M6
hole_dx = {g.SCREW_DX * MM:g};     // each hole from the foot's middle
notch = {g.NOTCH * MM:g};          // V notch in the long edge away from the arm
arm = [{g.ARM_W * MM:g}, {g.ARM_H * MM:g}];       // width, height
arm_end = {g.ARM_END * MM:g};      // the arm's end from the foot's middle
plate = [{g.PLATE[0] * MM:g}, {g.PLATE[1] * MM:g}, {g.PLATE[2] * MM:g}];  // side plate for the camera's back: thickness, width, height
cam_drop = {g.CAM_DROP * MM:g};    // the plate hangs this far below the arm's underside
m3 = 3.4;                          // two holes 20 mm apart for the camera's M3 back threads
$fn = 48;

difference() {{
    union() {{
        translate([-foot[0] / 2, -foot[1] / 2, 0]) cube(foot);
        translate([-arm[0] / 2, foot[1] / 2, 0]) cube([arm[0], arm_end - foot[1] / 2, arm[1]]);
        translate([arm[0] / 2, arm_end - plate[1], arm[1] - plate[2]]) cube(plate);
    }}
    for (sx = [-1, 1]) translate([sx * hole_dx, 0, -1]) cylinder(d = hole_d, h = foot[2] + 2);
    translate([0, -foot[1] / 2, -1]) linear_extrude(foot[2] + 2) polygon([[-notch, -0.01], [notch, -0.01], [0, notch]]);
    for (sy = [-10, 10]) translate([arm[0] / 2 - 1, arm_end - plate[1] / 2 + sy, arm[1] - plate[2] / 2]) rotate([0, 90, 0]) cylinder(d = m3, h = plate[0] + 2);
}}
"""


def svg() -> str:
    fx, fy, fz = (v * MM for v in g.FOOT)
    aw, ah, ae = g.ARM_W * MM, g.ARM_H * MM, g.ARM_END * MM
    pt, pw, ph = (v * MM for v in g.PLATE)
    cd, cw, ch = (v * MM for v in g.CAM)
    drop = g.CAM_DROP * MM
    out = []

    def rect(x, y, w, h, fill="none", dash=""):
        out.append(f'<rect x="{x * S:.1f}" y="{y * S:.1f}" width="{w * S:.1f}" height="{h * S:.1f}" fill="{fill}" stroke="#222" stroke-width="1"{dash}/>')

    def text(x, y, s, size=11, anchor="middle"):
        out.append(f'<text x="{x * S:.1f}" y="{y * S:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="#222">{s}</text>')

    def circle(x, y, d):
        out.append(f'<circle cx="{x * S:.1f}" cy="{y * S:.1f}" r="{d / 2 * S:.1f}" fill="none" stroke="#222" stroke-width="1"/>')

    dash = ' stroke-dasharray="4 3"'
    # plan view: the arm runs to the right (the bracket's y), the foot's long side runs down the page (its x)
    ox, oy = 80.0, 62.0
    out.append(f'<polygon points="{(ox - fy / 2) * S:.1f},{(oy - fx / 2) * S:.1f} {(ox + fy / 2) * S:.1f},{(oy - fx / 2) * S:.1f} {(ox + fy / 2) * S:.1f},{(oy + fx / 2) * S:.1f} '
               f'{(ox - fy / 2) * S:.1f},{(oy + fx / 2) * S:.1f} {(ox - fy / 2) * S:.1f},{(oy + g.NOTCH * MM) * S:.1f} {(ox - fy / 2 + g.NOTCH * MM) * S:.1f},{oy * S:.1f} '
               f'{(ox - fy / 2) * S:.1f},{(oy - g.NOTCH * MM) * S:.1f}" fill="#eee" stroke="#222" stroke-width="1"/>')
    rect(ox + fy / 2, oy - aw / 2, ae - fy / 2, aw, "#eee")
    rect(ox + ae - pw, oy + aw / 2, pw, pt, "#ddd")
    rect(ox + ae - cw, oy + aw / 2 + pt, cw, cd, "none", dash)
    for sx in (-1, 1):
        circle(ox, oy + sx * g.SCREW_DX * MM, g.HOLE_D * MM)
    text(ox, oy - fx / 2 - 4, f"foot {fx:g} x {fy:g} x {fz:g}")
    text(ox - fy / 2 - 3, oy + 1.5, f"V notch {g.NOTCH * MM:g} deep", 10, "end")
    text(ox - fy / 2, oy + fx / 2 + 6, f"2 holes {g.HOLE_D * MM:g}, {2 * g.SCREW_DX * MM:g} apart", 10, "start")
    text(ox + 110, oy - aw / 2 - 3, f"arm {aw:g} wide, {ah:g} high, to {ae:g} from the foot's middle")
    text(ox + ae, oy + aw / 2 + pt + cd + 6, f"side plate {pw:g} x {ph:g} x {pt:g}; camera {cw:g} x {ch:g} x {cd:g} (dashed)", 10, "end")
    text(ox + g.GRIP_Y * MM + 2, oy + 1.5, "hold here", 10, "start")
    out.append(f'<line x1="{(ox + g.GRIP_Y * MM) * S:.1f}" y1="{(oy - aw / 2) * S:.1f}" x2="{(ox + g.GRIP_Y * MM) * S:.1f}" y2="{(oy + aw / 2) * S:.1f}" stroke="#888" stroke-width="1"{dash}/>')
    text(ox - fy / 2, 12, "Plan (from above)", 13, "start")
    # side view: z up the page
    sy = 190.0
    rect(ox - fy / 2, sy - fz, fy, fz, "#eee")
    rect(ox + fy / 2, sy - ah, ae - fy / 2, ah, "#eee")
    rect(ox + ae - pw, sy - ah, pw, ph, "#ddd")
    for name, z0, z1 in (("screw", -g.SCREW_END * MM, fz + g.HEAD_H * MM),):
        rect(ox - 3, sy - z1, 6, z1 - z0, "none", dash)
    rect(ox - g.NUT_B * MM / 2, sy + g.NUT_TOP * MM, g.NUT_B * MM, g.NUT_F * MM, "none", dash)
    rect(ox - g.NUT_B * MM / 2, sy + (g.NUT_TOP - g.NUT_E) * MM, g.NUT_B * MM, g.NUT_E * MM, "none", dash)
    text(ox - fy / 2, sy - 40, "Side (along the foot's long side)", 13, "start")
    text(ox + 110, sy - ah - 3, f"undersides of foot and arm flush; the arm is {ah:g} high")
    text(ox + ae, sy - ah + ph + 6, f"plate {ph:g} high: top level with the arm's, {drop:g} below its underside", 10, "end")
    text(ox - fy / 2, sy + g.SCREW_END * MM + 10, "M6 x 16 cap screw and 80/20 3931 nut (dashed):", 10, "start")
    text(ox - fy / 2, sy + g.SCREW_END * MM + 15, f"nut threaded on until the screw's end, {g.SCREW_END * MM:g} below the foot, is flush with its underside", 10, "start")
    text(ox - fy / 2, 252, "Print in PETG, solid, foot down. Two 3.4 mm holes 20 mm apart in the side plate take the camera's M3 back threads. All sizes in mm.", 11, "start")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{310 * S:.0f}" height="{262 * S:.0f}" viewBox="0 0 {310 * S:.0f} {262 * S:.0f}">\n'
            f'<rect width="100%" height="100%" fill="white"/>\n' + "\n".join(out) + "\n</svg>\n")


if __name__ == "__main__":
    (HERE / "camera_bracket.scad").write_text(scad())
    (HERE / "camera_bracket.svg").write_text(svg())
    print("wrote camera_bracket.scad and camera_bracket.svg")
