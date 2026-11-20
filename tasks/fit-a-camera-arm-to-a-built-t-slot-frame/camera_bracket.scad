// Camera bracket for a 40 mm T-slot bar. Units: mm. Print in PETG, solid (100 % infill), foot down, supports under the arm and the side plate.
// Origin: the middle of the foot's underside. x along the foot's long side, y along the arm, z up.
foot = [80, 40, 6.4];
hole_d = 6.6;        // clearance for M6
hole_dx = 30;     // each hole from the foot's middle
notch = 3;          // V notch in the long edge away from the arm
arm = [20, 12];       // width, height
arm_end = 200;      // the arm's end from the foot's middle
plate = [4, 42, 42];  // side plate for the camera's back: thickness, width, height
cam_drop = 30;    // the plate hangs this far below the arm's underside
m3 = 3.4;                          // two holes 20 mm apart for the camera's M3 back threads
$fn = 48;

difference() {
    union() {
        translate([-foot[0] / 2, -foot[1] / 2, 0]) cube(foot);
        translate([-arm[0] / 2, foot[1] / 2, 0]) cube([arm[0], arm_end - foot[1] / 2, arm[1]]);
        translate([arm[0] / 2, arm_end - plate[1], arm[1] - plate[2]]) cube(plate);
    }
    for (sx = [-1, 1]) translate([sx * hole_dx, 0, -1]) cylinder(d = hole_d, h = foot[2] + 2);
    translate([0, -foot[1] / 2, -1]) linear_extrude(foot[2] + 2) polygon([[-notch, -0.01], [notch, -0.01], [0, notch]]);
    for (sy = [-10, 10]) translate([arm[0] / 2 - 1, arm_end - plate[1] / 2 + sy, arm[1] - plate[2] / 2]) rotate([0, 90, 0]) cylinder(d = m3, h = plate[0] + 2);
}
