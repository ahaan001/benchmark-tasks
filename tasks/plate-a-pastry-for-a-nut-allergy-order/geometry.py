"""Shapes and touch tests for the allergy-order task. NumPy only.

- Meshes built from numbers: a croissant (a ridged spindle with a flat base, as a drawn mesh plus convex collider
  pieces), its almond and icing-sugar topping (drawn only), and a plate (a turned profile).
- Convex pieces in the world (boxes and hulls) with points sampled on their surfaces, and :func:`gap`, the distance
  between two bodies: the smallest signed distance of any surface point of one to a convex piece of the other.
  ``touching`` is a gap under a threshold. A hull's signed distance is the largest of its face-plane distances, exact
  near a face and never larger than the true distance elsewhere, so a touch is never missed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# -- convex hulls -------------------------------------------------------------------------------------------------


def hull_planes(points, eps: float = 1e-7) -> np.ndarray:
    """The face planes (outward unit normal n and offset d, n . x = d) of the convex hull of at most 64 points."""
    p = np.asarray(points, float)
    n = len(p)
    planes = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nrm = np.cross(p[j] - p[i], p[k] - p[i])
                length = np.linalg.norm(nrm)
                if length < 1e-12:
                    continue
                nrm = nrm / length
                side = (p - p[i]) @ nrm
                if side.max() > eps and side.min() < -eps:
                    continue
                if side.max() > eps:
                    nrm = -nrm
                d = float(nrm @ p[i])
                key = tuple(np.round(np.r_[nrm, d], 6))
                if key not in seen:
                    seen.add(key)
                    planes.append(np.r_[nrm, d])
    if len(planes) < 4:
        raise ValueError("points are flat: no convex hull with volume")
    return np.array(planes)


def hull_faces(points, planes=None) -> np.ndarray:
    """Triangles (counter-clockwise from outside) covering the convex hull of ``points``."""
    p = np.asarray(points, float)
    planes = hull_planes(p) if planes is None else planes
    faces = []
    for pl in planes:
        nrm, d = pl[:3], pl[3]
        on = np.where(np.abs(p @ nrm - d) < 1e-6)[0]
        c = p[on].mean(axis=0)
        u = p[on[0]] - c
        u = u / np.linalg.norm(u)
        v = np.cross(nrm, u)
        order = on[np.argsort(np.arctan2((p[on] - c) @ v, (p[on] - c) @ u))]
        faces += [(int(order[0]), int(order[a]), int(order[a + 1])) for a in range(1, len(order) - 1)]
    return np.array(faces, int)


# -- meshes ---------------------------------------------------------------------------------------------------------


def vertex_normals(points, faces) -> np.ndarray:
    p, f = np.asarray(points, float), np.asarray(faces, int)
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    out = np.zeros_like(p)
    for c in range(3):
        np.add.at(out, f[:, c], fn)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def loft(rings) -> tuple[np.ndarray, np.ndarray]:
    """A closed surface through equal-sized rings of points (each counter-clockwise seen from the last ring's side),
    capped at both ends. Returns (points, triangles)."""
    rings = [np.asarray(r, float) for r in rings]
    k = len(rings[0])
    pts = np.vstack(rings + [rings[0].mean(axis=0)[None], rings[-1].mean(axis=0)[None]])
    first, last = len(rings) * k, len(rings) * k + 1
    faces = []
    for a in range(len(rings) - 1):
        for i in range(k):
            j = (i + 1) % k
            p, q, r, s = a * k + i, a * k + j, (a + 1) * k + j, (a + 1) * k + i
            faces += [(p, q, r), (p, r, s)]
    base = (len(rings) - 1) * k
    for i in range(k):
        j = (i + 1) % k
        faces += [(first, j, i), (last, base + i, base + j)]
    return pts, np.array(faces, int)


@dataclass(frozen=True)
class Croissant:
    """A straight butter croissant lying on its base: ``length`` along x, ``width`` along y, ``height`` up (m)."""

    length: float = 0.140
    width: float = 0.064
    height: float = 0.045
    ridges: int = 7
    step: float = 0.05  # each rolled layer's edge stands this fraction proud of the surface
    bow: float = 0.008  # how far the middle bows sideways (+y) from the line through the tips

    def _raw(self, s: float, k: int, ridged: bool, fit=(1.0, 1.0)) -> np.ndarray:
        taper = (1.0 - abs(s) ** 2.2) ** 0.7
        layer = abs(s) * self.ridges / 2
        swell = 1.0 + (self.step * (2 * (layer - math.floor(layer)) - 1) if ridged else 0.0)
        w = max(fit[0] * self.width / 2 * taper * swell, 0.004)
        h = max(fit[1] * self.height * (0.22 + 0.78 * taper) * swell, 0.008)
        phi = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
        y = w * np.cos(phi)
        up = np.sin(phi)
        z = np.where(up >= 0, h / 2 * (1 + up), h / 2 * (1 - np.abs(up) ** 0.4))
        x = np.full(k, s * self.length / 2 / 0.985)
        return np.stack([x, y + self.bow * (0.5 - s * s), z], axis=1)

    def _fit(self) -> tuple[float, float]:
        """Scale factors that make the ridged surface exactly ``width`` wide (bow included) and ``height`` tall."""
        fit = (1.0, 1.0)
        for _ in range(6):
            p = np.vstack([self._raw(v, 72, True, fit) for v in np.linspace(-0.985, 0.985, 199)])
            fit = (fit[0] * self.width / (p[:, 1].max() - p[:, 1].min()), fit[1] * self.height / p[:, 2].max())
        return fit

    def _section(self, s: float, k: int, ridged: bool) -> np.ndarray:
        """The outline at s in -1..1 along the length: k points (x, y, z), counter-clockwise seen from +x. Ridged, the
        surface steps down at the edge of each rolled layer going out from the middle, as a croissant's does."""
        if "fit" not in self.__dict__:
            object.__setattr__(self, "fit", self._fit())
        return self._raw(s, k, ridged, self.__dict__["fit"])

    def mesh(self, sections: int = 43, around: int = 20):
        """(points, triangles, normals) of the drawn surface, ridged."""
        s = np.linspace(-0.985, 0.985, sections)
        pts, faces = loft([self._section(v, around, True) for v in s])
        return pts, faces, vertex_normals(pts, faces)

    def pieces(self, count: int = 8, around: int = 12) -> list[np.ndarray]:
        """Convex collider pieces: the hull of each pair of neighbouring outlines, unridged, midway between the drawn
        surface's ridge crests and the steps below them."""
        s = np.linspace(-0.985, 0.985, count + 1)
        rings = [self._section(v, around, False) for v in s]
        return [np.vstack([rings[i], rings[i + 1]]) for i in range(count)]

    def topping(self, seed: int = 3, flakes: int = 16):
        """Drawn-only almond-croissant topping: (sugar mesh, list of flake meshes), each (points, triangles,
        normals). The sugar is a skin 0.8 mm over the top of the middle of the croissant; the flakes are sliced
        almonds, 16 x 9 x 1.6 mm, lying on it."""
        around, sections = 20, 31
        s = np.linspace(-0.72, 0.72, sections)
        top = [i for i in range(around) if 28.0 <= math.degrees(2 * math.pi * i / around) <= 152.0]
        rows = []
        for v in s:
            ring = self._section(v, around, True)
            c = ring.mean(axis=0)
            out = ring[top] - c
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
            rows.append(ring[top] + 0.0008 * out)
        k = len(top)
        pts = np.vstack(rows)
        faces = []
        for a in range(sections - 1):
            for i in range(k - 1):
                p, q, r, t = a * k + i, a * k + i + 1, (a + 1) * k + i + 1, (a + 1) * k + i
                faces += [(p, r, q), (p, t, r)]
        faces = np.array(faces, int)
        sugar = (pts, faces, vertex_normals(pts, faces))
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(flakes):
            v = rng.uniform(-0.62, 0.62)
            ring = self._section(v, 72, True)
            i = int(rng.integers(8, 29))  # 40 to 140 degrees round the outline: the top
            base = ring[i]
            c = ring.mean(axis=0)
            nrm = base - c
            nrm[0] = 0.0
            nrm = nrm / np.linalg.norm(nrm)
            yaw = rng.uniform(0, math.pi)
            t1 = np.array([math.cos(yaw), 0.0, 0.0]) + math.sin(yaw) * np.cross(nrm, [1.0, 0.0, 0.0])
            t1 = t1 - (t1 @ nrm) * nrm
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(nrm, t1)
            out.append(_flake(base + 0.0016 * nrm, t1, t2, nrm))
        return sugar, out


def _flake(centre, t1, t2, nrm, a=0.008, b=0.0045, c=0.0008):
    """A sliced almond: a flat ellipsoid with half-axes a, b, c along t1, t2, nrm."""
    rings = []
    for lat in np.linspace(-1.1, 1.1, 5):
        r, z = math.cos(lat), math.sin(lat)
        phi = np.linspace(0.0, 2 * np.pi, 10, endpoint=False)
        rings.append(centre + np.outer(a * r * np.cos(phi), t1) + np.outer(b * r * np.sin(phi), t2) + c * z * nrm)
    pts, faces = loft(rings)
    return pts, faces, vertex_normals(pts, faces)


def lathe(profile, segments: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a closed (radius, z) outline about z: (points, triangles, normals). Give the outline counter-clockwise
    in the (radius, z) half-plane, radius >= 0; points on the axis (radius 0) are allowed."""
    prof = np.asarray(profile, float)
    th = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    pts = np.vstack([np.stack([r * np.cos(th), r * np.sin(th), np.full(segments, z)], axis=1) for r, z in prof])
    m = len(prof)
    faces = []
    for a in range(m):
        b = (a + 1) % m
        for i in range(segments):
            j = (i + 1) % segments
            p, q, r, s = a * segments + i, a * segments + j, b * segments + j, b * segments + i
            if prof[a][0] > 1e-9:
                faces.append((p, q, r))
            if prof[b][0] > 1e-9:
                faces.append((p, r, s))
    faces = np.array(faces, int)
    return pts, faces, vertex_normals(pts, faces)


# -- convex pieces in the world and the gap between bodies --------------------------------------------------------


def box_points(size, spacing: float = 0.004) -> np.ndarray:
    """Points on a box's surface (its own frame, centred), no more than ``spacing`` apart."""
    half = np.asarray(size, float) / 2
    axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / spacing)) + 1)) for h in half]
    out = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        gb, gc = np.meshgrid(axes[b], axes[c], indexing="ij")
        for sign in (-1.0, 1.0):
            face = np.zeros((gb.size, 3))
            face[:, a] = sign * half[a]
            face[:, b] = gb.ravel()
            face[:, c] = gc.ravel()
            out.append(face)
    return np.unique(np.round(np.vstack(out), 7), axis=0)


def hull_points(points, faces, spacing: float = 0.005) -> np.ndarray:
    """A hull's vertices plus points spread over its faces, no more than about ``spacing`` apart."""
    p, f = np.asarray(points, float), np.asarray(faces, int)
    out = [p]
    for tri in f:
        a, b, c = p[tri]
        n = int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing))
        if n < 2:
            continue
        for i in range(n + 1):
            for j in range(n + 1 - i):
                out.append((a + (b - a) * (i / n) + (c - a) * (j / n))[None])
    return np.unique(np.round(np.vstack(out), 7), axis=0)


@dataclass
class Piece:
    """One convex piece of a body in the world: a box (centre, rotation, size) or a hull (vertices, planes), with
    points on its surface and the box around it."""

    kind: str
    points: np.ndarray  # surface points, world
    low: np.ndarray
    high: np.ndarray
    centre: np.ndarray | None = None
    rot: np.ndarray | None = None
    size: np.ndarray | None = None
    planes: np.ndarray | None = None  # world: n . x = d
    tag: str = ""

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance of points to this piece (negative inside)."""
        if self.kind == "box":
            q = np.abs((pts - self.centre) @ self.rot) - self.size / 2
            return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
        return (pts @ self.planes[:, :3].T - self.planes[:, 3]).max(axis=1)


def world_box(centre, rot, size, local_points, tag="") -> Piece:
    centre, rot, size = np.asarray(centre, float), np.asarray(rot, float), np.asarray(size, float)
    pts = local_points @ rot.T + centre
    return Piece("box", pts, pts.min(axis=0), pts.max(axis=0), centre=centre, rot=rot, size=size, tag=tag)


def world_hull(origin, rot, local_points, local_planes, tag="") -> Piece:
    origin, rot = np.asarray(origin, float), np.asarray(rot, float)
    pts = local_points @ rot.T + origin
    n = local_planes[:, :3] @ rot.T
    d = local_planes[:, 3] + n @ origin
    return Piece("hull", pts, pts.min(axis=0), pts.max(axis=0), planes=np.c_[n, d], tag=tag)


def gap(a: list[Piece], b: list[Piece], reach: float = 0.02) -> float:
    """The gap (m) between two bodies given as convex pieces: negative when they overlap, and ``inf`` when every
    pair of pieces is more than ``reach`` apart (only pairs whose boxes come that close are measured)."""
    best = math.inf
    for pa in a:
        for pb in b:
            if np.any(pa.low - pb.high > reach) or np.any(pb.low - pa.high > reach):
                continue
            near = pa.points[np.all((pa.points >= pb.low - reach) & (pa.points <= pb.high + reach), axis=1)]
            if len(near):
                best = min(best, float(pb.distance(near).min()))
            near = pb.points[np.all((pb.points >= pa.low - reach) & (pb.points <= pa.high + reach), axis=1)]
            if len(near):
                best = min(best, float(pa.distance(near).min()))
    return best


def extent(pieces: list[Piece]) -> tuple[np.ndarray, np.ndarray]:
    return np.min([p.low for p in pieces], axis=0), np.max([p.high for p in pieces], axis=0)
