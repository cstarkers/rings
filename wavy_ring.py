"""Wavy ring: finger along +Z, circular in XY.

Top and bottom edges follow a 3-lobe sine wave, each with its own
independent, seamless "natural" noise. Built as a mesh with numpy and
turned into a solid with manifold3d, so you can union or subtract other shapes.

    pip install manifold3d numpy trimesh fonttools
    python wavy_ring.py                    # writes wavy_ring.stl, for viewing on screen
    python wavy_ring.py --with-support     # writes wavy_ring_support.stl, standing on a pedestal
    python wavy_ring.py --all-interesting  # writes wavy_rings_all.stl, every worth_printing ring
    python wavy_ring.py --decode CODE      # turn an engraved code back into settings

Each ring is engraved with a code that packs every IMPORTANT_DETAILS value into a
few FDM-friendly characters (see encode_params), and decodes back exactly.
"""
import argparse
import math
from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from manifold3d import CrossSection, FillRule, JoinType, Manifold, Mesh
import trimesh


@dataclass
class RingParams:
    # Size
    inner_d: float = 21.3      
    thickness: float = 1.8     # radial wall thickness (mm)
    n: int = 360               # slices around the ring

    # Main wave
    wave_count: int = 4        # wavelengths per revolution
    wave_amp: float = 1.2      # sine amplitude (mm)
    wave_phase: float = 0.0    # rotate the wave (radians)
    wave_shape: float = 0.0    # 0 = sine; >0 rounds the outward tips (top peaks, bottom troughs)
                               # and tightens the inward notches. Around 1-3 is a strong effect.
    base_width: float = 5.0    # average band width (mm)
    min_width: float = 2.5     # never thinner than this (mm)

    # Fillets on the cross-section corners (radius, mm).
    # Outer = the two corners on the outside face, inner = the two against the finger.
    outer_fillet: float = 0.9
    inner_fillet: float = 0.7
    fillet_segments: int = 8   # points per quarter-circle arc

    # Noise
    top_noise_amp: float = 1.0
    bottom_noise_amp: float = 1.0
    top_seed: int = 11
    bottom_seed: int = 47
    h_min: int = 1             # lowest noise harmonic (1 = once round the ring, 2 = twice)
    h_max: int = 14            # highest harmonic (higher = finer wobble)
    falloff: float = 2.0       # how fast higher harmonics fade: 1 = old 1/k, higher = smoother

    # Thickness variation: the inner (skin) surface stays a perfect circle and only
    # the outer face bulges outwards. `thickness` above is the thinnest point.
    thickness_variation: float = 0.8  # extra thickness at the fattest point (mm)
    thickness_seed: int = 29

    # Inscription: the short code engraved on the inner (skin) face, readable from
    # inside the ring. It follows the wavy centre line of the band, and is sized to
    # the largest cap height that fits round the ring and across the inner face.
    inscription_max_height: float = 3.0  # cap height never exceeds this (mm)
    inscription_coverage: float = 0.92   # fraction of the inner circumference the text may use
    inscription_margin: float = 0.3      # min clearance to the inner fillets (mm)
    inscription_depth: float = 0.4       # engraving depth into the wall (mm)
    inscription_bold: float = 0.0        # widen the grooves by this much each side (mm). On
                                         # FDM the grooves stay open anyway; widening shrinks
                                         # the holes in 0, 8 etc. until they fill in.
    inscription_tracking: float = 0.0    # extra gap between characters (mm); use 2 x bold
                                         # to stop bold from fusing neighbouring characters
    inscription_angle: float = -np.pi / 2  # angle the text is centred on (radians); the gap
                                         # in the text then faces +Y, so set the slicer's
                                         # seam position to "Back" to keep the seam off it

    # Print pedestal (only with --with-support): the ring prints standing up, and its
    # wavy bottom edge rests on a thin wall that follows it down to a flat base.
    # The wall snaps off afterwards.
    pedestal_gap: float = 0.2      # air gap under the ring; set to one layer height (mm)
    pedestal_wall: float = 0.8     # wall width, about two nozzle widths (mm)
    pedestal_min_height: float = 1.0  # wall height at the ring's lowest point (mm)
    pedestal_base: float = 0.6     # thickness of the flat base on the bed (mm)
    pedestal_flange: float = 2.0   # base sticks out this far either side of the ring (mm)

    IMPORTANT_DETAILS = [
        "inner_d",
        "thickness",
        "wave_count",
        "wave_amp",
        "wave_shape",
        "base_width",
        "outer_fillet",
        "inner_fillet",
        "top_noise_amp",
        "bottom_noise_amp",
        "top_seed",
        "bottom_seed",
        "h_min",
        "h_max",
        "falloff",
        "thickness_variation",
        "thickness_seed",
    ]

    @property
    def identifier(self) -> str:
        #v1 versions the identifier so we can decode it against IMPORTANT_DETAILS
        return "v1|"+ "|".join(str(getattr(self, important_detail)) for important_detail in self.IMPORTANT_DETAILS)

    @property
    def short_code(self) -> str:
        """The engraved code: encode_params(self)."""
        return encode_params(self)


# Engraved code. Each IMPORTANT_DETAILS value is stored as a whole number of steps
# from the bottom of its range, and the fields are packed into one big integer
# (mixed radix: each field takes exactly as many values as its range needs), which
# is then written in base 17. The first character is the scheme version.
#
# The alphabet leaves out characters with enclosed holes (0 4 6 8 9 A B D O P Q R)
# and look-alikes (1 I, 2 Z, 5 S, G), which FDM printing blurs together.
#
# Changing any range, step or the field order breaks old codes: bump CODE_VERSION
# and keep the old table around to decode them.
CODE_ALPHABET = "CEFHJKLMNTUVWXY37"
CODE_VERSION = 1
CODE_GROUP = 5  # hyphenate into groups of at most this many characters, for reading
CODE_RANGES = {  # field: (lowest, highest, step)
    "inner_d": (14, 24, 0.1),
    "thickness": (1, 3, 0.1),
    "wave_count": (1, 8, 1),
    "wave_amp": (0, 3, 0.1),
    "wave_shape": (-3, 3, 0.1),
    "base_width": (2, 10, 0.1),
    "outer_fillet": (0, 5, 0.1),
    "inner_fillet": (0, 3, 0.1),
    "top_noise_amp": (0, 5, 0.1),
    "bottom_noise_amp": (0, 5, 0.1),
    "top_seed": (0, 999, 1),
    "bottom_seed": (0, 999, 1),
    "h_min": (1, 8, 1),
    "h_max": (1, 32, 1),
    "falloff": (0, 4, 0.1),
    "thickness_variation": (0, 3, 0.1),
    "thickness_seed": (0, 999, 1),
}
assert list(CODE_RANGES) == RingParams.IMPORTANT_DETAILS


def _levels(lo, hi, step) -> int:
    return round((hi - lo) / step) + 1


CODE_DIGITS = math.ceil(math.log(math.prod(_levels(*r) for r in CODE_RANGES.values()),
                                 len(CODE_ALPHABET)))


def encode_params(p: RingParams) -> str:
    """Pack p's IMPORTANT_DETAILS into a short code. Raises if a value is out of
    range or not a whole number of steps, since it would not decode exactly."""
    n = 0
    for name, (lo, hi, step) in CODE_RANGES.items():
        value = getattr(p, name)
        i = round((value - lo) / step)
        if not 0 <= i < _levels(lo, hi, step) or abs(lo + i * step - value) > 1e-6:
            raise ValueError(f"{name}={value} can't be encoded: it must be {lo} to {hi} "
                             f"in steps of {step} (see CODE_RANGES)")
        n = n * _levels(lo, hi, step) + i
    digits = ""
    for _ in range(CODE_DIGITS):
        n, d = divmod(n, len(CODE_ALPHABET))
        digits = CODE_ALPHABET[d] + digits
    code = CODE_ALPHABET[CODE_VERSION] + digits
    # Split into equal-ish groups of at most CODE_GROUP, longest first.
    groups = math.ceil(len(code) / CODE_GROUP)
    sizes = [len(code) // groups + (g < len(code) % groups) for g in range(groups)]
    cuts = np.cumsum([0] + sizes)
    return "-".join(code[a:b] for a, b in zip(cuts[:-1], cuts[1:]))


def decode_params(code: str) -> RingParams:
    """Inverse of encode_params. Fields not in the code keep their defaults."""
    code = code.upper().replace("-", "").replace(" ", "")
    if len(code) != CODE_DIGITS + 1 or any(c not in CODE_ALPHABET for c in code):
        raise ValueError(f"{code!r} is not a {CODE_DIGITS + 1}-character code")
    if CODE_ALPHABET.index(code[0]) != CODE_VERSION:
        raise ValueError(f"code version {CODE_ALPHABET.index(code[0])} isn't supported")
    n = 0
    for c in code[1:]:
        n = n * len(CODE_ALPHABET) + CODE_ALPHABET.index(c)
    types = {f.name: f.type for f in fields(RingParams)}
    values = {}
    for name, (lo, hi, step) in reversed(CODE_RANGES.items()):
        n, i = divmod(n, _levels(lo, hi, step))
        if types[name] is int:
            values[name] = int(lo + i * step)
        else:
            values[name] = round(lo + i * step, 6)
    if n:
        raise ValueError(f"{code!r} is not a valid code")
    return RingParams(**values)


FONT_PATH = Path(__file__).parent / "fonts" / "NotoSans-Bold.ttf"
CURVE_SEGMENTS = 6  # straight segments per font curve


class _PolygonPen(BasePen):
    """Collects glyph outlines as polylines, flattening the curves."""

    def __init__(self, glyph_set, offset_x: float):
        super().__init__(glyph_set)
        self.dx = offset_x
        self.contours: list[list[tuple[float, float]]] = []

    def _add(self, pt):
        self.contours[-1].append((pt[0] + self.dx, pt[1]))

    def _moveTo(self, pt):
        self.contours.append([])
        self._add(pt)

    def _lineTo(self, pt):
        self._add(pt)

    def _curveToOne(self, p1, p2, p3):
        p0 = np.array(self._getCurrentPoint())
        p1, p2, p3 = map(np.array, (p1, p2, p3))
        for t in np.linspace(0, 1, CURVE_SEGMENTS + 1)[1:]:
            self._add((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
                      + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)

    def _qCurveToOne(self, p1, p2):
        p0 = np.array(self._getCurrentPoint())
        p1, p2 = np.array(p1), np.array(p2)
        for t in np.linspace(0, 1, CURVE_SEGMENTS + 1)[1:]:
            self._add((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2)


@cache
def _font() -> TTFont:
    return TTFont(FONT_PATH)


def text_outline(text: str, tracking: float = 0.0) -> CrossSection:
    """Text as a flat outline, baseline on y=0, starting at x=0, cap height 1.

    `tracking` is extra space after each character, in cap heights.
    """
    font = _font()
    cap = font["OS/2"].sCapHeight
    cmap, glyph_set = font.getBestCmap(), font.getGlyphSet()
    contours, x = [], 0.0
    for ch in text:
        if ord(ch) not in cmap:
            raise ValueError(f"font has no glyph for {ch!r} in inscription {text!r}")
        glyph = glyph_set[cmap[ord(ch)]]
        pen = _PolygonPen(glyph_set, x)
        glyph.draw(pen)
        contours += pen.contours
        x += glyph.width + tracking * cap
    outline = CrossSection([np.array(c) for c in contours if len(c) >= 3], FillRule.NonZero)
    return outline.scale((1 / cap,) * 2)


def fit_inscription(p: RingParams) -> tuple[CrossSection, float]:
    """Outline of p.short_code in mm, centred on the origin, and its cap height.

    The cap height is the largest that fits both round the ring and across the
    narrowest part of the inner face, capped at p.inscription_max_height.
    """
    x0, y0, x1, y1 = text_outline(p.short_code).bounds()
    ri = p.inner_d / 2

    # The flat part of the inner face, between the inner fillets as wavy_ring builds them.
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    z_bot, z_top = edge_heights(p, theta)
    _, _, fi, _ = wall_radii(p, theta)
    fzi = np.minimum(fi, (z_top - z_bot) * 0.45)
    flat = (z_top - z_bot - 2 * fzi).min() - 2 * p.inscription_margin
    room = 2 * np.pi * ri * p.inscription_coverage

    # The bold offset adds 2 * bold to both the length and the height; tracking adds
    # a fixed length per gap between characters, whatever the text size.
    gaps = (len(p.short_code) - 1) * p.inscription_tracking
    h = min(p.inscription_max_height,
            (room - 2 * p.inscription_bold - gaps) / (x1 - x0),
            (flat - 2 * p.inscription_bold) / (y1 - y0))
    if h <= 0:
        raise ValueError("no room for the inscription on this ring")

    unit = text_outline(p.short_code, p.inscription_tracking / h)
    x0, y0, x1, y1 = unit.bounds()
    text = unit.translate((-(x0 + x1) / 2, -(y0 + y1) / 2)).scale((h, h))
    if p.inscription_bold:
        text = text.offset(p.inscription_bold, JoinType.Round)
    return text, h


def inscription_negative(p: RingParams) -> Manifold:
    """Solid to subtract from the ring to engrave p.short_code on the inner face."""
    ri = p.inner_d / 2
    overcut = 0.2  # poke into the finger hole so the cut is clean
    text, _ = fit_inscription(p)

    # The cross-section is built in (u, v) and extruded in -w, so that the warp
    # below (which flips handedness) keeps the normals pointing outwards.
    solid = text.extrude(p.inscription_depth + overcut).translate((0, 0, -p.inscription_depth))
    solid = solid.refine_to_length(1.0)  # bow on the curve stays ~0.01 mm

    def wrap(pts: np.ndarray) -> np.ndarray:
        u, v, w = pts[:, 0], pts[:, 1], pts[:, 2]
        # Viewed from inside the ring, +u runs towards decreasing theta.
        theta = p.inscription_angle - u / ri
        zb, zt = edge_heights(p, theta)
        r = ri - w
        return np.column_stack([r * np.cos(theta), r * np.sin(theta), (zb + zt) / 2 + v])

    return solid.warp_batch(wrap)


def edge_noise(theta: np.ndarray, seed: int, h_min: int, h_max: int,
               falloff: float = 1.0) -> np.ndarray:
    """Seamless noise: integer harmonics with random weights and phases.

    Harmonic k is scaled by (h_min / k) ** falloff relative to the lowest one, so
    the lowest harmonic keeps the same strength whatever the falloff, and a larger
    falloff rolls off the fast wiggles. falloff=1 is a plain 1/k spectrum.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(h_min, h_max + 1)
    weights = rng.random(len(k)) / h_min * (h_min / k) ** falloff
    phases = rng.random(len(k)) * 2 * np.pi
    raw = (weights[:, None] * np.sin(k[:, None] * theta + phases[:, None])).sum(axis=0)
    return raw / (1 / k).sum() * 2


def wave_profile(s: np.ndarray, shape: float) -> np.ndarray:
    """Reshape a sine s in [-1, 1] to another curve in [-1, 1], same peaks and troughs.

    Uses 1 - exp(-a (1 + s)), rescaled: an exponential that flattens towards the
    peaks and steepens towards the troughs, like a hanging chain turned upside
    down. Tends to the plain sine as shape -> 0, and stays smooth everywhere.
    """
    if abs(shape) < 1e-9:
        return s
    return 2 * np.expm1(-shape * (1 + s)) / np.expm1(-2 * shape) - 1


def edge_heights(p: RingParams, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Opposite shaping on each edge, so the tips pointing away from the band are the
    # broad, rounded ones on both edges and the tight curves are the inward notches.
    s = np.sin(p.wave_count * theta + p.wave_phase)
    top_wave = p.wave_amp * wave_profile(s, p.wave_shape)
    bot_wave = p.wave_amp * wave_profile(s, -p.wave_shape)
    z_bot = (-p.base_width / 2 + bot_wave
             + p.bottom_noise_amp * edge_noise(theta, p.bottom_seed, p.h_min, p.h_max, p.falloff))
    z_top = (p.base_width / 2 + top_wave
             + p.top_noise_amp * edge_noise(theta, p.top_seed, p.h_min, p.h_max, p.falloff))
    z_top = np.maximum(z_top, z_bot + p.min_width)
    return z_bot, z_top


def wall_radii(p: RingParams, theta: np.ndarray):
    """Inner and outer wall radius and the inner and outer fillet radii at each angle."""
    ri = p.inner_d / 2
    bulge = edge_noise(theta, p.thickness_seed, p.h_min, p.h_max, p.falloff)
    bulge = (bulge - bulge.min()) / np.ptp(bulge)       # 0 at thinnest, 1 at fattest
    thickness = p.thickness + p.thickness_variation * bulge
    ro = ri + thickness

    # Where the wall thins, shrink both fillets together so they never overlap.
    shrink = np.minimum(1.0, 0.9 * thickness / (p.outer_fillet + p.inner_fillet))
    fo, fi = p.outer_fillet * shrink, p.inner_fillet * shrink
    return ri, ro, fi, fo


def sweep(theta: np.ndarray, profile: list[tuple]) -> Manifold:
    """Solid swept round the Z axis through a cross-section that changes with angle.

    `profile` is a closed loop of (radius, z) points, counter-clockwise in the
    (r, z) plane; each coordinate may be a scalar or an array over theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    one = np.ones_like(theta)
    P = len(profile)
    verts = np.stack([np.column_stack([r * c, r * s, z * one])
                      for r, z in profile], axis=1).reshape(-1, 3)

    # Quads joining slice i to slice i+1, split into two triangles each.
    # Winding is counter-clockwise from outside, as Manifold expects.
    n = len(theta)
    i = np.repeat(np.arange(n), P)
    k = np.tile(np.arange(P), n)
    j = (i + 1) % n
    a, b = i * P + k, i * P + (k + 1) % P
    cc, d = j * P + (k + 1) % P, j * P + k
    tris = np.concatenate([np.column_stack([a, d, cc]),
                           np.column_stack([a, cc, b])])

    mesh = Mesh(vert_properties=verts.astype(np.float32),
                tri_verts=tris.astype(np.uint32))
    solid = Manifold(mesh)
    if solid.status().name != "NoError":
        raise RuntimeError(f"Manifold build failed: {solid.status()}")
    return solid


def wavy_ring(p: RingParams = RingParams()) -> Manifold:
    theta = np.linspace(0, 2 * np.pi, p.n, endpoint=False)
    z_bot, z_top = edge_heights(p, theta)
    ri, ro, fi, fo = wall_radii(p, theta)
    # Where the band is narrow, squash the arcs vertically so they never cross.
    half = (z_top - z_bot) * 0.45
    fzo, fzi = np.minimum(fo, half), np.minimum(fi, half)

    # Cross-section profile as (radius, z) pairs: one quarter-arc per corner,
    # going counter-clockwise: outer-top, inner-top, inner-bottom, outer-bottom.
    # Each corner: (arc centre r, arc centre z, radial radius, vertical radius, start angle)
    corners = [
        (ro - fo, z_top - fzo, fo, fzo,   0),
        (ri + fi, z_top - fzi, fi, fzi,  90),
        (ri + fi, z_bot + fzi, fi, fzi, 180),
        (ro - fo, z_bot + fzo, fo, fzo, 270),
    ]
    profile = []
    for cr, cz, rr, rz, start in corners:
        for phi in np.radians(np.linspace(start, start + 90, p.fillet_segments + 1)):
            profile.append((cr + rr * np.cos(phi), cz + rz * np.sin(phi)))
    return sweep(theta, profile)


def pedestal(p: RingParams) -> Manifold:
    """Breakaway print support: a flat base plus a thin wall up to the bottom edge.

    The wall sits under the flat strip between the two bottom fillets (the
    lowest part of the cross-section) and stops p.pedestal_gap short of it.
    """
    theta = np.linspace(0, 2 * np.pi, p.n, endpoint=False)
    z_bot, _ = edge_heights(p, theta)
    ri, ro, fi, fo = wall_radii(p, theta)
    mid = (ri + fi + ro - fo) / 2
    w = p.pedestal_wall / 2

    z_wall_top = z_bot - p.pedestal_gap
    z_base_top = z_wall_top.min() - p.pedestal_min_height
    z_base_bot = z_base_top - p.pedestal_base
    overlap = p.pedestal_base / 2  # sink the wall into the base so they fuse

    wall = sweep(theta, [(mid - w, z_base_top - overlap), (mid + w, z_base_top - overlap),
                         (mid + w, z_wall_top), (mid - w, z_wall_top)])
    r_out = ro.max() + p.pedestal_flange
    r_in = max(ri - p.pedestal_flange, 1.0)
    segments = p.n // 2
    base = (Manifold.cylinder(p.pedestal_base, r_out, circular_segments=segments)
            - Manifold.cylinder(p.pedestal_base, r_in, circular_segments=segments))
    return wall + base.translate((0, 0, z_base_bot))


def save_stl(m: Manifold, path: str) -> None:
    out = m.to_mesh()
    trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts).export(path)


worth_printing: list[RingParams] = [
    RingParams(base_width=6.5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=1.4,  top_seed=10, wave_shape=1), # this
    RingParams(base_width=6.5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, top_seed=10, wave_shape=0.8),
    RingParams(base_width=5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, top_seed=10),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, top_seed=10, falloff=1.4),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, bottom_noise_amp=3, top_noise_amp=3, falloff=1.7),
    RingParams(base_width=6.5, wave_count=4, outer_fillet=2, inner_fillet=1.2, wave_amp=0.7, bottom_noise_amp=3, top_noise_amp=3),
    RingParams(base_width=6.5, wave_count=4, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, bottom_noise_amp=3, top_noise_amp=3),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, bottom_noise_amp=3),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=4, inner_fillet=1.2, wave_amp=0.7, top_seed=10),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=2, wave_amp=0.7, top_seed=10),
    RingParams(base_width=6.5, wave_count=4, outer_fillet=2,wave_amp=0.5),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=2, thickness_variation=1.3),
    RingParams(base_width=6.5, wave_count=3, outer_fillet=2),
    RingParams(wave_count=3),
    RingParams(),
    RingParams(base_width=6.5, wave_count=4, outer_fillet=2),
]

def printable(p: RingParams, with_support: bool) -> Manifold:
    """The engraved ring, optionally on its pedestal, resting on z = 0."""
    ring = wavy_ring(p) - inscription_negative(p)
    if with_support:
        ring = ring + pedestal(p)
    return ring.translate((0, 0, -ring.bounding_box()[2]))


def layout(parts: list[Manifold], spacing: float = 4.0) -> Manifold:
    """Arrange parts in a square-ish grid on the XY plane, in list order.

    The first part is at the front left, then left to right, then row by row
    towards the back (+Y).
    """
    cols = int(np.ceil(np.sqrt(len(parts))))
    cell = max(max(b[3] - b[0], b[4] - b[1]) for b in (m.bounding_box() for m in parts)) + spacing
    placed = []
    for idx, m in enumerate(parts):
        b = m.bounding_box()
        centre = ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2)
        x, y = (idx % cols) * cell, (idx // cols) * cell
        placed.append(m.translate((x - centre[0], y - centre[1], 0)))
    return Manifold.compose(placed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--with-support", action="store_true",
                        help="add the breakaway print pedestal under each ring")
    parser.add_argument("--pedestal-gap", type=float, metavar="MM",
                        help="air gap between ring and pedestal; match the slicer's layer "
                             f"height (default {RingParams.pedestal_gap})")
    parser.add_argument("--all-interesting", action="store_true",
                        help="lay out every ring in worth_printing in one STL")
    parser.add_argument("--decode", metavar="CODE",
                        help="print the settings encoded in an engraved code, and exit")
    args = parser.parse_args()

    if args.decode:
        decoded = decode_params(args.decode)
        print(f"RingParams({', '.join(f'{k}={getattr(decoded, k)}' for k in CODE_RANGES)})")
        raise SystemExit

    rings = worth_printing if args.all_interesting else worth_printing[:1]
    if args.pedestal_gap is not None:
        rings = [replace(p, pedestal_gap=args.pedestal_gap) for p in rings]

    parts = []
    for idx, params in enumerate(rings):
        print(f"[{idx:2}] {params.short_code}  {params.identifier}  "
              f"(cap height {fit_inscription(params)[1]:.2f} mm)")
        parts.append(printable(params, args.with_support))
    out = layout(parts) if args.all_interesting else parts[0]
    suffix = "_support" if args.with_support else ""
    path = f"wavy_rings_all{suffix}.stl" if args.all_interesting else f"wavy_ring{suffix}.stl"

    b = out.bounding_box()
    print(f"{out.num_tri()} triangles, footprint {b[3] - b[0]:.0f} x {b[4] - b[1]:.0f} mm, "
          f"height {b[5] - b[2]:.1f} mm")
    save_stl(out, path)
    print(f"wrote {path}")
