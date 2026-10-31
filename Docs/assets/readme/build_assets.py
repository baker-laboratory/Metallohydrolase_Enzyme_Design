#!/usr/bin/env python3
"""Regenerate the README's self-contained SVG artwork.

    uv run --no-project --python 3.12 --with fonttools --with uharfbuzz \
        --with numpy --with biotite==2.0.0 python -I Docs/assets/readme/build_assets.py

Type is converted to outlines so the art renders identically without web fonts.
Mona Sans (SIL OFL) comes from a pinned google/fonts commit. The masthead uses
the checksum-verified ZAPP-1 model from the repository deposit, with P-SEA
secondary structure assigned only for drawing. The unchanged pipeline uses
1QJI from a live RCSB download; fonts and 1QJI are cached outside the repository.
Edit the content tables at the end of this file.

The pipeline art is schematic. Its noise cloud, overlaid trace and plate wells
are seeded or drawn illustrations: not model outputs, AlphaFold results or data.
"""
import argparse, hashlib, io, math, pathlib, urllib.request, zipfile
import numpy as np
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from xml.sax.saxutils import escape

OUT = pathlib.Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ZAPP1_ARCHIVE = ROOT / "Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/supplemental_data/supp_data__denovo_PTE_design_models.zip"
FONT_COMMIT = "6157dfaa640ca5738b4d1a7070fe389c3036d7fc"
FONT_BASE = f"https://raw.githubusercontent.com/google/fonts/{FONT_COMMIT}/ofl"
SOURCES = {
    "MonaSans.ttf": f"{FONT_BASE}/monasans/MonaSans%5Bwdth,wght%5D.ttf",
    "1QJI.pdb": "https://files.rcsb.org/download/1QJI.pdb",
}

# IPD identity colours and the dark substrate that makes them read as a thin film.
BLUE, PINK, PURPLE = "#4D97B7", "#B11186", "#4B2E83"
INK = "#0C0A1A"
WHITE, SOFT, MUTED = "#F7F5FF", "#DAD4EE", "#A9A2C6"
PEARL = [(0, "#BDE6F6"), (0.48, "#F6BEE6"), (1, "#D2C4FB")]


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def hexc(c):
    return "#%02X%02X%02X" % tuple(int(round(max(0, min(255, v)))) for v in c)


def ramp(stops, t):
    t = max(0.0, min(1.0, t))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            return rgb(ca) + (rgb(cb) - rgb(ca)) * ((t - a) / (b - a))
    return rgb(stops[-1][1])


def num(v, prec=1):
    return (f"{v:.{prec}f}".rstrip("0").rstrip(".") if prec else str(int(round(v)))) or "0"


# ---------------------------------------------------------------- type
class Type:
    fonts = {}

    def __init__(self, file, **axes):
        key = (file, tuple(sorted(axes.items())))
        if key not in Type.fonts:
            font = TTFont(CACHE / file)
            font = instancer.instantiateVariableFont(font, axes, inplace=False)
            buf = io.BytesIO()
            font.save(buf)
            Type.fonts[key] = (font, hb.Font(hb.Face(buf.getvalue())))
        self.font, self.hb = Type.fonts[key]
        self.upem = self.font["head"].unitsPerEm

    def layout(self, text, size, tracking=0.0):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        names, s, x, glyphs = self.font.getGlyphOrder(), size / self.upem, 0.0, []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            glyphs.append((names[info.codepoint], x + pos.x_offset * s, pos.y_offset * s))
            x += pos.x_advance * s + tracking * size
        return glyphs, x - tracking * size, s

    def width(self, text, size, tracking=0.0):
        return self.layout(text, size, tracking)[1]

    def d(self, text, size, x=0, y=0, anchor="start", tracking=0.0, prec=2):
        glyphs, width, s = self.layout(text, size, tracking)
        x -= width if anchor == "end" else width / 2 if anchor == "middle" else 0
        shapes = self.font.getGlyphSet()
        pen = SVGPathPen(shapes, ntos=lambda v: num(v, prec))
        for name, gx, gy in glyphs:
            shapes[name].draw(TransformPen(pen, (s, 0, 0, -s, x + gx, y - gy)))
        return pen.getCommands()

    def path(self, text, size, x=0, y=0, fill=WHITE, anchor="start", tracking=0.0, prec=2):
        return f'<path fill="{fill}" d="{self.d(text, size, x, y, anchor, tracking, prec)}"/>'


def svg(width, height, label, body, defs="", vb=None):
    vb = vb or (width, height)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {vb[0]} {vb[1]}" role="img" aria-label="{escape(label)}">'
        f"<title>{escape(label)}</title><defs>{defs}</defs>{body}</svg>\n"
    )


def gradient(id_, stops, x1=0, y1=0, x2=1, y2=0, units=None):
    u = f' gradientUnits="{units}"' if units else ""
    s = "".join(f'<stop offset="{num(o, 3)}" stop-color="{c}"/>' for o, c in stops)
    return f'<linearGradient id="{id_}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{u}>{s}</linearGradient>'


def glow(id_, color, opacity):
    return (f'<radialGradient id="{id_}"><stop offset="0" stop-color="{color}" stop-opacity="{opacity}"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>')


# ---------------------------------------------------------------- structure
class Structure:
    """Backbone, secondary structure and active site of 1QJI, chain A."""

    def __init__(self, file):
        self.ca, self.o, self.ss, self.res, self.seq = {}, {}, {}, {}, {}
        self.zn, self.tsa = None, []
        for ln in pathlib.Path(file).read_text(encoding="utf-8").splitlines():
            rec = ln[:6].strip()
            if rec in ("ATOM", "HETATM") and ln[16] in " A" and ln[21] == "A":
                name, resn, num_ = ln[12:16].strip(), ln[17:20].strip(), int(ln[22:26])
                xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
                el = ln[76:78].strip() or name[0]
                if rec == "ATOM":
                    self.res.setdefault(num_, {})[name] = xyz
                    self.seq[num_] = resn
                    if name == "CA":
                        self.ca[num_] = xyz
                    if name == "O":
                        self.o[num_] = xyz
                elif resn == "ZN":
                    self.zn = xyz
                elif resn == "PKF" and el != "H":
                    self.tsa.append((name, el, xyz))
            elif rec == "HELIX":
                a, b = int(ln[21:25]), int(ln[33:37])
                if b - a >= 4:
                    self.ss.update({i: "H" for i in range(a, b + 1)})
            elif rec == "SHEET":
                a, b = int(ln[22:26]), int(ln[33:37])
                self.ss.update({i: "E" for i in range(a, b + 1)})
        self.ids = sorted(i for i in self.ca if i in self.o)
        self.centroid = np.mean([self.ca[i] for i in self.ids], axis=0)

    def rotation(self, roll=0.0, tilt=0.0, yaw=0.0):
        """Look down the centroid-to-zinc axis so the active-site cleft faces the viewer."""
        z = self.zn - self.centroid
        z /= np.linalg.norm(z)
        u = np.array([0.0, 1.0, 0.0]) - z * z[1]
        u /= np.linalg.norm(u)
        base = np.array([np.cross(u, z), u, z])
        c, s = math.cos, math.sin
        r, t, w = map(math.radians, (roll, tilt, yaw))
        Rz = np.array([[c(r), -s(r), 0], [s(r), c(r), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, c(t), -s(t)], [0, s(t), c(t)]])
        Ry = np.array([[c(w), 0, s(w)], [0, 1, 0], [-s(w), 0, c(w)]])
        return Ry @ Rx @ Rz @ base

    def ribbon_frames(self, per=4):
        """Carson-style guide: spline through peptide midpoints with peptide-plane sides."""
        ids, ca, o = self.ids, self.ca, self.o
        mids, sides, kinds = [], [], []
        prev = None
        for a, b in zip(ids, ids[1:]):
            axis = ca[b] - ca[a]
            normal = np.cross(axis, o[a] - ca[a])
            normal /= np.linalg.norm(normal)
            side = np.cross(normal, axis)
            side /= np.linalg.norm(side)
            if prev is not None and side @ prev < 0:
                side = -side
            prev = side
            kind = self.ss.get(a, "L")
            mid = (ca[a] + ca[b]) / 2 + (normal * 1.3 if kind == "H" else 0)
            mids.append(mid), sides.append(side), kinds.append(kind)
        mids, sides = np.array(mids), np.array(sides)
        P = np.vstack([mids[0], mids, mids[-1]])
        S = np.vstack([sides[0], sides, sides[-1]])
        out = []
        for i in range(len(mids) - 1):
            for k in range(per):
                u = k / per
                b = np.array([(1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3]) / 6
                pos = b @ P[i:i + 4]
                side = b @ S[i:i + 4]
                out.append((pos, side / (np.linalg.norm(side) or 1), kinds[i], i, u))
        return out

    def trace(self):
        return np.array([self.ca[i] for i in self.ids])


ZAPP1_MEMBER = (
    "all_design_models/"
    "denovo_PTE__design_campaign1__037_scaffold_03____Plate1__D1__ZAPP1.pdb"
)
ZAPP1_ARCHIVE_SHA256 = "772a498af2cd4bed16b5eed3cdf602799b9e2bdbc32dbee13d78722ecf2b066c"
ZAPP1_PDB_SHA256 = "2d4f1ecb8111f363db5f4e4380caeee5aac4af7a1394aca212d37a2a7b791709"


def load_zapp1(structure_type, archive_path):
    """Create a Structure-compatible ZAPP-1 object from the verified deposit.

    The original Structure initializer and the existing pipeline object remain
    untouched. Coordinate arrays come directly from the deposited PDB records.
    ``zn`` is a camera target (metal midpoint), not an additional drawn atom.
    Dashed links represent N/O/S donor proximity within 2.8 A in this model.
    """
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    archive = pathlib.Path(archive_path)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ZAPP1_ARCHIVE_SHA256:
        raise ValueError("ZAPP-1 archive checksum differs from the verified deposit")
    with zipfile.ZipFile(archive) as source:
        data = source.read(ZAPP1_MEMBER)
    if hashlib.sha256(data).hexdigest() != ZAPP1_PDB_SHA256:
        raise ValueError("ZAPP-1 PDB checksum differs from the verified deposit")
    text = data.decode("utf-8")
    st = structure_type.__new__(structure_type)
    st.ca, st.o, st.ss, st.res, st.seq = {}, {}, {}, {}, {}
    st.metals, st.metal_names, st.tsa = [], [], []
    donors = []
    backbone = {"N", "CA", "C", "O", "OXT"}
    for line in text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or line[16] not in " A":
            continue
        name, resn, chain = line[12:16].strip(), line[17:20].strip(), line[21]
        resid, element = int(line[22:26]), line[76:78].strip().upper()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if record == "ATOM" and chain == "A":
            st.res.setdefault(resid, {})[name] = xyz
            st.seq[resid] = resn
            if name == "CA":
                st.ca[resid] = xyz
            elif name == "O":
                st.o[resid] = xyz
            if element in {"N", "O", "S"} and name not in backbone:
                donors.append((resid, xyz))
        elif record == "HETATM" and (chain, resn, resid) == ("B", "YYE", 199):
            if element == "ZN":
                st.metals.append(xyz)
                st.metal_names.append(name)
            elif element not in {"H", "D"}:
                st.tsa.append((name, element, xyz))
                if element in {"N", "O", "S"}:
                    donors.append((None, xyz))
    if len(st.ca) != 198 or len(st.o) != 198 or len(st.metals) != 2:
        raise ValueError("Unexpected ZAPP-1 protein/metal inventory")

    atoms = PDBFile.read(io.StringIO(text)).get_structure(model=1, altloc="first")
    protein = atoms[(atoms.chain_id == "A") & struc.filter_amino_acids(atoms)]
    residue_ids, _ = struc.get_residues(protein)
    sse = struc.annotate_sse(protein)
    styles = {
        struc.PSeaElement.HELIX: "H",
        struc.PSeaElement.STRAND: "E",
        struc.PSeaElement.COIL: "L",
        struc.PSeaElement.NONE: "L",
    }
    st.ss = {int(resid): styles[struc.PSeaElement(kind)]
             for resid, kind in zip(residue_ids, sse, strict=True)}
    st.ids = sorted(st.ca.keys() & st.o.keys())
    st.centroid = np.mean([st.ca[i] for i in st.ids], axis=0)
    st.zn = np.mean(st.metals, axis=0)
    st.coordinating_residues, st.coordination = set(), []
    for resid, donor in donors:
        for metal in st.metals:
            if np.linalg.norm(donor - metal) <= 2.8:
                st.coordination.append((donor, metal))
                if resid is not None:
                    st.coordinating_residues.add(resid)
    return st


def film_color(facing, phase, light=0.0):
    """IPD-inspired shading based on facet angle and sequence position."""
    t = 0.62 * facing ** 1.2 + 0.38 * phase
    base = ramp([(0, "#5A36A2"), (0.3, PINK), (0.52, "#BC4FAD"), (0.76, BLUE), (1, "#A9DFF2")], t)
    return base + (np.array([255, 244, 252.0]) - base) * light


def ribbon(st, rot, scale, cx, cy, per=6, fog=0.7, halfwidth=1.35):
    """Depth-sorted facets for the backbone and active site, in SVG coordinates."""
    origin = st.centroid
    project = lambda p: rot @ (p - origin)
    to2d = lambda v: (cx + v[0] * scale, cy - v[1] * scale)
    frames = st.ribbon_frames(per)
    zs = [project(f[0])[2] for f in frames]
    zmin, zmax = min(zs), max(zs)
    depth = lambda z: (z - zmin) / (zmax - zmin)
    ink = rgb(INK)
    light_dir = np.array([-0.35, 0.55, 0.76])
    half = light_dir + np.array([0, 0, 1.0])
    half /= np.linalg.norm(half)
    items = []

    # Ribbon half-widths with arrowheads on strands and tapered transitions.
    n = len(frames)
    target = []
    kinds = [f[2] for f in frames]
    for i, kind in enumerate(kinds):
        w = {"H": halfwidth, "E": halfwidth, "L": 0.3}[kind]
        if kind == "E":
            end = i
            while end + 1 < n and kinds[end + 1] == "E":
                end += 1
            left = end - i
            if left < per * 1.4:
                w = halfwidth * 1.75 * (left / (per * 1.4)) + 0.3
        target.append(w)
    widths = list(target)
    for i in range(1, n - 1):
        if kinds[i] == "L" or kinds[i - 1] != kinds[i]:
            widths[i] = (target[i - 1] + target[i] + target[i + 1]) / 3

    # Thin coil runs at half the sampling rate: the facets are flat, so detail there is noise.
    step = lambda i: 2 if kinds[i] == "L" and i + 2 < n and kinds[i + 2] == "L" else 1
    i = 0
    while i < n - 1:
        j = i + step(i)
        (p0, s0, kind, res, _), (p1, s1, *_rest) = frames[i], frames[j]
        a, b = project(p0), project(p1)
        tangent = b - a
        tangent /= np.linalg.norm(tangent) or 1
        if kind == "L":
            side0 = np.cross(tangent, [0, 0, 1.0])
            side0 /= np.linalg.norm(side0) or 1
            side1 = side0
            normal = np.cross(tangent, side0)
            facing = 0.42 + 0.58 * (1 - abs(tangent[2]))
        else:
            side0, side1 = rot @ s0, rot @ s1
            normal = np.cross(tangent, side0 + side1)
            normal /= np.linalg.norm(normal) or 1
            facing = abs(normal[2])
        if normal[2] < 0:
            normal = -normal
        phase = 0.5 + 0.5 * math.sin(res * 2 * math.pi / 52 + 1.1)
        spec = max(0.0, normal @ half) ** 36 * 0.68
        lambert = 0.76 + 0.24 * max(0.0, normal @ light_dir)
        color = film_color(facing, phase, spec) * lambert
        z = (a[2] + b[2]) / 2
        color = ink + (color - ink) * (1 - fog * (1 - depth(z)) ** 1.4)
        w0, w1 = widths[i], widths[j]
        pts = [to2d(a + side0 * w0), to2d(b + side1 * w1), to2d(b - side1 * w1), to2d(a - side0 * w0)]
        items.append((z, "q", pts, hexc(color)))
        i = j

    # Metal-coordinating residues and bound ligand, drawn from source coordinates.
    def stick(p, q, color, width, dash=False):
        a, b = project(p), project(q)
        z = (a[2] + b[2]) / 2 + 0.6
        fade = 1 - 0.45 * (1 - depth(z))
        items.append((z, "s", (to2d(a), to2d(b)), hexc(ink + (rgb(color) - ink) * fade), width * scale, dash))

    atom_color = {"N": "#A6DAEE", "O": "#F39AD8", "P": "#FFFFFF", "S": "#FFFFFF"}
    ligands = sorted(getattr(st, "coordinating_residues", [92, 96, 102]))
    for r in ligands:
        atoms = st.res[r]
        names = list(atoms)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                if x in ("N", "C", "O") or y in ("N", "C", "O"):
                    continue
                if np.linalg.norm(atoms[x] - atoms[y]) < 1.75:
                    stick(atoms[x], atoms[y], WHITE, 0.26)
        if not hasattr(st, "coordination"):
            stick(atoms["NE2"], st.zn, "#D8CFF7", 0.11, True)
    for donor, metal in getattr(st, "coordination", []):
        stick(donor, metal, "#D8CFF7", 0.11, True)
    for i, (nx, ex, x) in enumerate(st.tsa):
        for ny, ey, y in st.tsa[i + 1:]:
            if np.linalg.norm(x - y) < (2.0 if "P" in (ex, ey) else 1.75):
                color = atom_color.get(ex, atom_color.get(ey, "#F2EEFF"))
                stick(x, y, color, 0.19)
    for metal in getattr(st, "metals", [st.zn]):
        zv = project(metal)
        items.append((zv[2] + 1.2, "z", to2d(zv), 0.92 * scale))
    return sorted(items, key=lambda it: it[0])


def render_items(items, prec=1):
    out, run, run_color = [], [], None

    def flush():
        nonlocal run, run_color
        if run:
            out.append(f'<path fill="{run_color}" stroke="{run_color}" d="{"".join(run)}"/>')
        run, run_color = [], None

    for it in items:
        if it[1] == "q":
            if it[3] != run_color:
                flush()
                run_color = it[3]
            p = it[2]
            run.append("M" + "L".join(f"{num(x, prec)} {num(y, prec)}" for x, y in p) + "Z")
        elif it[1] == "s":
            flush()
            (x1, y1), (x2, y2) = it[2]
            dash = f' stroke-dasharray="{num(it[4] * 1.6)} {num(it[4] * 2.2)}"' if it[5] else ""
            out.append(f'<path stroke="{it[3]}" stroke-width="{num(it[4], 2)}"{dash} '
                       f'd="M{num(x1, prec)} {num(y1, prec)}L{num(x2, prec)} {num(y2, prec)}"/>')
        else:
            flush()
            (x, y), r = it[2], it[3]
            out.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r * 2.6)}" fill="url(#zng)"/>'
                       f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="url(#zn)"/>')
    flush()
    return f'<g stroke-width=".55" stroke-linejoin="round" stroke-linecap="round" fill="none">{"".join(out)}</g>'


ZN_DEFS = (
    '<radialGradient id="zn" cx=".36" cy=".3" r=".78"><stop offset="0" stop-color="#FFFFFF"/>'
    '<stop offset=".38" stop-color="#E4DAFA"/><stop offset=".72" stop-color="#9FA6DF"/>'
    '<stop offset="1" stop-color="#6A4FA8"/></radialGradient>' + glow("zng", "#F4E9FF", 0.5)
)


# ---------------------------------------------------------------- surfaces
BAND = [(0, BLUE), (0.5, PINK), (1, PURPLE)]          # the three IPD colours, in order
FILM = [(0, "#0E3B4F"), (0.3, "#262B62"), (0.56, "#4E0C40"), (0.8, "#361F60"), (1, "#2B1A55")]


def slice_stops(stops, t0, t1, n=4):
    return [(k / (n - 1), hexc(ramp(stops, t0 + (t1 - t0) * k / (n - 1)))) for k in range(n)]


def surface(w, h, t0, t1, radius=6, band=3):
    """A dark plate carrying one crop of a single continuous film, with an exact IPD edge."""
    defs = (
        gradient("film", slice_stops(FILM, t0, t1), 0, 0, 1, 0.7)
        + gradient("band", slice_stops(BAND, t0, t1))
        + '<linearGradient id="sink" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset=".1" stop-color="{INK}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{INK}" stop-opacity=".72"/></linearGradient>'
        f'<clipPath id="c"><rect width="{w}" height="{h}" rx="{radius}"/></clipPath>'
    )
    body = (
        f'<g clip-path="url(#c)"><rect width="{w}" height="{h}" fill="url(#film)"/>'
        f'<rect width="{w}" height="{h}" fill="url(#sink)"/>'
        f'<rect width="{w}" height="{band}" fill="url(#band)"/></g>'
    )
    return defs, body


def arrow(kind, x, y, color=SOFT, s=1.0):
    """Where the control goes: across to another file, down this page, or off to a publisher."""
    d = {
        "right": "M0 4.5H9M5.2 .7 9 4.5 5.2 8.3",
        "down": "M4.5 0V9M.7 5.2 4.5 9 8.3 5.2",
        "out": "M1 8 8 1M2.4 1H8V6.6",
    }[kind]
    return (f'<path transform="translate({num(x)} {num(y)}) scale({num(s, 2)})" d="{d}" fill="none" '
            f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')


def fit(font, text, size, maxw, tracking=0.0, floor=8.0):
    while size > floor and font.width(text, size, tracking) > maxw:
        size -= 0.5
    return size


# ---------------------------------------------------------------- controls
TILE = (170, 60)


def tile(label, sub, kind, index, total, alt):
    w, h = TILE
    t0, t1 = index / total, (index + 1) / total
    defs, body = surface(w, h, t0, t1)
    body += arrow(kind, w - 28, 21, SOFT)
    body += UI_SEMI.path(label, fit(UI_SEMI, label, 16.5, w - 52), 14, 28, WHITE)
    body += UI_BODY.path(sub, fit(UI_BODY, sub, 12, w - 28), 14, 47, SOFT)
    return svg(w, h, alt, body, defs)


def plate(role, study, doi, index, total, alt, size=24):
    w, h = 272, 112
    t0, t1 = index / total, (index + 1) / total
    defs, body = surface(w, h, t0, t1, radius=6, band=4)
    body += UI_SEMI.path(role, fit(UI_SEMI, role, 12.5, w - 60), 17, 32, SOFT)
    body += arrow("out", w - 32, 20, SOFT)
    body += UI_SEMI.path(study, size, 17, 71, WHITE)
    body += UI_BODY.path(doi, fit(UI_BODY, doi, 11.5, w - 34), 17, 94, MUTED)
    return svg(w, h, alt, body, defs)


def spectrum(w=1200, h=5):
    return svg(w, h, "", f'<rect width="{w}" height="{h}" rx="{h / 2}" fill="url(#band)"/>', gradient("band", BAND))


# ---------------------------------------------------------------- masthead
VIEW = (0, -35, 20)   # roll, tilt, yaw about the centroid-to-zinc axis
MAST_VIEW = (0, -35, 20)
MAST_ALT = ("Computational design of metallohydrolases. "
            "Hypothesis-driven enzyme design using quantum chemistry and generative deep learning. "
            "Ribbon drawing of the deposited ZAPP-1 computational design model with its two zinc ions and modeled ligand.")
CREDIT = "ZAPP-1 · computational design model · RFdiffusion3"


def backdrop(w, h, spots):
    defs, body = "", f'<rect width="{w}" height="{h}" fill="{INK}"/>'
    for i, (cx, cy, rx, ry, color, op) in enumerate(spots):
        defs += glow(f"s{i}", color, op)
        body += f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="url(#s{i})"/>'
    return defs, body


def masthead_art(st, x, y, w, h):
    """Fit the deposited model inside the artwork panel without cropping it."""
    rot = st.rotation(*MAST_VIEW)
    points = np.array(list(st.ca.values()) + [a[2] for a in st.tsa] + st.metals)
    projected = (points - st.centroid) @ rot.T
    lo, hi = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    mid = (lo + hi) / 2
    scale = min((w - 32) / (hi[0] - lo[0]), (h - 32) / (hi[1] - lo[1]))
    return render_items(ribbon(st, rot, scale,
                               x + w / 2 - mid[0] * scale,
                               y + h / 2 + mid[1] * scale))


def masthead(st, w=1200, h=384):
    defs, body = backdrop(w, h, [
        (930, 186, 430, 320, PINK, 0.34),
        (1140, 50, 330, 280, BLUE, 0.38),
        (750, 366, 420, 240, PURPLE, 0.58),
    ])
    defs += ZN_DEFS + gradient("band", BAND)
    defs += f'<clipPath id="c"><rect width="{w}" height="{h}" rx="8"/></clipPath>'
    art = masthead_art(st, 804, 14, 382, 354)
    title = fit(UI_SEMI, "Computational design", 64, 700)
    copy = (
        UI_SEMI.path("Computational design", title, 54, 145, WHITE)
        + UI_SEMI.path("of metallohydrolases", title, 54, 222, WHITE)
        + UI_BODY.path("Hypothesis-driven enzyme design using", 23, 56, 277, SOFT)
        + UI_BODY.path("quantum chemistry and generative deep learning", 23, 56, 310, SOFT)
        + UI_BODY.path(CREDIT, 13, 56, 350, MUTED)
    )
    inner = f'{body}{art}<rect width="{w}" height="5" fill="url(#band)"/>{copy}'
    return svg(w, h, MAST_ALT, f'<g clip-path="url(#c)">{inner}</g>', defs)


def masthead_narrow(st, w=720, h=584):
    defs, body = backdrop(w, h, [
        (380, 150, 430, 290, PINK, 0.32),
        (630, 40, 300, 250, BLUE, 0.38),
        (110, 300, 400, 280, PURPLE, 0.6),
    ])
    defs += ZN_DEFS + gradient("band", BAND)
    defs += f'<clipPath id="c"><rect width="{w}" height="{h}" rx="10"/></clipPath>'
    art = masthead_art(st, 42, 18, 636, 304)
    title = fit(UI_SEMI, "Computational design", 60, w - 72)
    copy = (
        UI_SEMI.path("Computational design", title, 36, 382, WHITE)
        + UI_SEMI.path("of metallohydrolases", title, 36, 450, WHITE)
        + UI_BODY.path("Hypothesis-driven enzyme design", 26, 38, 492, SOFT)
        + UI_BODY.path("using quantum chemistry and", 26, 38, 524, SOFT)
        + UI_BODY.path("generative deep learning", 26, 38, 556, SOFT)
    )
    inner = f'{body}{art}<rect width="{w}" height="6" fill="url(#band)"/>{copy}'
    return svg(w, h, MAST_ALT, f'<g clip-path="url(#c)">{inner}</g>', defs)


# ---------------------------------------------------------------- pipeline glyphs
def poly(points):
    return "M" + "L".join(f"{num(x)} {num(y)}" for x, y in points)


def spline(pts, n=4):
    P = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(len(pts) - 1):
        for k in range(n):
            u = k / n
            b = np.array([(1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3]) / 6
            out.append(b @ P[i:i + 4])
    return np.array(out)


def g_site(st, cx, cy, s):
    """Catalytic geometry: the three imidazole rings and the zinc of 1QJI, from the deposited coordinates.

    Only the ring atoms are drawn, so no CB, CA or backbone connections appear.
    """
    rot = st.rotation(14, -28, 12)
    to = lambda p: (cx + (rot @ (p - st.zn))[0] * s, cy - (rot @ (p - st.zn))[1] * s)
    ring = ("CG", "ND1", "CE1", "NE2", "CD2")
    bonds, dashes = "", ""
    for r in (92, 96, 102):
        atoms = st.res[r]
        for x, y in zip(ring, ring[1:] + ring[:1]):
            bonds += poly([to(atoms[x]), to(atoms[y])])
        dashes += poly([to(atoms["NE2"]), to(st.zn)])
    x, y = to(st.zn)
    return (f'<path d="{dashes}" fill="none" stroke="#CFC2F2" stroke-width="1.7" stroke-dasharray="2.2 3.6" stroke-linecap="round"/>'
            f'<path d="{bonds}" fill="none" stroke="{WHITE}" stroke-width="2.9" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(s * 2.6)}" fill="url(#zng)"/>'
            f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(s * 1.05)}" fill="url(#zn)"/>')


def g_scaffold(st, cx, cy, s, seed=7):
    """Scaffold generation, schematic: seeded noise around the 1QJI trace, not a diffusion output."""
    rot = st.rotation(*VIEW)
    pts = (st.trace() - st.centroid) @ rot.T
    rng = np.random.default_rng(seed)
    span = pts[:, 0].max() - pts[:, 0].min()
    done = np.clip((pts[:, 0] - pts[:, 0].min()) / span, 0, 1)[:, None]
    noise = pts + rng.normal(0, 1.0, pts.shape) * (5.4 * (1 - done) + 0.6)
    to = lambda p: (cx + p[0] * s, cy - p[1] * s)
    ghost = "".join(
        f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(1.4 + 1.6 * (1 - d), 2)}" opacity="{num(0.25 + 0.6 * (1 - d), 2)}"/>'
        for (x, y), d in zip(map(to, noise), done[:, 0])
    )
    trace = poly([to(p) for p in spline(pts, 3)])
    return (f'<g fill="#9E7BD8">{ghost}</g>'
            f'<path d="{trace}" fill="none" stroke="url(#emerge)" stroke-width="2.8" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def g_sequence(st, cx, cy, size):
    """Sequence design, schematic: 1QJI residues 90-107 with its zinc-binding histidines marked."""
    three = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I",
                 LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
    cols, rows, fixed = 6, 3, {92, 96, 102}
    gx, gy, start = size * 1.22, size * 1.5, 90
    out = ""
    for k in range(cols * rows):
        res = start + k
        x = cx + (k % cols - (cols - 1) / 2) * gx
        y = cy + (k // cols - (rows - 1) / 2) * gy
        if res in fixed:
            out += f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(size * 0.82)}" fill="{PINK}"/>'
        out += SANS_SEMI.path(three[st.seq[res]], size, x, y + size * 0.36, WHITE if res in fixed else MUTED, "middle")
    return out


def g_validation(st, cx, cy, s, seed=3):
    """Structure validation, schematic: a seeded perturbation of the 1QJI trace, not a prediction."""
    rot = st.rotation(*VIEW)
    pts = (st.trace() - st.centroid) @ rot.T
    rng = np.random.default_rng(seed)
    drift = pts + np.cumsum(rng.normal(0, 0.1, pts.shape), axis=0) * 0.9
    to = lambda p: (cx + p[0] * s, cy - p[1] * s)
    a = poly([to(p) for p in spline(pts, 3)])
    b = poly([to(p) for p in spline(drift, 3)])
    return (f'<path d="{b}" fill="none" stroke="{PINK}" stroke-width="4" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity=".9"/>'
            f'<path d="{a}" fill="none" stroke="#9FD8EE" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')


def g_assay(cx, cy, s):
    """Experimental characterization: a schematic plate read, not measured data."""
    cols, rows, out = 6, 3, ""
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - (cols - 1) / 2) * s * 2.5
            y = cy + (r - (rows - 1) / 2) * s * 2.5
            level = min(1.0, (c / (cols - 1)) * (1.15 - 0.18 * r))
            fill = hexc(ramp([(0, "#1A1733"), (0.45, PINK), (1, "#F7C8EC")], level))
            out += (f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(s)}" fill="{fill}" '
                    f'stroke="#6A5E92" stroke-width="1"/>')
    return out


STAGES = [
    ("Active-site geometry", "theozyme", "site"),
    ("Scaffold generation", "RFdiffusion2 · RFdiffusion3", "scaffold"),
    ("Sequence design", "ProteinMPNN · LigandMPNN", "sequence"),
    ("Structure validation", "AlphaFold2 · AlphaFold3", "validation"),
    ("Experimental characterization", "activity assays", "assay"),
]
PIPE_ALT = ("Schematic of the design pipeline: active-site geometry (theozyme), scaffold generation with "
            "RFdiffusion2 and RFdiffusion3, sequence design with ProteinMPNN and LigandMPNN, structure "
            "validation with AlphaFold2 and AlphaFold3, and experimental characterization by activity assays. "
            "Illustration only: the noise cloud is not a model output, the overlaid traces are not an "
            "AlphaFold result, and the assay wells are not measurements.")


def glyph(st, kind, cx, cy, s):
    if kind == "site":
        return g_site(st, cx, cy, s * 0.165)
    if kind == "scaffold":
        return g_scaffold(st, cx, cy, s * 0.037)
    if kind == "sequence":
        return g_sequence(st, cx, cy, s * 0.21)
    if kind == "validation":
        return g_validation(st, cx, cy, s * 0.037)
    return g_assay(cx, cy, s * 0.082)


def pipe_defs(w, h, rx):
    emerge = ('<linearGradient id="emerge"><stop offset="0" stop-color="#BDE6F6" stop-opacity=".05"/>'
              '<stop offset=".45" stop-color="#E7C4F2" stop-opacity=".55"/>'
              '<stop offset="1" stop-color="#F6E7FF" stop-opacity="1"/></linearGradient>')
    return (ZN_DEFS + gradient("pearl", PEARL, 0, 0, 1, 0) + gradient("band", BAND) + emerge
            + glow("hz", PURPLE, 0.45)
            + f'<clipPath id="c"><rect width="{w}" height="{h}" rx="{rx}"/></clipPath>')


def pipeline(st, w=1120, h=236):
    cols = len(STAGES)
    step = w / cols
    body = f'<rect width="{w}" height="{h}" fill="{INK}"/><ellipse cx="{w / 2}" cy="{h}" rx="{w * 0.5}" ry="{h * 0.6}" fill="url(#hz)"/>'
    line = h - 72
    body += f'<rect x="28" y="{line - 1.5}" width="{w - 56}" height="3" rx="1.5" fill="url(#band)" opacity=".85"/>'
    for i, (title, note, kind) in enumerate(STAGES):
        cx = step * (i + 0.5)
        body += glyph(st, kind, cx, 76, 96)
        body += f'<circle cx="{num(cx)}" cy="{line}" r="6" fill="{INK}"/><circle cx="{num(cx)}" cy="{line}" r="4" fill="{WHITE}"/>'
        body += SANS_SEMI.path(title, fit(SANS_SEMI, title, 15, step - 18, -0.008), cx, line + 32, WHITE, "middle", -0.008)
        body += SANS.path(note, fit(SANS, note, 12.5, step - 14), cx, line + 52, MUTED, "middle")
    return svg(w, h, PIPE_ALT, f'<g clip-path="url(#c)">{body}</g>', pipe_defs(w, h, 8))


def pipeline_narrow(st, w=640, h=620):
    rows = len(STAGES)
    step = h / rows
    body = f'<rect width="{w}" height="{h}" fill="{INK}"/><ellipse cx="0" cy="{h / 2}" rx="{w * 0.7}" ry="{h * 0.5}" fill="url(#hz)"/>'
    line = 116
    body += f'<rect x="{line - 1.5}" y="24" width="3" height="{h - 48}" rx="1.5" fill="url(#band)" opacity=".85"/>'
    for i, (title, note, kind) in enumerate(STAGES):
        cy = step * (i + 0.5)
        body += glyph(st, kind, 56, cy, 50)
        body += f'<circle cx="{line}" cy="{num(cy)}" r="7" fill="{INK}"/><circle cx="{line}" cy="{num(cy)}" r="4.5" fill="{WHITE}"/>'
        body += SANS_SEMI.path(title, fit(SANS_SEMI, title, 25, w - line - 50, -0.01), line + 30, cy + 1, WHITE, tracking=-0.01)
        body += SANS.path(note, fit(SANS, note, 19, w - line - 50), line + 30, cy + 30, MUTED)
    return svg(w, h, PIPE_ALT, f'<g clip-path="url(#c)">{body}</g>', pipe_defs(w, h, 12))


# ---------------------------------------------------------------- content
NAV = [
    ("docs", "Documentation", "Setup and workflows", "right", "Documentation: setup and workflows"),
    ("tutorials", "Tutorials", "RFdiffusion2 and 3", "down", "Tutorials for RFdiffusion2 and RFdiffusion3"),
    ("data", "Datasets", "Measurements and figures", "right", "Four experimental datasets with analysis apps"),
    ("cite", "Cite", "BibTeX and CITATION.cff", "down", "Citations, BibTeX and CITATION.cff"),
]
PAPERS = [
    ("nature", "Main paper · Nature (2026)", "Metallohydrolases", "10.1038/s41586-025-09746-w",
     "Computational design of metallohydrolases, Nature 649, 246-253 (2026), doi 10.1038/s41586-025-09746-w"),
    ("nature-methods", "Method · Nature Methods (2026)", "RFdiffusion2", "10.1038/s41592-025-02975-x",
     "Atom-level enzyme active site scaffolding using RFdiffusion2, Nature Methods 23, 96-105 (2026), "
     "doi 10.1038/s41592-025-02975-x"),
    ("science", "Method · Science (2026)", "RFdiffusion3", "10.1126/science.aeh2982",
     "Atomic functional site scaffolding in biopolymers with RFdiffusion3, Science eaeh2982 (2026), "
     "doi 10.1126/science.aeh2982"),
]


def fetch(name, url):
    path = CACHE / name
    if not path.exists():
        print(f"fetching {name}")
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def main():
    global CACHE, UI_BODY, UI_SEMI, SANS, SANS_SEMI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=pathlib.Path.home() / ".cache" / "metallohydrolase-readme",
                    help="where fonts and coordinates are downloaded (outside the repository)")
    args = ap.parse_args()
    CACHE = pathlib.Path(args.cache).expanduser()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        fetch(name, url)

    UI_BODY = Type("MonaSans.ttf", wght=400, wdth=100)
    UI_SEMI = Type("MonaSans.ttf", wght=600, wdth=100)
    SANS = Type("MonaSans.ttf", wght=450, wdth=100)
    SANS_SEMI = Type("MonaSans.ttf", wght=620, wdth=100)
    st = Structure(CACHE / "1QJI.pdb")
    design = load_zapp1(Structure, ZAPP1_ARCHIVE)

    assets = {"masthead.svg": masthead(design), "masthead-narrow.svg": masthead_narrow(design),
              "pipeline.svg": pipeline(st), "pipeline-narrow.svg": pipeline_narrow(st),
              "spectrum.svg": spectrum()}
    for i, (name, label, sub, kind, alt) in enumerate(NAV):
        assets[f"go-{name}.svg"] = tile(label, sub, kind, i, len(NAV), alt)
    size = min(fit(UI_SEMI, p[2], 24, 238) for p in PAPERS)
    for i, (name, role, study, doi, alt) in enumerate(PAPERS):
        assets[f"paper-{name}.svg"] = plate(role, study, doi, i, len(PAPERS), alt, size)
    for name, text in assets.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"{name:24} {len(text) / 1024:6.1f} kB")


if __name__ == "__main__":
    main()
