#!/usr/bin/env python3
"""Build the flycrane URDF (3 drones on 1 m rigid cables under the payload) for a given drone, and
optionally convert it to USD with Isaac Lab 3.0's converter.

Starts from Jack's URDF (assets/data/AMR/flycrane_rod_data/flycrane.urdf, the source of the
flycrane_rod.usd the hover task loads) and keeps the payload, the cables and the joint tree as they are.
It changes only:
  * names: "/" -> "_" in every link / joint name. The Isaac Lab 3.0 URDF importer would otherwise encode
    "Falcon1/base_link" into something like "tn__Falcon1base_link_VP" and the env's regexes
    ("Falcon.*_base_link_inertia", "Falcon.*_rotor_.*", ...) would find nothing. The old USD already
    had the "_" names, so with "_" the bodies keep exactly the names the env expects.
  * with --drone omninxt (default), the three drones, from controllers/drone_params.py (OMNINXT):
      Falcon<i>_base_link          = cable attachment point (the cable's last sphere joint ends here),
                                     given a tiny explicit mass (Jack's has no <inertial>)
      Falcon<i>_base_link_inertia  = drone body at the CoM: total mass minus the small bodies, the
                                     inertia tensor (diagonal + off-diagonal), a box collider
      Falcon<i>_rotor_k            = thrust application points at the real rotor hubs (near-massless)
      Falcon<i>_imu_link           = at the CoM;  Falcon<i>_odometry_sensor_link = at the attachment point
    Visuals are primitives (box body + propeller discs), not the Falcon meshes.
  * with --drone falcon: nothing else (only the names). Converting this one and replaying a trained
    controller on it must give the same results as the old USD: that validates the conversion pipeline
    before the drone changes.

    python3 scripts/assets/build_flycrane_asset.py --drone falcon      # URDF only, plain python3
    python3 scripts/assets/build_flycrane_asset.py --drone omninxt
    sbatch jobs/build_flycrane_asset.sh omninxt                         # URDF + USD on the cluster
Outputs: assets/data/AMR/flycrane_rod_data/flycrane_<drone>.urdf (next to the meshes it references)
         assets/data/AMR/flycrane_<drone>/flycrane_<drone>/flycrane_<drone>.usda (after conversion)
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import subprocess
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "exts/MARL_mav_carry_ext/MARL_mav_carry_ext"
AMR = EXT / "assets/data/AMR"
SRC_URDF = AMR / "flycrane_rod_data/flycrane.urdf"
N_DRONES = 3
SMALL_MASS = 1e-4          # kg, attachment point / rotor hubs (thrust application points only)
SMALL_INERTIA = 1e-9       # kg m^2
SENSOR_MASS = 1e-5         # kg, Jack's imu / odometry links (kept)

# OmniNxt simplified visual + collider, design (Fusion) frame, m (same values as the hook-grasp builder)
BODY_BOX_MIN = (-0.0950, -0.0689, 0.0)
BODY_BOX_MAX = (0.0866, 0.0689, 0.0548)
PROP_RADIUS = 0.0435
PROP_Z = (-0.0221, -0.0152)
# payload box when --no_meshes (from Jack's payload inertia: 1.4 kg box 0.55 x 0.45 x 0.15 m at the CoM)
LOAD_BOX = (0.55, 0.45, 0.15)


def load_params(name):
    path = EXT / "controllers/drone_params.py"
    spec = importlib.util.spec_from_file_location("_drone_params", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.get_drone_params(name)


def xyz(v):
    return " ".join(f"{c:.6f}" for c in v)


def sub(a, b):
    return tuple(a[k] - b[k] for k in range(3))


def rename_all(root):
    """'/' -> '_' in every link / joint name and reference."""
    for el in root.iter():
        for key in ("name", "link"):
            if el.tag in ("link", "joint", "parent", "child") and key in el.attrib:
                el.set(key, el.get(key).replace("/", "_"))


def find(root, tag, name):
    el = root.find(f"{tag}[@name='{name}']")
    if el is None:
        raise SystemExit(f"{tag} {name} not found in {SRC_URDF}")
    return el


def set_origin(el, pos, rpy=(0.0, 0.0, 0.0)):
    o = el.find("origin")
    if o is None:
        o = ET.SubElement(el, "origin")
    o.set("xyz", xyz(pos))
    o.set("rpy", xyz(rpy))


def set_inertial(link, mass, diag, offdiag=(0.0, 0.0, 0.0)):
    for old in link.findall("inertial"):
        link.remove(old)
    i = ET.Element("inertial")
    ET.SubElement(i, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(i, "mass", value=f"{mass:.6f}")
    ET.SubElement(i, "inertia", ixx=f"{diag[0]:.6e}", iyy=f"{diag[1]:.6e}", izz=f"{diag[2]:.6e}",
                  ixy=f"{offdiag[0]:.6e}", ixz=f"{offdiag[1]:.6e}", iyz=f"{offdiag[2]:.6e}")
    link.insert(0, i)


def set_shapes(link, tag, shapes):
    """Replace all <visual> or <collision> of a link: shapes = [(geometry_tag, attrs, pos)]."""
    for old in link.findall(tag):
        link.remove(old)
    for geom_tag, attrs, pos in shapes:
        v = ET.SubElement(link, tag)
        ET.SubElement(v, "origin", xyz=xyz(pos), rpy="0 0 0")
        g = ET.SubElement(v, "geometry")
        ET.SubElement(g, geom_tag, **attrs)


def apply_drone(root, d):
    """Replace the three Falcons by drone d (DroneParams). Frames: base_link = cable attachment point."""
    att = d.wire_attach                                   # design frame
    com_rel = sub(d.com, att)                             # CoM w.r.t. the attachment point
    m_body = d.mass - 4 * SMALL_MASS - SMALL_MASS - 2 * SENSOR_MASS
    box_c = tuple(0.5 * (BODY_BOX_MIN[k] + BODY_BOX_MAX[k]) for k in range(3))
    box_s = tuple(BODY_BOX_MAX[k] - BODY_BOX_MIN[k] for k in range(3))
    for i in range(1, N_DRONES + 1):
        p = f"Falcon{i}_"
        base = find(root, "link", p + "base_link")
        set_inertial(base, SMALL_MASS, (SMALL_INERTIA,) * 3)
        set_origin(find(root, "joint", p + "base_joint"), com_rel)
        body = find(root, "link", p + "base_link_inertia")
        set_inertial(body, m_body, d.inertia_diag, d.inertia_offdiag)
        set_shapes(body, "visual", [("box", {"size": xyz(box_s)}, sub(box_c, d.com))])
        set_shapes(body, "collision", [("box", {"size": xyz(box_s)}, sub(box_c, d.com))])
        for k, hub in enumerate(d.rotor_pos):
            set_origin(find(root, "joint", f"{p}rotor_{k}_joint"), sub(hub, att))
            rotor = find(root, "link", f"{p}rotor_{k}")
            set_inertial(rotor, SMALL_MASS, (SMALL_INERTIA,) * 3)
            disc = {"length": f"{PROP_Z[1] - PROP_Z[0]:.6f}", "radius": f"{PROP_RADIUS:.6f}"}
            dz = 0.5 * (PROP_Z[0] + PROP_Z[1]) - hub[2]
            set_shapes(rotor, "visual", [("cylinder", disc, (0.0, 0.0, dz))])
            set_shapes(rotor, "collision", [("cylinder", disc, (0.0, 0.0, dz))])
        set_origin(find(root, "joint", p + "imu_joint"), com_rel)
    return m_body


def strip_meshes(root):
    """Replace every mesh by a primitive (fallback if the converter cannot read the .dae files)."""
    for link in root.findall("link"):
        n = link.get("name")
        for tag in ("visual", "collision"):
            for v in link.findall(tag):
                g = v.find("geometry")
                if g is None or g.find("mesh") is None:
                    continue
                g.remove(g.find("mesh"))
                if n == "load_link":
                    ET.SubElement(g, "box", size=xyz(LOAD_BOX))
                    o = v.find("origin")
                    if o is None:
                        o = ET.SubElement(v, "origin")
                    o.set("xyz", "0 0 0.06")
                    o.set("rpy", "0 0 0")
                elif "rotor" in n:
                    ET.SubElement(g, "cylinder", length="0.005", radius="0.127")
                else:
                    ET.SubElement(g, "box", size="0.1 0.1 0.05")


def summary(root, drone):
    def mass(prefix):
        tot = 0.0
        for link in root.findall("link"):
            if link.get("name").startswith(prefix):
                i = link.find("inertial")
                tot += float(i.find("mass").get("value")) if i is not None else 0.0
        return tot
    print(f"[flycrane] drone {drone}: mass per drone {mass('Falcon1_'):.4f} kg (all Falcon1_* links), "
          f"payload {mass('load'):.4f} kg, cables {mass('rope_'):.4f} kg")
    j = find(root, "joint", "Falcon1_base_joint").find("origin").get("xyz")
    r = [find(root, "joint", f"Falcon1_rotor_{k}_joint").find("origin").get("xyz") for k in range(4)]
    print(f"[flycrane] CoM above the cable attachment: {j} | rotor hubs (rel. attachment): {r}")
    names = [el.get("name") for el in root.iter() if el.tag in ("link", "joint")]
    assert not any("/" in n for n in names), "a name still contains '/'"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone", default="omninxt", help="falcon (names only, for validation) or omninxt")
    ap.add_argument("--no_meshes", action="store_true",
                    help="replace the remaining .dae meshes (payload, and the Falcon in falcon mode) by boxes")
    ap.add_argument("--convert", action="store_true", help="also convert to USD (needs Isaac Sim)")
    ap.add_argument("--convert_tool", type=Path, default=Path("/workspace/isaaclab/scripts/tools/convert_urdf.py"))
    a = ap.parse_args()

    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    rename_all(root)
    if a.drone != "falcon":
        apply_drone(root, load_params(a.drone))
    if a.no_meshes:
        strip_meshes(root)
    root.set("name", f"flycrane_{a.drone}")
    summary(root, a.drone)

    out = AMR / "flycrane_rod_data" / f"flycrane_{a.drone}.urdf"
    text = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    text = "\n".join(line for line in text.splitlines() if line.strip())
    out.write_text(text + "\n")
    print(f"[flycrane] wrote {out}")

    if a.convert:
        usd_dir = AMR / f"flycrane_{a.drone}"
        cmd = [sys.executable, str(a.convert_tool), str(out), str(usd_dir),
               "--joint-stiffness", "0.0", "--joint-damping", "0.0", "--joint-target-type", "none"]
        print("[flycrane] converting:\n  " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print(f"[flycrane] expected USD: {usd_dir}/flycrane_{a.drone}/flycrane_{a.drone}.usda")


if __name__ == "__main__":
    main()