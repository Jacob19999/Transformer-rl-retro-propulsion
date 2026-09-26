"""Export the detailed Blender/CAD drone as the mission-control render model.

Run with Blender (not the Isaac interpreter):
    blender -b "CAD/EDF Drone v1/Blender/usd_v2.blend" --python simulation/isaac/mission_control/export_visual_model.py

The physics USD (export_geometry.py -> geometry.json) stays the kinematic truth:
the viewer poses one node per rigid link from recorded PhysX link poses. This
script therefore re-expresses the Blender meshes in the same link-local frames
(Body, FwdFin, AftFin, LeftFin, RightFin) and refuses to write if their bounds
disagree with geometry.json, so a CAD edit can never silently offset the fins.
"""
import json
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
DEST = HERE / 'static/drone_visual.glb'
MANIFEST = HERE / 'static/geometry.json'
FINS = ('FwdFin', 'AftFin', 'LeftFin', 'RightFin')
# 691k CAD triangles -> ~150k keeps servo horns, bolts and EDF blades legible
# while the GLB stays a few MB for the browser.
BODY_DECIMATE_RATIO = 0.22
BOUNDS_TOLERANCE_M = 1e-3
# Inspection of usd_v2.blend: this connected CAD solid is the twelve blades
# and spinner (10,603 vertices), distinct from the fixed duct and motor.
ROTOR_BOUNDS = ((-.045, -.045, -.023681), (.045, .045, .021599))


def link_bounds(obj):
    points = [Vector(corner) for corner in obj.bound_box]
    return [[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]]


def separate_rotor(body):
    """Preserve CAD materials/UVs and rotate only the identified fan solid."""
    mesh = body.data
    neighbors = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        neighbors[a].append(b)
        neighbors[b].append(a)
    unseen = set(range(len(mesh.vertices)))
    candidates = []
    while unseen:
        seed = unseen.pop()
        component, pending = {seed}, [seed]
        while pending:
            for index in neighbors[pending.pop()]:
                if index in unseen:
                    unseen.remove(index)
                    component.add(index)
                    pending.append(index)
        points = [mesh.vertices[i].co for i in component]
        bounds = [[min(p[i] for p in points) for i in range(3)],
                  [max(p[i] for p in points) for i in range(3)]]
        error = max(abs(a - b) for got, expected in zip(bounds, ROTOR_BOUNDS)
                    for a, b in zip(got, expected))
        if error <= BOUNDS_TOLERANCE_M:
            candidates.append(component)
    if len(candidates) != 1:
        sys.exit(f'Expected one CAD EDF rotor solid; found {len(candidates)}. Re-identify rotor bounds after a CAD change.')
    indices = candidates[0]
    rotor = body.copy()
    rotor.data = mesh.copy()
    rotor.name = 'EDFRotor'
    bpy.context.collection.objects.link(rotor)
    for obj, remove_rotor in ((body, True), (rotor, False)):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if (v.index in indices) == remove_rotor], context='VERTS')
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    # Meshes use the Body frame: its Z axis is the EDF shaft through X=Y=0.
    # Parenting carries the rotor with recorded body poses without moving the
    # duct, motor, stators, fins or any physics geometry.
    rotor.parent = body
    rotor.matrix_parent_inverse = Matrix.Identity(4)
    rotor.matrix_basis = Matrix.Identity(4)
    return rotor


def main():
    body_link = bpy.data.objects['Body']
    body = bpy.data.objects['edf_drone']
    # Bake each mesh into its rigid-link frame, then put every link node at
    # the origin with identity rotation: the viewer overwrites node poses.
    relative = body_link.matrix_world.inverted() @ body.matrix_world
    body.data.transform(relative)
    body.parent = None
    body.matrix_world = Matrix.Identity(4)
    # Fin objects are the hinge links themselves: mesh data is already local.
    for name in FINS:
        fin = bpy.data.objects[name]
        fin.parent = None
        fin.matrix_world = Matrix.Identity(4)
    for empty in ('Drone', 'Body'):
        bpy.data.objects.remove(bpy.data.objects[empty])
    body.name = 'Body'
    rotor = separate_rotor(body)

    for obj in (body, rotor):
        decimate = obj.modifiers.new('decimate', 'DECIMATE')
        decimate.ratio = BODY_DECIMATE_RATIO
        decimate.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=decimate.name)
        # CAD tessellation is flat-shaded; split normals at hard edges only.
        bpy.ops.object.shade_smooth_by_angle(angle=0.6)

    manifest = {link['name']: link for link in json.loads(MANIFEST.read_text())['links']}
    for name in ('Body', *FINS):
        obj = bpy.data.objects[name]
        got, expected = link_bounds(obj), manifest[name]['local_bounds']
        if name == 'Body':
            rotor_bounds = link_bounds(rotor)
            got = [[min(a, b) for a, b in zip(got[0], rotor_bounds[0])],
                   [max(a, b) for a, b in zip(got[1], rotor_bounds[1])]]
        error = max(abs(g - e) for pair in zip(got, expected) for g, e in zip(*pair))
        print(f'{name}: faces={len(obj.data.polygons)} bounds_error={error * 1000:.3f} mm')
        if error > BOUNDS_TOLERANCE_M:
            sys.exit(f'{name} link frame disagrees with geometry.json by {error:.4f} m')

    bpy.ops.object.select_all(action='DESELECT')
    for name in ('Body', 'EDFRotor', *FINS):
        bpy.data.objects[name].select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(DEST), export_format='GLB', use_selection=True,
                              export_yup=False, export_apply=True, export_materials='EXPORT',
                              export_texcoords=True, export_normals=True, export_cameras=False,
                              export_lights=False, export_animations=False)
    print(f'wrote {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
