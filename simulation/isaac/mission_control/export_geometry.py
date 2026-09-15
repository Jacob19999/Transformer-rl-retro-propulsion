"""Export actual Isaac USD meshes in each rigid link's local coordinates."""
from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'assets/usd/drone_v2_physics.usd'
    dest = ROOT / 'mission_control/static'
    dest.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    cache = UsdGeom.XformCache()
    scene = trimesh.Scene()
    records = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        link = str(prim.GetPath()).split('/')[2]
        link_prim = stage.GetPrimAtPath('/Drone/' + link)
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if np.all(counts == 3):
            faces = indices.reshape(-1, 3)
        else:
            faces, start = [], 0
            for count in counts:
                face = indices[start:start + count]
                faces.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
                start += count
        geometry = trimesh.Trimesh(vertices, faces, process=False)
        # USD row-vector convention -> trimesh column-vector convention.
        relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(link_prim).GetInverse()
        geometry.apply_transform(np.asarray(relative).T)
        bounds = geometry.bounds.tolist()
        original_faces = len(geometry.faces)
        if original_faces > 60000:
            geometry = geometry.simplify_quadric_decimation(face_count=60000)
        geometry.visual = trimesh.visual.ColorVisuals(geometry, face_colors=[170, 184, 192, 255] if link == 'Body' else [110, 245, 220, 255])
        scene.add_geometry(geometry, node_name=link, geom_name=link)
        world = cache.GetLocalToWorldTransform(link_prim)
        q = world.ExtractRotationQuat()
        records.append(dict(name=link, source_prim=str(prim.GetPath()), source_faces=original_faces,
                            rendered_faces=len(geometry.faces), local_bounds=bounds,
                            neutral_position=list(world.ExtractTranslation()),
                            neutral_quaternion=[q.GetReal(), *q.GetImaginary()]))
    (dest / 'drone.glb').write_bytes(scene.export(file_type='glb'))
    manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    coordinate_frame='USD Z up, metres; geometry local to each rigid link', links=records)
    (dest / 'geometry.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
