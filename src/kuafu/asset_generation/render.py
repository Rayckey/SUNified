"""Small CPU z-buffer renderer for reproducible previews without OpenGL/Blender."""

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .geometry import BuiltAsset

VIEWS = [
    (-90, 15, "Front (-Y)"),
    (0, 15, "Right (+X)"),
    (90, 15, "Back (+Y)"),
    (180, 15, "Left (-X)"),
    (-45, 30, "Three-quarter"),
    (-90, 85, "Top"),
]


def shade_srgb(colors: np.ndarray, illumination: np.ndarray) -> np.ndarray:
    """Apply simple diffuse illumination in linear light, then encode the preview as sRGB."""
    srgb = np.asarray(colors, dtype=float) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lit = np.maximum(0, linear * np.asarray(illumination)[:, None])
    encoded = np.where(lit <= 0.0031308, 12.92 * lit, 1.055 * lit ** (1 / 2.4) - 0.055)
    return np.clip(np.round(encoded * 255), 0, 255).astype(np.uint8)


def render_mesh(
    mesh: trimesh.Trimesh, azimuth: float = -45, elevation: float = 25, size: int = 320
) -> Image.Image:
    az, el = np.deg2rad([azimuth, elevation])
    outward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.cross(outward, right)
    vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) @ np.stack([right, up, outward], axis=1)
    factor = (size * 0.82) / max(float(np.linalg.norm(mesh.extents)), 1e-10)
    screen = np.column_stack(
        [vertices[:, 0] * factor + size / 2, size / 2 - vertices[:, 1] * factor, vertices[:, 2]]
    )
    pixels = np.full((size, size, 3), [239, 242, 247], dtype=np.uint8)
    zbuffer = np.full((size, size), -np.inf)
    lighting = 0.4 + 0.6 * np.maximum(0, mesh.face_normals @ np.array([0.3, -0.5, 0.8124]))
    textured = mesh.visual.kind == "texture" and mesh.visual.uv is not None
    if textured:
        material = mesh.visual.material
        texture = np.asarray(material.baseColorTexture.convert("RGB"))
        multiplier = material.baseColorFactor
        multiplier = np.ones(3) if multiplier is None else np.asarray(multiplier[:3]) / 255
    else:
        colors = shade_srgb(mesh.visual.face_colors[:, :3], lighting)
    for face_index, face in enumerate(mesh.faces):
        a, b, c = screen[face]
        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(size - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(size - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10 or x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1] + 0.5
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w2 = 1 - w0 - w1
        depth = w0 * a[2] + w1 * b[2] + w2 * c[2]
        region = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        mask = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8) & (depth > region)
        region[mask] = depth[mask]
        if textured and mask.any():
            uv = mesh.visual.uv[face]
            samples = (w0[mask, None] * uv[0] + w1[mask, None] * uv[1] + w2[mask, None] * uv[2]) % 1
            tx = np.minimum((samples[:, 0] * texture.shape[1]).astype(int), texture.shape[1] - 1)
            ty = np.minimum(((1 - samples[:, 1]) * texture.shape[0]).astype(int), texture.shape[0] - 1)
            srgb = texture[ty, tx].astype(float) / 255
            linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
            lit = np.maximum(0, linear * multiplier * lighting[face_index])
            encoded = np.where(lit <= 0.0031308, 12.92 * lit, 1.055 * lit ** (1 / 2.4) - 0.055)
            color = np.clip(np.round(encoded * 255), 0, 255).astype(np.uint8)
        else:
            color = [0, 0, 0] if textured else colors[face_index]
        pixels[y0 : y1 + 1, x0 : x1 + 1][mask] = color
    return Image.fromarray(pixels)


def render_views(asset: BuiltAsset, out: Path) -> list[Path]:
    from kuafu.paths import resolve_artifact_path

    out = resolve_artifact_path(out, "Asset preview directory")
    out.mkdir(parents=True, exist_ok=True)
    mesh = preview_mesh(asset.visuals)
    sheet = Image.new("RGB", (3 * 320, 2 * 350), "#eff2f7")
    draw = ImageDraw.Draw(sheet)
    paths = []
    for i, (az, el, label) in enumerate(VIEWS):
        view = render_mesh(mesh, az, el)
        path = out / f"view_{i:02d}.png"
        view.save(path)
        paths.append(path)
        x, y = i % 3 * 320, i // 3 * 350
        sheet.paste(view, (x, y))
        draw.text((x + 12, y + 328), label, fill="#172033")
    sheet.save(out / "preview.png")
    collision = trimesh.util.concatenate(asset.collisions)
    collision.visual.face_colors = [71, 181, 143, 255]
    render_mesh(collision).save(out / "collisions.png")
    return paths


def preview_mesh(meshes):
    """Merge preview materials without trimesh 4.5's uint8 color-factor overflow."""
    prepared = []
    for source in meshes:
        mesh = source.copy()
        if mesh.visual.kind == "texture":
            material = mesh.visual.material
            factor = getattr(material, "baseColorFactor", None)
            if factor is not None and material.baseColorTexture is not None:
                srgb = np.asarray(material.baseColorTexture.convert("RGB"), dtype=float) / 255
                linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
                linear *= np.asarray(factor[:3], dtype=float) / 255
                encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
                material.baseColorTexture = Image.fromarray(
                    np.clip(np.round(encoded * 255), 0, 255).astype(np.uint8)
                )
                material.baseColorFactor = None
        prepared.append(mesh)
    return trimesh.util.concatenate(prepared)


def write_viewer(asset: BuiltAsset, out: Path) -> None:
    """Self-contained, offline orbit viewer with a collision toggle."""
    from kuafu.paths import resolve_artifact_path

    out = resolve_artifact_path(out, "Asset viewer")

    def packed(mesh):
        return {
            "vertices": np.round(mesh.vertices, 7).tolist(),
            "faces": mesh.faces.tolist(),
            "colors": mesh.visual.face_colors[:, :3].tolist(),
        }

    collision = trimesh.util.concatenate(asset.collisions)
    collision.visual.face_colors = [71, 181, 143, 255]
    data = json.dumps(
        {
            "visual": packed(asset.mesh()),
            "collision": packed(collision),
            "bounds": asset.mesh().bounds.tolist(),
        },
        separators=(",", ":"),
    )
    page = VIEWER.replace("__MESH_DATA__", data)
    out.write_text(page)


VIEWER = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kuafu asset preview</title>
<style>
body{margin:0;background:#eff2f7;color:#172033;font:15px system-ui}
header{position:absolute;top:20px;left:24px}
h1{font-size:22px;margin:0 0 8px}
button{padding:8px 12px;border:1px solid #bbc4d1;border-radius:6px;background:white}
canvas{display:block;width:100vw;height:100vh;touch-action:none}p{max-width:500px}footer{position:absolute;bottom:18px;left:24px}
</style><header><h1>Kuafu</h1><p>Drag to orbit · Scroll to zoom · Meters, Z-up</p>
<button id="toggle">Show collisions</button> <button id="reset">Reset view</button></header>
<canvas id="view" aria-label="Interactive 3D asset preview"></canvas>
<footer id="stats"></footer><script>
'use strict';
const data=__MESH_DATA__, canvas=document.getElementById('view'), ctx=canvas.getContext('2d');
let az=-Math.PI/4, el=0.45, zoom=1, collision=false, drag=null;
const center=data.bounds[0].map((v,i)=>(v+data.bounds[1][i])/2);
const extent=Math.hypot(...data.bounds[1].map((v,i)=>v-data.bounds[0][i]));
function draw(){
 const w=canvas.width=innerWidth, h=canvas.height=innerHeight, mesh=collision?data.collision:data.visual;
 const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),s=Math.min(w,h)*.7/extent*zoom;
 const points=mesh.vertices.map(v=>{const x=v[0]-center[0],y=v[1]-center[1],z=v[2]-center[2];
  return [w/2+s*(-sa*x+ca*y),h/2-s*(-se*ca*x-se*sa*y+ce*z),ce*ca*x+ce*sa*y+se*z];});
 const faces=mesh.faces.map((f,i)=>({f,i,z:f.reduce((v,j)=>v+points[j][2],0)})).sort((a,b)=>a.z-b.z);
 ctx.clearRect(0,0,w,h);
 for(const {f,i} of faces){
  const [a,b,c]=f.map(j=>points[j]), p=mesh.vertices[f[0]],q=mesh.vertices[f[1]],r=mesh.vertices[f[2]];
  const u=q.map((v,j)=>v-p[j]),v=r.map((v,j)=>v-p[j]);
  const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],len=Math.hypot(...n)||1;
  const light=.4+.6*Math.max(0,(n[0]*.3-n[1]*.5+n[2]*.8124)/len);
  ctx.fillStyle='rgb('+mesh.colors[i].map(c=>{
   const s=c/255,l=(s<=.04045?s/12.92:Math.pow((s+.055)/1.055,2.4))*light;
   return Math.round(255*(l<=.0031308?l*12.92:1.055*Math.pow(l,1/2.4)-.055));
  }).join(',')+')';
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill();
 }
 document.getElementById('stats').textContent=mesh.faces.length.toLocaleString()+' triangles · '+
  data.bounds[1].map((v,i)=>((v-data.bounds[0][i])*1000).toFixed(1)).join(' × ')+' mm';
}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;az-=(e.clientX-drag[0])*.008;el=Math.max(-1.5,Math.min(1.5,el+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw();};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
document.getElementById('toggle').onclick=()=>{
 collision=!collision;
 document.getElementById('toggle').textContent=collision?'Show visual':'Show collisions';draw();
};
document.getElementById('reset').onclick=()=>{az=-Math.PI/4;el=.45;zoom=1;draw();};window.onresize=draw;draw();
</script></html>
"""
