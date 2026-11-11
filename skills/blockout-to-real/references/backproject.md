# Back-projection

`b2r.py backproject CASE --iter K` writes the decomposed maps into each object's own UV textures and builds one
ordinary Principled BSDF material per object. It is the step that makes the result a scene instead of a
picture: after it, the look lives in textures that render from any camera and under any light.

## Method

For every object, every texel of its `b2r_uv` map is rasterised to a 3D point on the surface, projected
through the camera (`camera.json`) and tested for visibility against the g-buffers:

- same object id at that pixel,
- the pixel's `position.exr` is within a tolerance of the texel's point (plane tolerance about 5 mm plus 2 mm
  per metre of depth; distance tolerance grows with the pixel footprint),
- the surface is not seen at a grazing angle (cosine at least 0.15).

Seen texels sample the maps bilinearly, using only pixels of the same object near the texel's point, so
colours never bleed across object edges. Texture resolution per object is the power of two that covers the
99th percentile of the on-screen texel footprint, clipped to 512..4096.

If a mesh has no `b2r_uv`, its active UV map is used and the fallback is recorded.

Normals: the estimated normal minus the g-buffer's geometric normal, high-passed (sigma 6 px) so that only
detail remains, written as a tangent-space normal map. The blockout's shape stays the shape; the photo adds
relief.

## What the camera did not see

Texels the camera never saw are filled from the same object's seen texels: the 8 nearest in 3D, inverse-
distance weighted, blended towards the object's robust mean as the distance grows (falloff 0.25 m), with a
flat normal. Texels seen only at a grazing angle give fill values, not projected ones. An object the camera
never saw at all gets the robust mean of the most similar seen objects (same category first, then closest in
size, at most three). `coverage.json` records per object what was seen and how the rest was filled, with the
donors. These fills are why a moved camera shows smeared floor under furniture and a flat wall: the method
has one view.

## Materials

One Principled BSDF per object: base colour, roughness, metallic, normal map and a bump from the height map,
all image textures in `b2r_uv`, all images packed into the `.blend`. No emission shaders, no compositor, no
camera-facing image planes. `b2r.py inspect BLEND OUT.json` proves that for any `.blend`.

## Checks

- **Round trip.** An unlit render of the textured scene's base colour from the fixed camera, compared with the
  albedo map away from object edges. The first run reached 44.8 dB (mean error 0.42 of 255). A low number
  means the projector, the camera or the UVs are wrong; fix that before anything else.
- **Novel views.** Renders of the textured scene from cameras orbited about the point the camera looks at
  (`roundtrip_novel_<i>.png`) show the fills.
- **What the projection costs.** Resampling through UV textures blurs: in the first run the albedo round trip
  alone cost about 0.05 LPIPS, roughly a third of the final 0.134.

## Files

`backproject/iter<k>/textures/<object>_{basecolor,roughness,metallic,normal,height,coverage}.png`,
`scene.blend` (materials, original lights), `coverage.json`, `roundtrip.png`, `roundtrip.json`,
`backproject_log.txt`. The light fit then writes `scene_lit.blend` next to it (`references/lighting.md`).
