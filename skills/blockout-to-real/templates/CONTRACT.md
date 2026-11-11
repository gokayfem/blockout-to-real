# Contract

Write this before the first target exists. It says what will be judged; do not change it after seeing results.

## Scene
- Input: (file name, kind, sha256 from case.json)
- Camera and size: (from case.json)
- Look: (case.json "look", copied here)

## What success means
1. The final .blend is ordinary meshes, one Principled BSDF material per object with image textures in `b2r_uv`, and lights. No emission tricks, no image planes, nothing composited (`b2r.py inspect`).
2. The re-render of the final scene from the case camera is closer to the target than the grey blockout is (LPIPS, `acceptance.json`).
3. A fresh reviewer, shown only blockout, target and re-render, places the re-render nearer the target and lists what still reads as CG.
4. The scene still renders from a moved camera and with changed lights without breaking.

## What is out of scope
- (list anything this case will not attempt)

## Stop rule
Loop at most 3 iterations; stop when LPIPS against a fixed target stops improving.
