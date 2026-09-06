# Regression tests

Run from the repository root. The application and LightX2V deliberately use separate Python environments.

Application tests (73 tests, including real Textual → CLI → Pixel Art Fixer → cache execution, video assembly, character audit/publication and process cancellation):

```bash
PYTHONPATH=tests .venv/bin/python -m unittest \
  test_workflow_properties test_workflow_run_state \
  test_flux2_klein_conditioning test_generation_contracts \
  test_image_io_contracts test_vosr_randomness test_qwen_residency \
  test_model_artifact_identity test_workflow_image_flow \
  test_workflow_results_tui test_workflow_result_contracts \
  test_video_timing test_workflow_video_flow test_workflow_video_tui \
  test_workflow_video_contracts test_workflow_video_batching \
  test_character_edit_audit test_workflow_character_flow \
  test_qwen_masked_canvas test_workflow_mask_flow -v
```

Qwen reference and native masked-sampling tests (9 tests, in the installed LightX2V runtime):

```bash
PYTHONPATH=/home/boaz/aigen:/home/boaz/.cache/aigen-lightx2v/LightX2V \
  /home/boaz/.cache/aigen-lightx2v/venv/bin/python -m unittest \
  discover -s tests -p test_qwen_reference_resolution.py -v
PYTHONPATH=/home/boaz/aigen:/home/boaz/.cache/aigen-lightx2v/LightX2V \
  /home/boaz/.cache/aigen-lightx2v/venv/bin/python -m unittest \
  discover -s tests -p test_qwen_masked_sampling.py -v
```

The Klein tests execute real Diffusers image processing, reference packing, position IDs, noise generation and scheduling on CPU with neural doubles. The VOSR test loads the installed upstream tiled inference function and runs real batch/cache execution with CPU neural doubles. It requires the VOSR source installation; no model weights are loaded. The Qwen tests exercise preprocessing and staging without model inference.

The Qwen residency tests cover total token accounting, aliased weight storage,
and admission before the first and subsequent GPU block clones. The GPU
quantization equivalence probe and its measured outputs are in
`runs/evidence/pipeline-fixes-2026-09-06/gpu-acceptance/fp8_quantization_probe.py`.

These tests establish dataflow and state contracts. GPU memory use and neural image quality require separate model runs with reviewed prompts.

Model artifact identity tests replace local weights and processor configurations
within one process and verify that subsequent workflow cache signatures change.
They use small files, without loading model weights. Segmentation runtime
checks cover missing unrelated packages and engine-specific cache invalidation.

The image-flow tests cover per-seed caching, completed-output publication before
a batch failure, pinned selection after restart, original generation metadata,
legacy records, form import, and historical input integrity. The Textual tests
exercise comparison, record inspection, original export, selection/save, and
continuation through the real CPU CLI. A delayed widget teardown reproduces
stale Inspect events while switching between histories of different sizes;
both a successful switch and a failed load must retire the old view. OS viewer
dispatch is mocked; no external viewer is opened by the tests.

Video tests use FFmpeg/PyAV for actual CFR/VFR files, frame order, audio stream
selection and offsets, short/long audio, transparent-frame backgrounds and
portable frame archive reassembly. Graph tests exercise timing/audio propagation
through pixel-cache reuse and relocation of the source audio file. The video
batching tests replace only neural inference and verify shared seed sweeps,
publication before a later seed fails, resumption and per-seed invalidation.
Backend contract tests validate the installed AnimeGen/LTX files without loading
weights; Hunyuan's missing runtime is an explicit preflight-error test.

Character tests replace neural inference and VOSR with explicit CPU doubles.
They exercise raw-before-audit ordering, best-of-N selection, group-independent
retry seeds, bounded failure, malformed responses, durable partial audit
reports, cache identity and the actual selected seed. Qwen workflow tests use
the real native-reference/control preparation, including joint pose/scene
ordering and small scene maps that must fill the generation canvas.


Masked-edit tests exercise real source-canvas preparation, PNG compositing,
RGBA preservation, feathered masks, typed mask cache publication, source-byte
binding and region-plan checksum retention. The graph tests replace SAM neural
inference, native Qwen inference and the audit VLM; all intervening owners run.
SAM form import is exercised with no extra reference pack and blank optional
guidance fields, through compilation and execution as well as CLI construction.
Native sampling tests run real Diffusers/LightX2V scheduler math on CPU and
verify source initialization, strength suffix, next-sigma projection, mask
packing, and unchanged unmasked Euler behavior. They load no model weights.
