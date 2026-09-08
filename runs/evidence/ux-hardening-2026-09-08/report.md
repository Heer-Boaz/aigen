# Workflow UX and execution hardening — 2026-09-08

Baseline: `1ba595419cd78fc3192010e45f2d444c529310ae` on `main`.
This round fixes reproducible lifecycle, history and input-integrity defects in
the existing workflow implementation.

## Corrected behavior

| Failure | Owning change | Regression evidence |
| --- | --- | --- |
| Editing an input after signature construction could publish different pixels under that signature. Historical comparisons also followed mutable originals. | `workflow_sources.py` captures independent, content-addressed bytes for images, reference packs, LoRAs, video and audio. Consumers receive captured paths. | Replace an input between intake and generation; verify generated output and subsequent cache reuse. Remove original references and reload the captured pack. |
| Replacing a reference pack or LoRA after compilation could invalidate compiled routing or architecture. | Preserve ordered reference-pack entries; compare the captured specification and LoRA architecture with the compiled configuration. Include ctime in LoRA inspection identity. | Reorder pack entries; replace actual SafeTensors headers with another architecture while preserving file size and mtime. Fail before publishing the source result. |
| Stop could leave resistant children alive; Quit blocked the event loop; malformed events could strand controls. | `tui_process.py` owns subprocess groups, drains pipes, shields cleanup and escalates TERM to KILL after five seconds. The TUI owns one asynchronous worker through generation and contact-sheet completion. | Real resistant parent/child processes, cancellation during spawn, immediate Stop, missing executable, malformed events, UTF-8 across chunks and a 155,535-byte record. Real FFmpeg contact-sheet flow and a responsive Quit heartbeat. |
| A signal could arrive between atomic node publication and in-memory registration. SIGKILL could leave persistent state running. | `workflow_run_records.py` owns preassigned run IDs and published-node recovery. Normal interruption and supervisor cleanup use atomic node manifests as completion evidence. The TUI restores their completed/reused statuses before finishing. | SIGINT/SIGTERM handler restoration, interruption immediately after publication, a real forced SIGKILL with a missing completion event, terminal history and TUI status assertions, run-ID confinement and collision rejection. |
| Opening a saved selection displayed a newer collection; navigation cancelled export tracking even while copying continued. | Results starts at the pinned selection. Exports belong to the application, retain completion/failure notifications after Results closes, and complete before Quit. | Older pin with newer history, delayed real export during Open/Close, and independent reviewer reproduction through Records/Close/Quit. |
| Bytes could change between export validation and copying. Full preflight hashing also duplicated the copy's reads. | Export resolves recorded identity and metadata, then hashes the actual copied stream before atomic publication. Other resolver callers retain content verification before use. | Count actual bytes read for source images, cached images and frame archives: one pass per payload. Same-size corruption with restored mtime and a changed second frame produce no destination or partial archive. |
| Video/audio inspector Browse controls did not have matching dispatch handlers. | Dispatch both source types through the existing file browser with the appropriate media filters. | Real Textual inspector → Browse → file selection → save/reload for video and audio. |

## Validation

The complete application command in [tests/README.md](../../../tests/README.md)
passed: **97 tests in 40.826 seconds**, exit code 0. The unmodified output is
saved in [application-tests.log](application-tests.log).

`python -m compileall -q aigen tests` passed. The final `git diff --check` passed.
The focused interruption/process bundle passed all 12 tests; the focused
export/result/video-flow bundle passed all 13 tests. Both independent reviews
closed their reproduced findings after the corresponding fixes.

These tests use real filesystem, subprocess, Textual, FFmpeg/PyAV and workflow
owners. Neural inference is replaced by CPU doubles where applicable; the
installed VOSR tiled inference code is exercised with a CPU neural double.
No model weights were downloaded and no GPU inference was run in this round.
This evidence establishes execution and UX behavior, not a new image-quality
or VRAM measurement.

## Performance and compatibility

Input capture copies and hashes together with a bounded 256 KiB buffer. A stat
revision index avoids recopying unchanged inputs; a per-run map avoids repeated
snapshot verification for the same revision. Captured files are independent
copies of editable originals. Full image resolution and reference order are
preserved. Snapshots consume additional disk space under `cache/sources`.

The executor revision advances from 4 to 5 so automatic reuse cannot accept
outputs made under the earlier mutable-input contract. Existing records remain
on disk. Saved selections retain their recorded identity and full integrity
checks. Generated outputs remain governed by the existing immutable cache.

Reference conditioning, prompts, sampling, the separate Klein/Qwen routes and
the user's character assets were not modified. The pre-existing inkstyle
corpus edits and raw run artifacts are outside this change.

## Production implementation references studied

- [AnyIO subprocess ownership](https://github.com/agronholm/anyio/blob/master/src/anyio/_core/_subprocesses.py)
  and [asyncio backend](https://github.com/agronholm/anyio/blob/master/src/anyio/_backends/_asyncio.py):
  scoped process lifetime, cancellation shielding and pipe draining.
- [Textual workers](https://github.com/Textualize/textual/blob/main/src/textual/worker.py)
  and [worker manager](https://github.com/Textualize/textual/blob/main/src/textual/worker_manager.py):
  worker ownership and cancellation behavior, including view versus application lifetime.
- [CPython asyncio runner](https://github.com/python/cpython/blob/3.13/Lib/asyncio/runners.py):
  coordinated SIGINT cancellation and signal-handler restoration.
- [DVC data hashing/staging](https://github.com/iterative/dvc-data/blob/main/src/dvc_data/hashfile/build.py):
  stream-derived object identity and state-based avoidance of repeated intake work.
- [pip stream hash verification](https://github.com/pypa/pip/blob/main/src/pip/_internal/utils/hashes.py):
  verify expected hashes against the actual consumed chunks.
- [VS Code working-copy file operations](https://github.com/microsoft/vscode/blob/main/src/vs/workbench/services/workingCopy/common/workingCopyFileService.ts):
  file-operation ownership with explicit completion and failure outcomes.
- [Dagster run monitoring](https://github.com/dagster-io/dagster/blob/master/python_modules/dagster/dagster/_daemon/monitoring/run_monitoring.py):
  a supervisor reconciles stored run state after worker termination. Here this is
  limited to the TUI's own known run after its process group has exited.
