# Workflow property editing — 2026-09-08

This slice replaces single-line prompt inputs with native multiline editors and
keeps property widgets mounted while their node/configuration binding stays the
same. It builds on the preceding workflow-interactions slice; no model inference
or model downloads were needed.

## Behavior and ownership

- `prompt` and `negative_prompt` use Textual TextArea with soft wrapping, native
  selection, clipboard, scrolling, and undo/redo. Other fields keep their input
  or select control. Property widgets and their layout live in
  `aigen/workflow_property_widgets.py`; the inspector owns visibility/options.
- Enter inserts a newline, Ctrl+Enter applies property edits, and Ctrl+S applies
  and saves. Tab/Shift+Tab navigate fields. F7 selects all text. Graph undo/redo
  is scoped away from text editors, including when local history is empty.
- Commits update property baselines without rebuilding editors. Backend and
  seed-mode changes update values, choices, and visibility in place. Native text
  history is retained when the committed text matches the editor; an external
  graph change loads the new text. Node/document replacement creates a new
  binding. There is no parallel draft-text cache or custom text-history engine.
- Editable properties precede backend descriptions. The status line displays
  text-editing keys while a multiline field has focus. Wide and narrow layouts
  retain the same editor, selection, scroll position, and undo history.
- An explicit Character edit backend change to Klein now resets `pose_mode` to
  `native` alongside the existing backend defaults. Previously, Qwen's keypoint
  choice remained in the document and conflicted with the native-only selector.
  Graph undo restores the full previous Qwen configuration.

## Reproductions and dependency fix

The original workflow Paste-event reproduction retained only the first pasted
line. Textual Input's paste handler explicitly selects `splitlines()[0]`.
The replacement was checked with actual Paste and keyboard events, including
blank lines, a final newline, leading/trailing spaces, and Unicode.

The long-paste undo test also reproduced Textual issue
[#6686](https://github.com/Textualize/textual/issues/6686): scrollbar refresh
observed a cursor row from the longer document while undo restored the shorter
one. The project pins Textual commit
[`aff7b76b5157c38a87ccf0aec1f57df63f398ac5`](https://github.com/Textualize/textual/commit/aff7b76b5157c38a87ccf0aec1f57df63f398ac5)
from [PR #6687](https://github.com/Textualize/textual/pull/6687). It bounds the
visual cursor calculation while leaving the actual selection/history intact.
The PR was still open and unreleased when inspected. Relative to v8.2.8 there
are three commits: the fix/test plus comment/docstring typo corrections.
`textual-source.json` records the installed source identity.

Pip skips a VCS replacement with the same package version, even with --upgrade.
The installer therefore compares the installed source commit with the canonical
pyproject pin before dependency installation. Missing/index/different-source
installations receive the pinned source with --force-reinstall --no-deps; the
correct installed commit is skipped. `installation-checks.log` records these
cases. README documents upgrading an existing environment.

Reference implementations studied before implementation:

- [Textual TextArea](https://github.com/Textualize/textual/blob/main/src/textual/widgets/_text_area.py): native document/edit history, soft wrap, focus-oriented Tab handling, and load_text resetting history.
- [VS Code settings renderers](https://github.com/microsoft/vscode/blob/main/src/vs/workbench/contrib/preferences/browser/settingsTree.ts): separate multiline controls and retained editor templates with value rebinding.

## Validation

`application-tests.log`: **121 tests passed in 90.576 seconds**. This includes
eight new text-interaction tests, one backend-default regression test, and the
existing workflow, results, subprocess, image/video/mask, provenance, and native
conditioning contract tests. In particular:

- Lossless multiline paste through apply/save/reload, without an opening edit.
- Local undo/redo, Enter/Tab navigation, clipboard copy, and deletion ownership.
- Widget/selection/history retention through commits, backend changes, and resizes.
- Atomic rejection of invalid numeric input while retaining the text draft.
- Conditional fields/options, node switches, graph undo/redo, and document replacement.
- Shift+F5 reaches a real CPU subprocess from the text editor and drains its output.

The earlier 29-test focused run found the keypoint-to-Klein default bug; its
correction passed focused checks and the subsequent full 121-test run.
Compileall, shell syntax, and git diff --check passed.

An independent read-only review found no remaining blockers in this slice. Its
standalone native 80x24 long-paste/undo check also restored text and selection
correctly across a scrollbar visibility change.

The global `pip check` still reports FLUX dependency-version declarations and
missing optional packages, plus decord platform metadata. No Textual dependency
conflict was reported. This slice changed only Textual and the editable aigen
package metadata in the environment; the generation packages were not changed.

Both screenshots were rendered with Chromium and visually inspected:

- `properties-160x50.svg` / `properties-160x50.png`
- `properties-80x24.svg` / `properties-80x24.png`

No GPU quality claim is made by these CPU/UI tests. Existing inkstyle assets and
other run artifacts were left untouched. This slice has not been committed.
