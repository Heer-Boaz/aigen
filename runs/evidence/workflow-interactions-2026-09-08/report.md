# Workflow interaction and responsive UI — 2026-09-08

The workflow editor now supports selection and dragging across the node body,
port connections and reconnection, selection-specific menus, and a single
document toolbar. Earlier/Later controls and their unused directional ordering
API were removed. Multiple-input order is edited beside the selected wire and
persisted as an insertion in one undo step.

## Verified causes and fixes

- The previous mouse handler only started dragging on a node's title row.
  Body drags now work; port hit tests retain precedence.
- Textual's text selection could consume body gestures. The canvas disables
  text selection and owns click propagation, including double-clicks.
- Committing a property after a gesture started replaced the canvas document
  and cancelled that gesture. Valid property drafts are now committed before
  hit testing and capture. Invalid drafts focus the invalid input and reveal
  its inspector when the drawer was closed.
- Asynchronous move messages lost rapid key repeats, and old selection messages
  could temporarily show the wrong inspector. The canvas now awaits selection,
  movement, and connection updates at the editor boundary before accepting the
  next input. The edit buffer remains the document/history owner.
- A delayed MouseRelease event could cancel a new gesture that already owned
  capture. The release handler now consults Textual's current capture owner.
- Drag deltas use canvas coordinates, including changes in scrolling during a
  gesture. The existing retained scene, spatial hit indexes and row refreshes
  remain responsible for previews.
- Inspector layout changes retain the same widget and property drafts. Wide
  terminals use a split view; narrow terminals use a right-side drawer without
  reducing the canvas to a short row.
- Document and context-menu commands use the same action availability and
  handlers. The inspector has one compact **⋯** menu in its heading; duplicate
  Run/Results buttons were removed after the user's live feedback. The node
  picker uses native OptionList keyboard
  and mouse interaction with filtering.
- The connection dialog used noncompact buttons in a single-row container,
  leaving their content area zero rows high. Native compact buttons restore
  visible Cancel/Connect labels. A regression test checks the rendered label
  and clicks Connect to verify the resulting graph connection.
- Run becomes Stop. Shift+F5 belongs to the application process owner and works
  through modal menus. Form deletion shortcuts are scoped to the form screen;
  editing property text cannot delete nodes or undo the workflow.

## Validation

The full existing application bundle plus the new interaction suite passed:
**112 tests in 85.898 seconds**. See [application-tests.log](application-tests.log).
The 15 new tests exercise Textual Pilot mouse and keyboard events, with direct
Screen forwarding for batches that intentionally omit Pilot's usual event
flush. They check saved documents and undo state, rather than calling canvas
selection setters as a substitute for mouse interaction.

Covered cases include body drag, click/double-click, rapid release/recapture,
queued key repeats, rapid selection with a property draft, valid/invalid
drafts, Escape/capture loss, scrolled drag, port connection/reconnection,
context-menu targeting, rendered dialog button labels, input key scope, input order/save/undo, node search,
resizing from 60×20 through 160×50, and stopping a real CPU subprocess through
an open menu. The broader bundle covers the existing image/video/character
workflow, results, source snapshot, and process lifecycle contracts.

Two independent read-only reviews reproduced the input-ordering issues before
the fixes and confirmed the corrected sequences afterwards. Stop through a
menu was independently checked at 80×24, 120×40 and 160×50 with real CPU
processes: exit -15, stdout drained and generation worker cleared. The 80×24
drawer close button was hit-tested and visibly inspected.

`python -m compileall -q aigen tests` and `git diff --check` passed. No model
generation was needed for these UI changes.

## Screenshots

These snapshots use the repository's existing image-style workflow template.
SVGs were exported by Textual, and PNGs were rendered in Chromium with GPU
acceleration disabled. Canvas, inspector, context menu and node picker were
visually inspected.

- [Wide canvas, 160×50](canvas-160x50.png) ([SVG](canvas-160x50.svg))
- [Split layout, 120×40](canvas-120x40.svg)
- [Canvas, 80×24](canvas-80x24.png) ([SVG](canvas-80x24.svg))
- [Inspector drawer, 80×24](inspector-80x24.png) ([SVG](inspector-80x24.svg))
- [Node context menu, 80×24](context-80x24.png) ([SVG](context-80x24.svg))
- [Searchable node picker, 80×24](add-80x24.png) ([SVG](add-80x24.svg))
- [Connection dialog, 80×24](connect-80x24.png) ([SVG](connect-80x24.svg))

## Production references studied before implementation

- [XYFlow drag ownership](https://github.com/xyflow/xyflow/blob/main/packages/system/src/xydrag/XYDrag.ts):
  coordinate transforms, selection before drag, transient movement and final
  updates at gesture completion.
- [Textual ScrollView](https://github.com/Textualize/textual/blob/main/src/textual/scroll_view.py)
  and [DataTable](https://github.com/Textualize/textual/blob/main/src/textual/widgets/_data_table.py):
  retained line rendering, virtual content dimensions, and explicit ownership
  of interaction on selectable line-based widgets.
- [VS Code toolbar](https://github.com/microsoft/vscode/blob/main/src/vs/platform/actions/browser/toolbar.ts):
  primary/secondary action placement and shared contextual command handling.
