from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.manifest_io import ManifestIOError
from aigen.workflow_cache import WorkflowCacheError
from aigen.progress import StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT
from aigen.workflow_compilation import (
    compile_workflow,
    compile_workflow_run,
)
from aigen.workflow_document_io import (
    load_workflow_document,
    save_workflow_document,
)
from aigen.workflow_execution import (
    WorkflowExecutionError,
    WorkflowInterrupted,
    execute_workflow,
    format_workflow_event,
)
from aigen.workflow_templates import character_workflow_template, image_style_workflow_template, keyframed_video_workflow_template


DEFAULT_WORKFLOW_RUNS_ROOT = PROJECT_ROOT / "runs" / "workflows"


def add_workflow_commands(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "workflow",
        help="Create, validate, and execute visual workflows",
    )
    operations = command.add_subparsers(
        dest="workflow_operation",
        required=True,
    )

    new = operations.add_parser(
        "new",
        help="Write a new image, character or keyframed-video workflow document",
    )
    new.add_argument("--output", type=Path, required=True)
    new.add_argument("--template", choices=("image", "character", "video"), default="image")

    validate = operations.add_parser(
        "validate",
        help="Validate a workflow document for execution",
    )
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--target", action="append", dest="targets", help="Execute only this node and its dependencies; repeat for multiple targets")

    run = operations.add_parser(
        "run",
        help="Execute a workflow with exact-node resume",
    )
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--target", action="append", dest="targets", help="Execute only this node and its dependencies; repeat for multiple targets")
    run.add_argument("--run-id", help="Preassigned execution ID for a supervising caller")
    run.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_WORKFLOW_RUNS_ROOT,
    )


def run_workflow_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        if args.workflow_operation == "new":
            output = args.output.expanduser().resolve()
            if output.exists():
                raise WorkflowExecutionError(
                    f"workflow document already exists: {output}"
                )
            graph = {"image": image_style_workflow_template, "character": character_workflow_template,
                     "video": keyframed_video_workflow_template}[args.template]()
            save_workflow_document(graph, output)
            payload = {
                "status": "completed",
                "kind": "workflow-document",
                "path": output.as_posix(),
            }
        elif args.workflow_operation == "validate":
            path = args.input.expanduser().resolve()
            workflow = compile_workflow(load_workflow_document(path), target_node_ids=args.targets)
            payload = {
                "status": "completed",
                "kind": "workflow-validation",
                "path": path.as_posix(),
                "workflow_digest": workflow.digest,
                "execution_order": list(workflow.execution_order),
            }
        elif args.workflow_operation == "run":
            workflow = compile_workflow_run(
                load_workflow_document(args.input.expanduser().resolve()),
                target_node_ids=args.targets,
            )
            forward_progress = os.environ.get("AIGEN_PROGRESS") == "json"

            def emit_event(event: dict[str, object]) -> None:
                if not forward_progress:
                    return
                stdout.write(format_workflow_event(event) + "\n")
                stdout.flush()

            def emit_node_progress(
                node_id: str,
                node_progress: dict[str, object],
            ) -> None:
                if not forward_progress:
                    return
                emit_event(
                    {
                        "node_id": node_id,
                        "node_kind": workflow.node(node_id).node.kind,
                        "status": "running",
                        "progress": node_progress,
                    }
                )

            previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

            def interrupt_workflow(
                _signum: int,
                _frame: object,
            ) -> None:
                raise WorkflowInterrupted("workflow interrupted")

            try:
                for signum in previous_handlers:
                    signal.signal(signum, interrupt_workflow)
                run_result = execute_workflow(
                    workflow,
                    runs_root=args.runs_root,
                    progress=progress,
                    event_sink=emit_event,
                    node_progress_sink=emit_node_progress,
                    run_id=args.run_id,
                )
                payload = run_result.to_json()
                emit_event(
                    {
                        "kind": payload["kind"],
                        "status": payload["status"],
                        "run_dir": payload["run_dir"],
                        "result": payload["result"],
                    }
                )
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        else:
            raise RuntimeError("unsupported workflow operation")
    except KeyboardInterrupt as error:
        dump_json(stderr, command_error_payload(WorkflowInterrupted(str(error) or "workflow interrupted")), pretty=True)
        return 130
    except (OSError, ValueError, ManifestIOError, WorkflowCacheError, WorkflowExecutionError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1

    dump_json(stdout, payload, pretty=True)
    return 0
