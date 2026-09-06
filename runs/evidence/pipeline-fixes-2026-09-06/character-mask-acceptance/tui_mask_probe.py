"""Actual SAM form -> workflow -> native masked Qwen -> audit/cache/export."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from textual.widgets import Input, TabbedContent
from aigen import image_tui
from aigen.manifest_io import atomic_write_json, sha256_file, file_manifest
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import NodeKind, NodePortRef
from aigen.workflow_results import load_node_result, node_result_history

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'image-flow-acceptance'))
from tui_probe import run_action, select_node

async def run(stage, manifest=None, attempt='tui-v1'):
    manifest = manifest or Path(__file__).resolve().with_name('manifest.json')
    root=manifest.resolve().parent
    request=json.loads(manifest.read_text())
    assert sha256_file(Path(request['source']['path']))==request['source']['sha256']
    directory=root/attempt
    directory.mkdir(exist_ok=True)
    graph_path=directory/'workflow.json'
    image_tui.STATE_PATH=directory/'form.json'
    image_tui.DEFAULT_WORKFLOW_RUNS_ROOT=directory/'workflows'
    app=image_tui.ImageGenerationApp()
    if stage=='segment':
        app.sam_form.field('input').value=request['source']['path']
        for key,value in request['sam'].items():
            app.sam_form.set_value(app.sam_form.field(key),str(value).lower() if isinstance(value,bool) else str(value))
    else:
        if stage=='refine':
            assert request['review_status']=='approved'
            assert sha256_file(Path(request['mask']['path']))==request['mask']['sha256']
        app.workflow_buffer=WorkflowEditBuffer(load_workflow_document(graph_path),document_path=graph_path)
        app.workflow_document_paths=[graph_path]
    async with app.run_test(size=(160,54)) as pilot:
        if stage=='segment':
            app.query_one('#tabs',TabbedContent).active='sam-edit'
            await pilot.pause()
            assert await pilot.click('#sam-workflow')
            await pilot.pause()
            buffer=app.workflow_buffer
            node=buffer.add_node(NodeKind.CHARACTER_REFINE, x=82, y=2)
            for key,value in request['edit'].items():
                buffer.update_node_config(node.id,key,value)
            buffer.connect_ports(NodePortRef(node_id='source',port='image'),NodePortRef(node_id=node.id,port='source'))
            buffer.connect_ports(NodePortRef(node_id='segment',port='mask'),NodePortRef(node_id=node.id,port='mask'))
            buffer.auto_layout()
            save_workflow_document(buffer.document,graph_path)
            buffer.mark_saved(graph_path)
            atomic_write_json(directory/'nodes.json',{'refine':node.id})
            app._open_workflow_editor()
        else:
            app._open_workflow_editor()
        await pilot.pause()
        editor=app.workflow_editor
        node_id='segment' if stage.startswith('segment') else json.loads((directory/'nodes.json').read_text())['refine']
        await select_node(app,editor,pilot,node_id)
        generated=await run_action(app,editor,pilot,'#workflow-run-target',directory,stage,request['min_free_mib'])
        if stage=='refine':
            cached=await run_action(app,editor,pilot,'#workflow-run-target',directory,'refine-cache',request['min_free_mib'])
            assert cached['statuses'][node_id]=='reused'
        else:
            cached=None
        assert await pilot.click('#workflow-results')
        await pilot.pause()
        results=app.screen
        await results.workers.wait_for_complete()
        if stage.startswith('segment'):
            index,candidate=next((i,c) for i,c in enumerate(results.candidates) if getattr(c,'port',None)=='mask')
            results.inspect_candidate(candidate,index,results._view_id)
            await results.workers.wait_for_complete()
        (directory/f'{stage}-results.svg').write_text(app.export_screenshot())
        assert await pilot.click('#result-export')
        await pilot.pause()
        output=directory/('mask-original.png' if stage.startswith('segment') else 'accepted.png')
        app.screen.query_one(Input).value=str(output)
        assert await pilot.click('#dialog-ok')
        await pilot.pause()
        await results.workers.wait_for_complete()
        graph=app.workflow_buffer.document
        manifest=node_result_history(directory/'workflows',graph.workflow_id,node_id)[0]
        result=load_node_result(manifest)
        port='mask' if stage.startswith('segment') else 'image'
        assert sha256_file(output)==result.outputs[port].content_sha256
        atomic_write_json(directory/f'{stage}-acceptance.json',{'generation':generated,'cache':cached,'result':result.model_dump(mode='json'),'export':file_manifest(output)})
        if stage.startswith('segment'):
            request['mask']=file_manifest(output)
            request['sam_result']=str(manifest)
            atomic_write_json(root/'manifest.json',request)
        print(json.dumps({'status':'completed','stage':stage,'output':str(output)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('segment','segment-resume','refine'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--attempt', default='tui-v1')
    args=parser.parse_args()
    asyncio.run(run(args.stage,args.manifest,args.attempt))
