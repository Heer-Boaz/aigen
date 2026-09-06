"""Diagnostic: retain native LightX2V offload; disable only extra local resident blocks."""
import json
from pathlib import Path
import sys
import traceback
from threading import Event, Thread
import time
from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.generation import qwen_image_edit_lightx2v_worker as worker
import torch

gpu=nvidia_smi_memory_snapshot()
print(json.dumps({'gpu_preflight':gpu}),flush=True)
assert gpu['nvidia_smi_device_total_mb']-gpu['nvidia_smi_used_mb']>=12000
source=Path(sys.argv[1])
dest=source.parent.parent/sys.argv[2]
dest.mkdir()
request=json.loads(source.read_text())
request['cases'][0]['outputs'][0]['path']=str(dest/'image.png')
(dest/'request.json').write_text(json.dumps(request,indent=2)+'\n')
def native_streaming(torch, runner, max_tokens):
    print(json.dumps({'diagnostic':'native double-buffer block offload','extra_resident_blocks':0}),flush=True)
    return [], {"max_denoise_tokens": max_tokens, "diagnostic_extra_resident_blocks": 0}
worker._enable_resident_blocks=native_streaming
stop = Event()
started = time.monotonic()
def monitor():
    with (dest/'telemetry.jsonl').open('w',buffering=1) as log:
        while not stop.is_set():
            log.write(json.dumps({'elapsed_seconds':time.monotonic()-started,
                **nvidia_smi_memory_snapshot(),
                'torch_allocated_mib':torch.cuda.memory_allocated()/1024**2,
                'torch_reserved_mib':torch.cuda.memory_reserved()/1024**2,
                'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})+'\n')
            stop.wait(1)
thread=Thread(target=monitor,daemon=True)
thread.start()
try:
    response=worker._run(request,sys.stdout)
except BaseException as error:
    traceback.print_exc()
    response={'status':'error','error':type(error).__name__,'message':str(error)}
    raise
finally:
    stop.set()
    thread.join()
    response['diagnostic_peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
    (dest/'response.json').write_text(json.dumps(response,indent=2)+'\n')
