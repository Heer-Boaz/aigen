from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from aigen.generation import qwen_image_edit_lightx2v_worker as worker


class QwenResidencyTests(unittest.TestCase):
    def test_reference_and_longest_text_tokens_determine_largest_case(self):
        cases = [
            {"name": "many_references", "width": 512, "height": 512},
            {"name": "larger_output", "width": 1024, "height": 1024},
        ]
        conditions = {
            "many_references": {"vae_group": ("one", "two"), "txt_seq_lens": [121, 321]},
            "larger_output": {"vae_group": ("three",), "txt_seq_lens": [730]},
        }
        groups = {
            ("one", "two"): [{"image_latents": torch.empty((1, length, 64), device="meta")}
                             for length in (5, 13000)],
            ("three",): [{"image_latents": torch.empty((1, 97, 64), device="meta")}],
        }
        self.assertEqual(worker._max_denoise_tokens(cases, conditions, groups), 14350)
        self.assertEqual(worker._max_denoise_tokens(cases[::-1], conditions, groups), 14350)

    def test_weight_aliases_and_views_share_one_storage_budget(self):
        shared = torch.empty((8, 16), dtype=torch.bfloat16)
        independent = shared.clone()
        first = SimpleNamespace(weight=shared, buffer=shared.t())
        second = SimpleNamespace(weight=shared[2:], bias=independent)
        module = SimpleNamespace(_modules={"first": first, "second": second}, _parameters={})
        second._modules = {"cycle": module}
        self.assertEqual(worker._weight_storage_bytes(torch, module, first, device="cpu"), 512)
        self.assertEqual(worker._weight_storage_bytes(torch, module, device="cuda"), 0)

    def test_no_clone_or_inference_replacement_when_first_block_does_not_fit(self):
        for remaining in (-100, 339):
            with self.subTest(remaining=remaining):
                runtime, runner, original = self.runtime_with_budget(remaining)
                with patch.object(worker, "_weight_storage_bytes", side_effect=[100, 340]), \
                     patch("copy.deepcopy") as clone:
                    blocks, budget = worker._enable_resident_blocks(runtime, runner, 4096)
                self.assertEqual(blocks, [])
                self.assertEqual(budget["resident_budget_bytes"], remaining)
                self.assertEqual(budget["late_weight_bytes"], 100)
                self.assertIs(runner.model.transformer_infer.infer_func, original)
                clone.assert_not_called()

    def test_every_clone_is_admitted_before_allocation(self):
        runtime, runner, original = self.runtime_with_budget(2 * 340 + 339)
        buffers = [Mock(), Mock()]
        with patch.object(worker, "_weight_storage_bytes", side_effect=[100, 340]), \
             patch("copy.deepcopy", side_effect=buffers) as clone:
            blocks, budget = worker._enable_resident_blocks(runtime, runner, 4096)
        self.assertEqual(blocks, buffers)
        self.assertEqual(clone.call_count, 2)
        self.assertIsNot(runner.model.transformer_infer.infer_func, original)
        for index, buffer in enumerate(buffers):
            buffer.load_state_dict.assert_called_once_with({"block": index}, index, None)
        self.assertEqual(budget["block_bytes"], 340)

    @staticmethod
    def runtime_with_budget(remaining):
        free = remaining + 100 + 4096 * worker.DENOISE_WORKSPACE_BYTES_PER_TOKEN + worker.DENOISE_HEADROOM_BYTES
        runtime = SimpleNamespace(cuda=Mock())
        runtime.cuda.mem_get_info.return_value = (free, 16 * 1024**3)
        config = SimpleNamespace(temporarily_unlocked=nullcontext)
        template = SimpleNamespace(config=config)
        original = Mock()
        infer = SimpleNamespace(offload_manager=SimpleNamespace(cuda_buffers=[template]), infer_func=original)
        blocks = [Mock() for _ in range(3)]
        for index, block in enumerate(blocks):
            block.state_dict.return_value = {"block": index}
        runner = SimpleNamespace(config=config, model=SimpleNamespace(
            config=config, transformer_infer=infer, transformer_weights=SimpleNamespace(blocks=blocks),
            pre_weight=Mock(), post_weight=Mock()))
        return runtime, runner, original


if __name__ == "__main__":
    unittest.main()
