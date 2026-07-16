#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one patch target, found {count}")
    return source.replace(old, new, 1)


def patch_trainer(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    model_argument = '''    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
'''
    text_encoder_argument = model_argument + '''    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default=None,
        help="Optional standalone text encoder used to precompute frozen prompt embeddings.",
    )
'''
    source = replace_once(
        source,
        model_argument,
        text_encoder_argument,
        "standalone text encoder argument",
    )

    tokenizer_load = '''    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
'''
    tokenizer_load_from_conditioner = '''    text_encoder_model_path = (
        args.pretrained_text_encoder_name_or_path or args.pretrained_model_name_or_path
    )
    text_encoder_subfolder = "" if args.pretrained_text_encoder_name_or_path else "text_encoder"
    tokenizer_subfolder = "" if args.pretrained_text_encoder_name_or_path else "tokenizer"
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        text_encoder_model_path,
        subfolder=tokenizer_subfolder,
        revision=args.revision,
    )
'''
    source = replace_once(
        source,
        tokenizer_load,
        tokenizer_load_from_conditioner,
        "standalone text encoder tokenizer",
    )

    eager_text_encoder_load = '''    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)

'''
    source = replace_once(
        source,
        eager_text_encoder_load,
        "",
        "deferred text encoder load",
    )

    eager_transformer_move = '''    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)
'''
    source = replace_once(
        source,
        eager_transformer_move,
        '''    if not is_fsdp and not args.offload:
        transformer.to(**transformer_to_kwargs)
''',
        "deferred transformer move",
    )

    adapter_install = '''    transformer.add_adapter(transformer_lora_config)
'''
    adapter_install_then_transformer_offload = adapter_install + '''
    if not is_fsdp and args.offload and args.bnb_quantization_config_path is not None:
        transformer.to("cpu")
        free_memory()

    text_encoder_kwargs = {
        "subfolder": text_encoder_subfolder,
        "revision": args.revision,
        "variant": args.variant,
    }
    if args.pretrained_text_encoder_name_or_path is not None:
        text_encoder_kwargs.update(
            dtype="auto",
            device_map=accelerator.device,
            low_cpu_mem_usage=True,
        )
    text_encoder = Qwen3ForCausalLM.from_pretrained(text_encoder_model_path, **text_encoder_kwargs)
    text_encoder.requires_grad_(False)
    if args.pretrained_text_encoder_name_or_path is None:
        text_encoder.to(**to_kwargs)
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )
    if args.pretrained_text_encoder_name_or_path is not None:
        text_encoding_pipeline.to("cpu")
        free_memory()
'''
    source = replace_once(
        source,
        adapter_install,
        adapter_install_then_transformer_offload,
        "quantized transformer cache offload",
    )

    eager_text_encoder_pipeline = '''    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

'''
    source = replace_once(
        source,
        eager_text_encoder_pipeline,
        "",
        "deferred text encoder pipeline",
    )

    cache_argument = '''    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
'''
    cache_argument_with_persistence = cache_argument + '''    parser.add_argument(
        "--precomputed_cache_path",
        type=str,
        default=None,
        help="Persist and reuse the precomputed VAE latents and per-image prompt embeddings.",
    )
'''
    source = replace_once(
        source,
        cache_argument,
        cache_argument_with_persistence,
        "persistent-cache argument",
    )

    parse_validation = '''    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")
'''
    parse_validation_with_cache = parse_validation + '''    if args.precomputed_cache_path is not None and not args.cache_latents:
        raise ValueError("`--precomputed_cache_path` requires `--cache_latents`.")
    if args.precomputed_cache_path is not None and args.with_prior_preservation:
        raise ValueError("Persistent precomputed caches do not support prior preservation.")
'''
    source = replace_once(
        source,
        parse_validation,
        parse_validation_with_cache,
        "persistent-cache validation",
    )

    cache_block_start = source.index("    # if cache_latents is set to True, we encode images to latents and store them.\n")
    cache_block_end = source.index(
        "    # move back to cpu before deleting to ensure memory is freed see:",
        cache_block_start,
    )
    old_cache_block = source[cache_block_start:cache_block_end]
    new_cache_block = '''    # Cache image latents and per-image prompt embeddings. The persistent form makes
    # checkpointed training resumable without re-running the expensive frozen encoders.
    persistent_cache_path = Path(args.precomputed_cache_path) if args.precomputed_cache_path else None
    persistent_cache_signature = None
    persistent_cache_loaded = False
    if persistent_cache_path is not None:
        if train_dataset.custom_instance_prompts is None:
            raise ValueError("Persistent precomputed caches require per-image captions.")
        signature = insecure_hashlib.sha256()
        signature.update(
            json.dumps(
                {
                    "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                    "pretrained_text_encoder_name_or_path": args.pretrained_text_encoder_name_or_path,
                    "revision": args.revision,
                    "resolution": args.resolution,
                    "buckets": buckets,
                    "center_crop": args.center_crop,
                    "random_flip": args.random_flip,
                    "repeats": args.repeats,
                    "max_sequence_length": args.max_sequence_length,
                    "text_encoder_out_layers": args.text_encoder_out_layers,
                    "image_count": train_dataset.num_instance_images,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for (pixel_values, bucket_idx), prompt in zip(
            train_dataset.pixel_values,
            train_dataset.custom_instance_prompts,
            strict=True,
        ):
            signature.update(memoryview(pixel_values.contiguous().numpy()))
            signature.update(str(bucket_idx).encode("ascii"))
            signature.update(prompt.encode("utf-8"))
        persistent_cache_signature = signature.hexdigest()

        if persistent_cache_path.is_file():
            cached = torch.load(
                persistent_cache_path,
                map_location=accelerator.device,
                weights_only=True,
            )
            if cached["signature"] != persistent_cache_signature:
                raise ValueError(
                    f"Precomputed training cache does not match this dataset and preprocessing: {persistent_cache_path}"
                )
            instance_latents_cache = cached["instance_latents"]
            prompt_embeds_cache = cached["prompt_embeds"]
            text_ids_cache = cached["text_ids"]
            persistent_cache_loaded = True
            logger.info(f"Loaded precomputed training cache from {persistent_cache_path}")

    if not persistent_cache_loaded:
        if args.cache_latents:
            instance_latents_cache = [None] * train_dataset.num_instance_images
            class_latents_cache = (
                [None] * train_dataset.num_instance_images if args.with_prior_preservation else None
            )
        if train_dataset.custom_instance_prompts:
            prompt_embeds_cache = [None] * train_dataset.num_instance_images
            text_ids_cache = [None] * train_dataset.num_instance_images
        if precompute_latents:
            cache_batch_sampler = BucketBatchSampler(
                train_dataset, batch_size=args.train_batch_size, drop_last=False, seed=args.seed
            )
            cache_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=cache_batch_sampler,
                collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
                num_workers=args.dataloader_num_workers,
            )
        if args.cache_latents:
            with offload_models(vae, device=accelerator.device, offload=args.offload), torch.no_grad():
                for batch in tqdm(cache_dataloader, desc="Caching latents"):
                    sample_indices = batch["indices"]
                    batch["pixel_values"] = batch["pixel_values"].to(
                        accelerator.device, non_blocking=True, dtype=vae.dtype
                    )
                    latents = vae.encode(batch["pixel_values"]).latent_dist.mode()
                    if args.with_prior_preservation:
                        instance_latents, class_latents = torch.chunk(latents, 2, dim=0)
                    else:
                        instance_latents = latents
                    for i, idx in enumerate(sample_indices):
                        instance_latents_cache[idx] = instance_latents[i : i + 1]
                        if args.with_prior_preservation:
                            class_latents_cache[idx] = class_latents[i : i + 1]
            assert all(latents is not None for latents in instance_latents_cache), (
                "Latent cache has unfilled entries."
            )
            if args.with_prior_preservation:
                assert all(latents is not None for latents in class_latents_cache), (
                    "Class latent cache has unfilled entries."
                )
        if train_dataset.custom_instance_prompts:
            text_encoder_context = (
                nullcontext()
                if args.fsdp_text_encoder
                else offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload)
            )
            with text_encoder_context, torch.no_grad():
                for batch in tqdm(cache_dataloader, desc="Caching prompts"):
                    sample_indices = batch["indices"]
                    prompt_embeds, text_ids = compute_text_embeddings(
                        batch["instance_prompts"], text_encoding_pipeline
                    )
                    for i, idx in enumerate(sample_indices):
                        prompt_embeds_cache[idx] = prompt_embeds[i : i + 1]
                        text_ids_cache[idx] = text_ids[i : i + 1]
            assert all(embeds is not None for embeds in prompt_embeds_cache), (
                "Prompt embedding cache has unfilled entries."
            )
            assert all(ids is not None for ids in text_ids_cache), "Text ID cache has unfilled entries."

        if persistent_cache_path is not None and accelerator.is_main_process:
            persistent_cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache_path = persistent_cache_path.with_suffix(persistent_cache_path.suffix + ".tmp")
            torch.save(
                {
                    "signature": persistent_cache_signature,
                    "instance_latents": [tensor.detach().cpu() for tensor in instance_latents_cache],
                    "prompt_embeds": [tensor.detach().cpu() for tensor in prompt_embeds_cache],
                    "text_ids": [tensor.detach().cpu() for tensor in text_ids_cache],
                },
                temporary_cache_path,
            )
            os.replace(temporary_cache_path, persistent_cache_path)
            logger.info(f"Saved precomputed training cache to {persistent_cache_path}")
        accelerator.wait_for_everyone()

'''
    source = source[:cache_block_start] + new_cache_block + source[cache_block_end:]

    encoder_release = '''    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoder, tokenizer
    free_memory()

'''
    encoder_release_then_transformer_move = encoder_release + '''    if not is_fsdp and args.offload:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        transformer = torch.compile(transformer)

'''
    source = replace_once(
        source,
        encoder_release,
        encoder_release_then_transformer_move,
        "post-cache transformer move",
    )
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trainer", type=Path)
    args = parser.parse_args()
    patch_trainer(args.trainer)


if __name__ == "__main__":
    main()
