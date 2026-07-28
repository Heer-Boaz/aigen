from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from aigen.pix2pix.dataset import PairedImage
from aigen.pix2pix.image_io import load_rgb_tensor


class PairedImageDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, pairs: Sequence[PairedImage], *, image_size: int) -> None:
        self._pairs = pairs
        self._image_size = image_size

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self,
        key: int | tuple[int, bool],
    ) -> dict[str, Tensor | str]:
        if isinstance(key, tuple):
            index, flip = key
        else:
            index, flip = key, False
        pair = self._pairs[index]
        source = load_rgb_tensor(pair.source_path, image_size=self._image_size)
        target = load_rgb_tensor(pair.target_path, image_size=self._image_size)
        if flip:
            source = source.flip(-1)
            target = target.flip(-1)
        return {"id": pair.id, "source": source, "target": target}


class EpochPairSampler(Sampler[tuple[int, bool]]):
    def __init__(
        self,
        pair_count: int,
        *,
        seed: int,
        horizontal_flip: bool,
    ) -> None:
        self._pair_count = pair_count
        self._seed = seed
        self._horizontal_flip = horizontal_flip
        self._epoch = 0
        self._start_offset = 0

    def set_position(self, *, epoch: int, start_offset: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not 0 <= start_offset < self._pair_count:
            raise ValueError("start_offset must identify a sample in the epoch")
        self._epoch = epoch
        self._start_offset = start_offset

    def __iter__(self) -> Iterator[tuple[int, bool]]:
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._epoch)
        order = torch.randperm(self._pair_count, generator=generator).tolist()
        flips = (
            torch.randint(0, 2, (self._pair_count,), generator=generator).tolist()
            if self._horizontal_flip
            else None
        )
        for position in range(self._start_offset, self._pair_count):
            yield order[position], bool(flips[position]) if flips is not None else False

    def __len__(self) -> int:
        return self._pair_count - self._start_offset


def create_training_loader(
    pairs: Sequence[PairedImage],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    horizontal_flip: bool,
    pin_memory: bool,
) -> tuple[DataLoader[dict[str, Tensor | str]], EpochPairSampler]:
    dataset = PairedImageDataset(pairs, image_size=image_size)
    sampler = EpochPairSampler(
        len(pairs),
        seed=seed,
        horizontal_flip=horizontal_flip,
    )
    worker_generator = torch.Generator()
    worker_generator.manual_seed(seed)
    options: dict[str, object] = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": worker_generator,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options), sampler


def create_evaluation_loader(
    pairs: Sequence[PairedImage],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Tensor | str]]:
    dataset = PairedImageDataset(pairs, image_size=image_size)
    worker_generator = torch.Generator()
    worker_generator.manual_seed(0)
    options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": worker_generator,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)
