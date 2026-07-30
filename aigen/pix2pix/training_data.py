from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from aigen.pix2pix.dataset import PairedImage
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.image_io import load_rgb_tensor


class PairedImageDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        pairs: Sequence[PairedImage],
        *,
        image_size: int,
        target_palettes: Sequence[Tensor] | None = None,
    ) -> None:
        self._pairs = pairs
        self._image_size = image_size
        self._target_palettes = target_palettes

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self,
        key: int | tuple[int, bool] | tuple[int, bool, int, bool],
    ) -> dict[str, Tensor | str]:
        if not isinstance(key, tuple):
            index = key
            flip = False
            mismatched_index = None
            mismatched_flip = False
        elif len(key) == 2:
            index, flip = key
            mismatched_index = None
            mismatched_flip = False
        else:
            index, flip, mismatched_index, mismatched_flip = key
        pair = self._pairs[index]
        source = load_rgb_tensor(pair.source_path, image_size=self._image_size)
        target = load_rgb_tensor(pair.target_path, image_size=self._image_size)
        if flip:
            source = source.flip(-1)
            target = target.flip(-1)
        sample: dict[str, Tensor | str] = {
            "id": pair.id,
            "source": source,
            "target": target,
        }
        if mismatched_index is not None:
            mismatched_pair = self._pairs[mismatched_index]
            mismatched_source = load_rgb_tensor(
                mismatched_pair.source_path,
                image_size=self._image_size,
            )
            if mismatched_flip:
                mismatched_source = mismatched_source.flip(-1)
            sample["mismatched_source"] = mismatched_source
        if self._target_palettes is not None:
            sample["target_palette"] = self._target_palettes[index]
        return sample


TrainingSampleKey = tuple[int, bool] | tuple[int, bool, int, bool]


class EpochPairSampler(Sampler[TrainingSampleKey]):
    def __init__(
        self,
        groups: Sequence[str],
        *,
        seed: int,
        horizontal_flip: bool,
        mismatched_source: bool,
    ) -> None:
        self._groups = tuple(groups)
        self._pair_count = len(groups)
        self._derangement_shift = 0
        if mismatched_source:
            largest_group = max(Counter(groups).values())
            if largest_group > self._pair_count - largest_group:
                raise Pix2PixError(
                    "a-contrario sampling requires every training group to have "
                    "at least as many pairs outside the group"
                )
            self._derangement_shift = largest_group
        self._seed = seed
        self._horizontal_flip = horizontal_flip
        self._mismatched_source = mismatched_source
        self._epoch = 0
        self._start_offset = 0

    def set_position(self, *, epoch: int, start_offset: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not 0 <= start_offset < self._pair_count:
            raise ValueError("start_offset must identify a sample in the epoch")
        self._epoch = epoch
        self._start_offset = start_offset

    def __iter__(self) -> Iterator[TrainingSampleKey]:
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._epoch)
        order = torch.randperm(self._pair_count, generator=generator).tolist()
        flips = (
            torch.randint(0, 2, (self._pair_count,), generator=generator).tolist()
            if self._horizontal_flip
            else None
        )
        mismatched_by_index = (
            _group_derangement(
                order,
                self._groups,
                shift=self._derangement_shift,
            )
            if self._mismatched_source
            else None
        )
        flip_by_index = (
            [False] * self._pair_count if self._mismatched_source else None
        )
        if self._mismatched_source and flips is not None:
            assert flip_by_index is not None
            for position, index in enumerate(order):
                flip_by_index[index] = bool(flips[position])
        for position in range(self._start_offset, self._pair_count):
            flip = bool(flips[position]) if flips is not None else False
            if not self._mismatched_source:
                yield order[position], flip
                continue
            assert mismatched_by_index is not None
            assert flip_by_index is not None
            mismatched_index = mismatched_by_index[order[position]]
            yield (
                order[position],
                flip,
                mismatched_index,
                flip_by_index[mismatched_index],
            )

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
    mismatched_source: bool,
    target_palette: bool,
    pin_memory: bool,
) -> tuple[DataLoader[dict[str, Tensor | str]], EpochPairSampler]:
    target_palettes = (
        tuple(
            torch.unique(
                load_rgb_tensor(
                    pair.target_path,
                    image_size=image_size,
                )
                .permute(1, 2, 0)
                .reshape(-1, 3),
                dim=0,
            )
            for pair in pairs
        )
        if target_palette
        else None
    )
    dataset = PairedImageDataset(
        pairs,
        image_size=image_size,
        target_palettes=target_palettes,
    )
    sampler = EpochPairSampler(
        tuple(pair.group for pair in pairs),
        seed=seed,
        horizontal_flip=horizontal_flip,
        mismatched_source=mismatched_source,
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


def _group_derangement(
    order: list[int],
    groups: Sequence[str],
    *,
    shift: int,
) -> list[int]:
    group_rank: dict[str, int] = {}
    for index in order:
        group_rank.setdefault(groups[index], len(group_rank))
    grouped = sorted(order, key=lambda index: group_rank[groups[index]])
    rotated = grouped[shift:] + grouped[:shift]
    mismatched_by_index = [0] * len(order)
    for index, mismatched_index in zip(grouped, rotated, strict=True):
        mismatched_by_index[index] = mismatched_index
    return mismatched_by_index


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


def validate_white_canvas_region_balance(
    pairs: Sequence[PairedImage],
    *,
    image_size: int,
    translation_margin: int = 0,
) -> None:
    for pair in pairs:
        target = load_rgb_tensor(pair.target_path, image_size=image_size)
        background = target.eq(1.0).all(dim=0)
        background_count = int(background.sum().item())
        if background_count == 0 or background_count == image_size * image_size:
            raise Pix2PixError(
                "white_canvas_equal_regions requires both exact-white background "
                f"and non-white foreground in target: {pair.id}"
            )
        if translation_margin and not (
            ~background[
                translation_margin:-translation_margin,
                translation_margin:-translation_margin,
            ]
        ).any():
            raise Pix2PixError(
                "translation requires target foreground inside the translation-safe "
                f"canvas region: {pair.id}"
            )
