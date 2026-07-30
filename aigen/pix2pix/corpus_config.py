from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from aigen.image_edit_defaults import FLUX2_KLEIN_SAMPLERS
from aigen.manifest_io import read_json
from aigen.pix2pix.errors import Pix2PixError


IRO_CORPUS_CONFIG_FORMAT = "aigen.pix2pix.iro-corpus.v1"
IRO_CORPUS_CONFIG_V2_FORMAT = "aigen.pix2pix.iro-corpus.v2"
CORPUS_SPLITS = ("train", "validation", "test")
SAFE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
SplitName = Literal["train", "validation", "test"]


class _CorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IroJob(_CorpusModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    lineage: str = Field(pattern=SAFE_NAME_PATTERN)
    species: str = Field(pattern=SAFE_NAME_PATTERN)
    genders: tuple[Literal[0, 1], ...] = Field(min_length=1)
    head_max: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_genders(self) -> IroJob:
        if len(set(self.genders)) != len(self.genders):
            raise ValueError(f"job {self.id} contains duplicate genders")
        return self


class IroBodyV2(_CorpusModel):
    gender: Literal[0, 1]
    rig_family: str = Field(pattern=SAFE_NAME_PATTERN)


class IroJobV2(_CorpusModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    lineage: str = Field(pattern=SAFE_NAME_PATTERN)
    species: str = Field(pattern=SAFE_NAME_PATTERN)
    split_group: str = Field(pattern=SAFE_NAME_PATTERN)
    bodies: tuple[IroBodyV2, ...] = Field(min_length=1)
    head_max: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bodies(self) -> IroJobV2:
        genders = [body.gender for body in self.bodies]
        if len(set(genders)) != len(genders):
            raise ValueError(f"job {self.id} contains duplicate body genders")
        return self


class IroAction(_CorpusModel):
    name: str = Field(pattern=SAFE_NAME_PATTERN)
    base: int = Field(ge=0)


class IroCharacterDefaults(_CorpusModel):
    head_direction: int = Field(ge=0)
    headgear: tuple[int, int, int]
    garment: int = Field(ge=0)
    body_palette: int
    madogear_type: int = Field(ge=0)
    outfit: int = Field(ge=0)


class LineagePairQuota(_CorpusModel):
    lineage: str = Field(pattern=SAFE_NAME_PATTERN)
    split: SplitName
    female: int = Field(ge=0)
    male: int = Field(ge=0)

    @property
    def count(self) -> int:
        return self.female + self.male


class SplitAxisQuotas(_CorpusModel):
    directions: dict[int, int] = Field(min_length=1)
    head_palettes: dict[int, int] = Field(min_length=1)
    actions: dict[int, int] = Field(min_length=1)
    heads_by_species: dict[str, tuple[int, ...]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_axes(self) -> SplitAxisQuotas:
        if any(direction < 0 or direction > 7 for direction in self.directions):
            raise ValueError("direction quota keys must be in [0, 7]")
        for name, quotas in (
            ("direction", self.directions),
            ("head palette", self.head_palettes),
            ("action", self.actions),
        ):
            if any(count < 0 for count in quotas.values()):
                raise ValueError(f"{name} quota counts must be non-negative")
        for species, heads in self.heads_by_species.items():
            if not re.fullmatch(SAFE_NAME_PATTERN, species):
                raise ValueError(f"invalid species name in head quotas: {species!r}")
            if not heads or any(head < 1 for head in heads):
                raise ValueError(f"head quotas for {species} must be positive")
            if len(set(heads)) != len(heads):
                raise ValueError(f"head quotas for {species} contain duplicates")
        return self


class SplitAxisQuotasV2(_CorpusModel):
    directions: dict[int, int] = Field(min_length=1)
    head_palettes: dict[int, int] = Field(min_length=1)
    actions: dict[int, int] = Field(min_length=1)
    heads_by_species: dict[str, tuple[int, ...]]

    @model_validator(mode="after")
    def validate_axes(self) -> SplitAxisQuotasV2:
        if any(direction < 0 or direction > 7 for direction in self.directions):
            raise ValueError("direction quota keys must be in [0, 7]")
        for name, quotas in (
            ("direction", self.directions),
            ("head palette", self.head_palettes),
            ("action", self.actions),
        ):
            if any(count < 0 for count in quotas.values()):
                raise ValueError(f"{name} quota counts must be non-negative")
        for species, heads in self.heads_by_species.items():
            if not re.fullmatch(SAFE_NAME_PATTERN, species):
                raise ValueError(f"invalid species name in head quotas: {species!r}")
            if not heads or any(head < 1 for head in heads):
                raise ValueError(f"head quotas for {species} must be positive")
            if len(set(heads)) != len(heads):
                raise ValueError(f"head quotas for {species} contain duplicates")
        return self


class FluxSourceConfig(_CorpusModel):
    prompt: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seed_base: int = Field(ge=0)
    shard_size: int = Field(gt=0, le=16)
    steps: Literal[4]
    sampler: str
    scheduler: Literal["flowmatch-dynamic-shift"]
    strength: float | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        if prompt != prompt.strip():
            raise ValueError("FLUX source prompt must not contain surrounding whitespace")
        return prompt

    @field_validator("sampler")
    @classmethod
    def validate_sampler(cls, sampler: str) -> str:
        if sampler not in FLUX2_KLEIN_SAMPLERS:
            raise ValueError(f"unsupported FLUX.2 Klein sampler: {sampler}")
        return sampler


class SourceRasterConfig(_CorpusModel):
    canvas_size: Literal[128]
    inner_width: int = Field(gt=0)
    inner_height: int = Field(gt=0)
    offset_x: int = Field(ge=0)
    offset_y: int = Field(ge=0)
    resample: Literal["lanczos"]
    background_rgb: tuple[int, int, int]

    @model_validator(mode="after")
    def validate_geometry(self) -> SourceRasterConfig:
        if self.offset_x + self.inner_width > self.canvas_size:
            raise ValueError("source raster exceeds the horizontal canvas boundary")
        if self.offset_y + self.inner_height > self.canvas_size:
            raise ValueError("source raster exceeds the vertical canvas boundary")
        if any(channel < 0 or channel > 255 for channel in self.background_rgb):
            raise ValueError("source raster background channels must be in [0, 255]")
        return self


class IroCorpusConfig(_CorpusModel):
    format: Literal["aigen.pix2pix.iro-corpus.v1"]
    name: str = Field(pattern=SAFE_NAME_PATTERN)
    endpoint: str = Field(min_length=1)
    canvas: str = Field(pattern=r"^128x128\+\d+\+\d+$")
    image_size: Literal[128]
    identity_seed: int = Field(ge=0)
    head_palettes: tuple[int, ...] = Field(min_length=1)
    lineage_splits: dict[str, SplitName] = Field(min_length=1)
    lineage_pair_quotas: tuple[LineagePairQuota, ...] = Field(min_length=1)
    split_axis_quotas: dict[SplitName, SplitAxisQuotas]
    actions: tuple[IroAction, ...] = Field(min_length=1)
    defaults: IroCharacterDefaults
    jobs: tuple[IroJob, ...] = Field(min_length=1)
    render_workers: int = Field(gt=0, le=8)
    request_timeout_seconds: float = Field(gt=0)
    flux: FluxSourceConfig
    source_raster: SourceRasterConfig

    @model_validator(mode="after")
    def validate_catalog(self) -> IroCorpusConfig:
        endpoint = urlparse(self.endpoint)
        if endpoint.scheme != "https" or not endpoint.netloc:
            raise ValueError("iRO renderer endpoint must be an absolute HTTPS URL")
        if len(set(self.head_palettes)) != len(self.head_palettes):
            raise ValueError("head_palettes contains duplicates")
        if any(palette < 1 for palette in self.head_palettes):
            raise ValueError("head palettes must be positive")

        job_ids = [job.id for job in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("iRO job catalog contains duplicate ids")
        action_names = [action.name for action in self.actions]
        if len(set(action_names)) != len(action_names):
            raise ValueError("iRO action catalog contains duplicate names")
        action_bases = [action.base for action in self.actions]
        if len(set(action_bases)) != len(action_bases):
            raise ValueError("iRO action catalog contains duplicate base values")

        catalog_lineages = {job.lineage for job in self.jobs}
        configured_lineages = set(self.lineage_splits)
        if catalog_lineages != configured_lineages:
            missing = sorted(catalog_lineages - configured_lineages)
            unexpected = sorted(configured_lineages - catalog_lineages)
            details = []
            if missing:
                details.append(f"missing lineages: {', '.join(missing)}")
            if unexpected:
                details.append(f"unknown lineages: {', '.join(unexpected)}")
            raise ValueError("; ".join(details))

        quota_lineages = [quota.lineage for quota in self.lineage_pair_quotas]
        if len(set(quota_lineages)) != len(quota_lineages):
            raise ValueError("lineage_pair_quotas contains duplicate lineages")
        if set(quota_lineages) != catalog_lineages:
            raise ValueError("lineage_pair_quotas must cover every catalog lineage exactly once")
        if set(self.split_axis_quotas) != set(CORPUS_SPLITS):
            raise ValueError("split_axis_quotas must define train, validation, and test")

        for split in CORPUS_SPLITS:
            split_quotas = [
                quota for quota in self.lineage_pair_quotas if quota.split == split
            ]
            split_pair_count = sum(quota.count for quota in split_quotas)
            if split_pair_count == 0:
                raise ValueError(f"lineage quotas contain no {split} pairs")
            axes = self.split_axis_quotas[split]
            for axis_name, quotas in (
                ("directions", axes.directions),
                ("head_palettes", axes.head_palettes),
                ("actions", axes.actions),
            ):
                if sum(quotas.values()) != split_pair_count:
                    raise ValueError(
                        f"{split} {axis_name} sum to {sum(quotas.values())}, "
                        f"expected {split_pair_count}"
                    )
            if set(axes.directions) != set(range(8)):
                raise ValueError(
                    f"{split} direction quotas must cover all eight directions"
                )
            if set(axes.head_palettes) != set(self.head_palettes):
                raise ValueError(
                    f"{split} head-palette quotas must match head_palettes"
                )
            if set(axes.actions) != set(action_bases):
                raise ValueError(f"{split} action quotas must cover every action base")

            jobs = [job for job in self.jobs if self.lineage_splits[job.lineage] == split]
            if not jobs:
                raise ValueError(f"iRO catalog has no jobs assigned to {split}")
            for quota in split_quotas:
                if self.lineage_splits[quota.lineage] != split:
                    raise ValueError(
                        f"lineage {quota.lineage} quota conflicts with lineage_splits"
                    )
                lineage_jobs = [job for job in jobs if job.lineage == quota.lineage]
                for gender, count in ((0, quota.female), (1, quota.male)):
                    if count and not any(gender in job.genders for job in lineage_jobs):
                        raise ValueError(
                            f"lineage {quota.lineage} has no jobs for gender {gender}"
                        )
            for job in jobs:
                heads = axes.heads_by_species.get(job.species)
                if heads is None:
                    raise ValueError(
                        f"{split} head quotas do not define species {job.species}"
                    )
                if max(heads) > job.head_max:
                    raise ValueError(
                        f"{split} head quotas exceed job {job.id} maximum {job.head_max}"
                    )
        return self


class IroCorpusConfigV2(_CorpusModel):
    format: Literal["aigen.pix2pix.iro-corpus.v2"]
    name: str = Field(pattern=SAFE_NAME_PATTERN)
    endpoint: str = Field(min_length=1)
    canvas: str = Field(pattern=r"^128x128\+\d+\+\d+$")
    image_size: Literal[128]
    identity_seed: int = Field(ge=0)
    head_palettes: tuple[int, ...] = Field(min_length=1)
    split_group_splits: dict[str, SplitName] = Field(min_length=1)
    lineage_pair_quotas: tuple[LineagePairQuota, ...] = Field(min_length=1)
    split_axis_quotas: dict[SplitName, SplitAxisQuotasV2]
    actions: tuple[IroAction, ...] = Field(min_length=1)
    defaults: IroCharacterDefaults
    jobs: tuple[IroJobV2, ...] = Field(min_length=1)
    render_workers: int = Field(gt=0, le=8)
    request_timeout_seconds: float = Field(gt=0)
    flux: FluxSourceConfig
    source_raster: SourceRasterConfig

    @model_validator(mode="after")
    def validate_catalog(self) -> IroCorpusConfigV2:
        endpoint = urlparse(self.endpoint)
        if endpoint.scheme != "https" or not endpoint.netloc:
            raise ValueError("iRO renderer endpoint must be an absolute HTTPS URL")
        if len(set(self.head_palettes)) != len(self.head_palettes):
            raise ValueError("head_palettes contains duplicates")
        if any(palette < 1 for palette in self.head_palettes):
            raise ValueError("head palettes must be positive")

        job_ids = [job.id for job in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("iRO job catalog contains duplicate ids")
        action_names = [action.name for action in self.actions]
        if len(set(action_names)) != len(action_names):
            raise ValueError("iRO action catalog contains duplicate names")
        action_bases = [action.base for action in self.actions]
        if len(set(action_bases)) != len(action_bases):
            raise ValueError("iRO action catalog contains duplicate base values")
        if any(action_base % 8 != 0 for action_base in action_bases):
            raise ValueError("v2 action bases must start on disjoint eight-frame blocks")

        for split_group in self.split_group_splits:
            if not re.fullmatch(SAFE_NAME_PATTERN, split_group):
                raise ValueError(f"invalid split-group name: {split_group!r}")
        configured_split_groups = set(self.split_group_splits)
        catalog_split_groups = {job.split_group for job in self.jobs}
        job_split_groups = [job.split_group for job in self.jobs]
        if len(set(job_split_groups)) != len(job_split_groups):
            raise ValueError("each v2 split group must own exactly one renderer job")
        if catalog_split_groups != configured_split_groups:
            missing = sorted(catalog_split_groups - configured_split_groups)
            unused = sorted(configured_split_groups - catalog_split_groups)
            details = []
            if missing:
                details.append(f"undefined split groups: {', '.join(missing)}")
            if unused:
                details.append(f"unused split groups: {', '.join(unused)}")
            raise ValueError("; ".join(details))

        catalog_lineages = {job.lineage for job in self.jobs}
        quota_keys = [
            (quota.lineage, quota.split) for quota in self.lineage_pair_quotas
        ]
        if len(set(quota_keys)) != len(quota_keys):
            raise ValueError(
                "lineage_pair_quotas contains duplicate lineage/split pairs"
            )
        quota_lineages = {lineage for lineage, _split in quota_keys}
        if quota_lineages != catalog_lineages:
            raise ValueError(
                "lineage_pair_quotas must cover every catalog lineage"
            )
        if set(self.split_axis_quotas) != set(CORPUS_SPLITS):
            raise ValueError("split_axis_quotas must define train, validation, and test")

        quota_key_set = set(quota_keys)
        job_quota_keys = {
            (job.lineage, self.split_group_splits[job.split_group])
            for job in self.jobs
        }
        if quota_key_set != job_quota_keys:
            missing = sorted(job_quota_keys - quota_key_set)
            unused = sorted(quota_key_set - job_quota_keys)
            details = []
            if missing:
                details.append(
                    "missing lineage/split quotas: "
                    + ", ".join(f"{lineage}/{split}" for lineage, split in missing)
                )
            if unused:
                details.append(
                    "unused lineage/split quotas: "
                    + ", ".join(f"{lineage}/{split}" for lineage, split in unused)
                )
            raise ValueError("; ".join(details))

        for split in CORPUS_SPLITS:
            split_quotas = [
                quota for quota in self.lineage_pair_quotas if quota.split == split
            ]
            split_pair_count = sum(quota.count for quota in split_quotas)
            if split in ("train", "validation") and split_pair_count == 0:
                raise ValueError(f"lineage quotas contain no {split} pairs")
            axes = self.split_axis_quotas[split]
            for axis_name, quotas in (
                ("directions", axes.directions),
                ("head_palettes", axes.head_palettes),
                ("actions", axes.actions),
            ):
                if sum(quotas.values()) != split_pair_count:
                    raise ValueError(
                        f"{split} {axis_name} sum to {sum(quotas.values())}, "
                        f"expected {split_pair_count}"
                    )
            if set(axes.directions) != set(range(8)):
                raise ValueError(f"{split} direction quotas must cover all eight directions")
            if set(axes.head_palettes) != set(self.head_palettes):
                raise ValueError(
                    f"{split} head-palette quotas must match head_palettes"
                )
            if set(axes.actions) != set(action_bases):
                raise ValueError(f"{split} action quotas must cover every action base")

            jobs = [
                job
                for job in self.jobs
                if self.split_group_splits[job.split_group] == split
            ]
            if split_pair_count == 0:
                if jobs:
                    raise ValueError(
                        f"iRO catalog assigns jobs to empty {split} split"
                    )
                if axes.heads_by_species:
                    raise ValueError(
                        f"empty {split} split must not define head species"
                    )
                continue
            if not jobs:
                raise ValueError(f"iRO catalog has no jobs assigned to {split}")
            for quota in split_quotas:
                lineage_jobs = [job for job in jobs if job.lineage == quota.lineage]
                for gender, count in ((0, quota.female), (1, quota.male)):
                    eligible_body_count = sum(
                        body.gender == gender
                        for job in lineage_jobs
                        for body in job.bodies
                    )
                    if count > 0 and eligible_body_count == 0:
                        raise ValueError(
                            f"lineage {quota.lineage}/{split} has no bodies "
                            f"for gender {gender}"
                        )
                    if count < eligible_body_count:
                        raise ValueError(
                            f"lineage {quota.lineage}/{split} gender {gender} "
                            f"quota {count} cannot cover {eligible_body_count} bodies"
                        )
                    if eligible_body_count:
                        maximum_body_count = (
                            count + eligible_body_count - 1
                        ) // eligible_body_count
                        positive_action_count = sum(
                            action_count > 0
                            for action_count in axes.actions.values()
                        )
                        positive_direction_count = sum(
                            direction_count > 0
                            for direction_count in axes.directions.values()
                        )
                        if maximum_body_count > min(
                            positive_action_count,
                            positive_direction_count,
                        ):
                            raise ValueError(
                                f"lineage {quota.lineage}/{split} gender {gender} "
                                f"assigns up to {maximum_body_count} samples per body, "
                                "exceeding unique action/direction capacity"
                            )
            for job in jobs:
                heads = axes.heads_by_species.get(job.species)
                if heads is None:
                    raise ValueError(
                        f"{split} head quotas do not define species {job.species}"
                    )
                if max(heads) > job.head_max:
                    raise ValueError(
                        f"{split} head quotas exceed job {job.id} maximum {job.head_max}"
                    )
        return self


IroCorpusConfigVersion = IroCorpusConfig | IroCorpusConfigV2


def load_iro_corpus_config(path: Path) -> IroCorpusConfigVersion:
    config_path = Path(path).expanduser().resolve()
    payload = read_json(config_path, label="iRO pix2pix corpus config")
    config_model = (
        IroCorpusConfigV2
        if isinstance(payload, dict)
        and payload.get("format") == IRO_CORPUS_CONFIG_V2_FORMAT
        else IroCorpusConfig
    )
    try:
        return config_model.model_validate(payload)
    except ValidationError as error:
        raise Pix2PixError(f"invalid iRO pix2pix corpus config: {error}") from error


def corpus_config_fingerprint(config: IroCorpusConfigVersion) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise Pix2PixError(f"cannot derive a safe corpus slug from {value!r}")
    return slug
