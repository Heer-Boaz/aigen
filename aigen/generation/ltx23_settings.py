from __future__ import annotations

from dataclasses import dataclass


LTX23_MODEL_TYPES = {"nvfp4": "ltx2_22B_nvfp4", "int8": "ltx2_22B"}
LTX23_DEFAULT_MODEL = "nvfp4"
LTX23_DEFAULT_FPS = 24
LTX23_DEFAULT_PHASES = 1
LTX23_DEFAULT_CONDITIONING_STRENGTH = 1.0
LTX23_DEFAULT_NEGATIVE_PROMPT = "Morphing, warping, flicker."
LTX23_MINIMUM_FRAMES = 17
LTX23_FRAME_STEP = 8
LTX23_PHASES = frozenset({1, 2})
LTX23_SOLVERS = frozenset({"distilled_8_steps", "euler", "res2s"})


class Ltx23KeyframesError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedLtx23Settings:
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    phases: int
    solver: str
    conditioning_strength: float
    model: str
    keyframe_fit: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def resolve_ltx23_settings(*, resolution: str, frames: int, fps: int, steps: int,
                          phases: int, solver: str, conditioning_strength: float,
                          model: str, keyframe_fit: str = "crop") -> ResolvedLtx23Settings:
    if frames < LTX23_MINIMUM_FRAMES or (frames - 1) % LTX23_FRAME_STEP:
        raise Ltx23KeyframesError(f"LTX-2.3 frames must be {LTX23_MINIMUM_FRAMES} or more in increments of {LTX23_FRAME_STEP}")
    if fps <= 0 or steps <= 0:
        raise Ltx23KeyframesError("frames per second and inference steps must be positive")
    if phases not in LTX23_PHASES:
        raise Ltx23KeyframesError(f"unsupported LTX-2.3 phase count: {phases}")
    if solver not in LTX23_SOLVERS:
        raise Ltx23KeyframesError(f"unsupported LTX-2.3 solver: {solver}")
    if model not in LTX23_MODEL_TYPES:
        raise Ltx23KeyframesError(f"unsupported LTX-2.3 model: {model}")
    if solver == "distilled_8_steps" and steps != 8:
        raise Ltx23KeyframesError("distilled_8_steps requires exactly 8 inference steps")
    if not 0 <= conditioning_strength <= 1:
        raise Ltx23KeyframesError("conditioning strength must be between 0 and 1")
    if keyframe_fit not in ("crop", "pad", "stretch"):
        raise Ltx23KeyframesError(f"unsupported keyframe fit: {keyframe_fit}")
    try:
        width_text, height_text = resolution.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise Ltx23KeyframesError("resolution must use WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise Ltx23KeyframesError("resolution dimensions must be positive")
    # The pinned native LTX model rounds up to 64 before latent construction.
    return ResolvedLtx23Settings((width + 63) // 64 * 64, (height + 63) // 64 * 64,
                                frames, fps, steps, phases, solver, conditioning_strength, model, keyframe_fit)


def validate_ltx23_positions(positions: tuple[int, ...], frames: int) -> None:
    if not positions:
        raise Ltx23KeyframesError("LTX-2.3 requires at least one keyframe")
    if len(set(positions)) != len(positions):
        raise Ltx23KeyframesError("duplicate LTX-2.3 keyframe position")
    invalid = next((position for position in positions if not 0 <= position < frames), None)
    if invalid is not None:
        raise Ltx23KeyframesError(f"keyframe position {invalid} is outside video frame range 0..{frames - 1}")
    if len(positions) == 1 and positions[0] != 0:
        raise Ltx23KeyframesError("a single LTX-2.3 keyframe must target frame 0")
