from __future__ import annotations


FLUX2_KLEIN_STEPS = 4
FLUX2_KLEIN_DEFAULT_SAMPLER = "flowmatch-euler"
FLUX2_KLEIN_SAMPLERS = (
    FLUX2_KLEIN_DEFAULT_SAMPLER,
    "euler-ancestral",
)
FLUX2_KLEIN_SCHEDULER = "flowmatch-dynamic-shift"

FLUX2_DEV_STEPS = 30
FLUX2_DEV_GUIDANCE = 4.0
FLUX2_DEV_SAMPLER = "flowmatch-euler"
FLUX2_DEV_SCHEDULER = "flowmatch-dynamic-shift"

QWEN_2511_LIGHTNING_DEFAULT_STEPS = 8
QWEN_2511_LIGHTNING_DEFAULT_GUIDANCE = 1.0
QWEN_2511_BASE_DEFAULT_STEPS = 40
QWEN_2511_BASE_DEFAULT_GUIDANCE = 4.0
QWEN_2511_SAMPLER = "flowmatch-euler"
QWEN_2511_SAMPLERS = (
    QWEN_2511_SAMPLER,
    "euler-ancestral",
)
QWEN_2511_DEFAULT_SCHEDULER = "flowmatch-dynamic-shift"
QWEN_2511_SCHEDULERS = (
    QWEN_2511_DEFAULT_SCHEDULER,
    "simple",
)

HIDREAM_DEFAULT_STEPS = 40
HIDREAM_DEFAULT_GUIDANCE = 5.0
HIDREAM_DEFAULT_SAMPLER = "dpmpp_2m_sde_gpu"
HIDREAM_SAMPLERS = (
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    HIDREAM_DEFAULT_SAMPLER,
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)
HIDREAM_DEFAULT_SCHEDULER = "normal"
HIDREAM_SCHEDULERS = (
    "normal",
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
)

BOOGU_DEFAULT_STEPS = 4
BOOGU_DEFAULT_GUIDANCE = 1.0
BOOGU_SAMPLER = "dmd"
BOOGU_SCHEDULER = "dmd-linear"
