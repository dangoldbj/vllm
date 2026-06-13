# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from functools import partial

import pytest

from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform

from ....conftest import IMAGE_ASSETS, VIDEO_ASSETS
from ...utils import dummy_hf_overrides
from .vlm_utils.builders import sample_frames_with_video_metadata


@dataclass
class VitCudagraphTestConfig:
    model: str
    modalities: list[str] = field(default_factory=lambda: ["image", "video"])
    image_prompt: str | None = None
    video_prompt: str | None = None
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_tokens: int = 64
    max_num_seqs: int = 2
    num_video_frames: int = 16
    needs_video_metadata: bool = False
    vllm_runner_kwargs: dict = field(default_factory=dict)
    compilation_config_overrides: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)


def params_with_marks(
    configs: dict[str, VitCudagraphTestConfig],
) -> list[pytest.param]:
    return [
        pytest.param(model_id, marks=cfg.marks) for model_id, cfg in configs.items()
    ]


def qwen_vl_chat_template(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def internvl_chat_template(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def step3_vl_chat_template(content: str) -> str:
    return (
        "<｜begin▁of▁sentence｜> You are a helpful assistant.<|BOT|>user\n "
        f"<im_patch>{content} <|EOT|><|BOT|>assistant\n"
    )


MODEL_CONFIGS: dict[str, VitCudagraphTestConfig] = {
    "llama4": VitCudagraphTestConfig(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        modalities=["image"],
        image_prompt=(
            "<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
            "<|image|>What is in this image?<|eot|>"
            "<|header_start|>assistant<|header_end|>\n\n"
        ),
        max_model_len=4096,
        max_tokens=32,
        max_num_seqs=2,
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="Llama4ForConditionalGeneration",
            ),
        },
        marks=[pytest.mark.core_model],
    ),
    "internvl": VitCudagraphTestConfig(
        model="OpenGVLab/InternVL3-1B",
        num_video_frames=8,
        image_prompt=internvl_chat_template("<image>\nWhat is in this image?"),
        video_prompt=internvl_chat_template(
            "<video>\nDescribe this video in one sentence."
        ),
        needs_video_metadata=False,
        vllm_runner_kwargs={"trust_remote_code": True},
        marks=[pytest.mark.core_model],
    ),
    "qwen2_5_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=False,
        marks=[pytest.mark.core_model],
    ),
    "qwen3_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "qwen3_5": VitCudagraphTestConfig(
        model="Qwen/Qwen3.5-0.8B",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "qwen2_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen2-VL-2B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=False,
        marks=[pytest.mark.core_model],
    ),
    "step3_vl": VitCudagraphTestConfig(
        model="stepfun-ai/Step3-VL-10B",
        modalities=["image"],
        image_prompt=step3_vl_chat_template("What is in this image?"),
        # Single bucket sized to cover the largest test image's output
        # tokens (1152 > 1141 for cherry_blossom). The default auto-
        # inferred range fans out into multiple power-of-2 buckets, each
        # holding a full ViT capture pool.
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1152],
        },
        # Shrink to 1 text + 1 vision layer with random weights so the
        # test runs on any CI GPU (incl. L4) and skips the 20 GiB weight
        # download. The test only validates that encoder CG capture/
        # replay functions correctly, not output quality.
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="StepVLForConditionalGeneration",
            ),
        },
    ),
    "glm4_1v": VitCudagraphTestConfig(
        model="zai-org/GLM-4.1V-9B-Thinking",
        image_prompt=(
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            "<|begin_of_image|><|image|><|end_of_image|>"
            "What is in this image?<|assistant|>assistant\n"
        ),
        video_prompt=(
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            "<|begin_of_video|><|video|><|end_of_video|>"
            "Describe this video in one sentence<|assistant|>assistant\n"
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="Glm4vForConditionalGeneration",
            ),
        },
    ),
}


def get_compilation_config(
    config: VitCudagraphTestConfig,
    *,
    cudagraph_mm_encoder: bool,
    compile_mm_encoder: bool = False,
):
    return {
        "cudagraph_mm_encoder": cudagraph_mm_encoder,
        "compile_mm_encoder": compile_mm_encoder,
        "encoder_cudagraph_max_vision_items_per_batch": 1,
        "encoder_cudagraph_max_frames_per_batch": 16,
        **config.compilation_config_overrides,
    }


def generate_greedy_under_encoder_config(
    vllm_runner,
    config: VitCudagraphTestConfig,
    prompts: list[str],
    mm_kwargs: dict,
    mm_limit: dict[str, int],
    *,
    cudagraph_mm_encoder: bool,
    compile_mm_encoder: bool = False,
):
    """Greedily generate with the given ViT encoder compilation mode."""
    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt=mm_limit,
        compilation_config=get_compilation_config(
            config,
            cudagraph_mm_encoder=cudagraph_mm_encoder,
            compile_mm_encoder=compile_mm_encoder,
        ),
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        return vllm_model.generate_greedy(prompts, config.max_tokens, **mm_kwargs)


def assert_encoder_cudagraph_parity(eager_outputs, cudagraph_outputs):
    """Encoder CUDA graph replay is numerically identical to eager execution,
    so greedy decoding must yield the same tokens. A mismatch means the
    captured graph corrupts the vision embeddings; an empty output means the
    encoder produced nothing.
    """
    assert len(eager_outputs) == len(cudagraph_outputs)
    for (eager_ids, eager_text), (cg_ids, cg_text) in zip(
        eager_outputs, cudagraph_outputs
    ):
        assert len(cg_ids) > 0 and len(cg_text) > 0
        assert eager_ids == cg_ids, (
            "Encoder CUDA graph output diverged from eager execution:\n"
            f"  eager:     {eager_text!r}\n"
            f"  cudagraph: {cg_text!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_vit_cudagraph_image(model_id, vllm_runner, image_assets):
    config = MODEL_CONFIGS[model_id]

    if "image" not in config.modalities:
        pytest.skip(f"{model_id} does not support the image modality.")

    image_prompts = IMAGE_ASSETS.prompts(
        {
            "stop_sign": config.image_prompt,  # type: ignore[typeddict-item]
            "cherry_blossom": config.image_prompt,  # type: ignore[typeddict-item]
        }
    )
    images = [[asset.pil_image] for asset in image_assets]
    mm_kwargs = {"images": images}
    mm_limit = {"image": 1}

    eager_outputs = generate_greedy_under_encoder_config(
        vllm_runner,
        config,
        image_prompts,
        mm_kwargs,
        mm_limit,
        cudagraph_mm_encoder=False,
    )
    cudagraph_outputs = generate_greedy_under_encoder_config(
        vllm_runner,
        config,
        image_prompts,
        mm_kwargs,
        mm_limit,
        cudagraph_mm_encoder=True,
    )

    assert len(cudagraph_outputs) == 2
    assert_encoder_cudagraph_parity(eager_outputs, cudagraph_outputs)


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_vit_cudagraph_video(model_id, vllm_runner, video_assets):
    config = MODEL_CONFIGS[model_id]

    if "video" not in config.modalities:
        pytest.skip(f"{model_id} does not support the video modality")

    video_prompts = VIDEO_ASSETS.prompts(
        {
            "baby_reading": config.video_prompt,  # type: ignore[typeddict-item]
        }
    )
    if config.needs_video_metadata:
        sampled_vids = [
            sample_frames_with_video_metadata(
                (asset.np_ndarrays, asset.metadata), config.num_video_frames
            )
            for asset in video_assets
        ]
    else:
        sampled_vids = [
            sample_frames_from_video(asset.np_ndarrays, config.num_video_frames)
            for asset in video_assets
        ]
    videos = [sampled_vids[0]]
    mm_kwargs = {"videos": videos}
    mm_limit = {"video": 1}

    eager_outputs = generate_greedy_under_encoder_config(
        vllm_runner,
        config,
        video_prompts,
        mm_kwargs,
        mm_limit,
        cudagraph_mm_encoder=False,
    )
    cudagraph_outputs = generate_greedy_under_encoder_config(
        vllm_runner,
        config,
        video_prompts,
        mm_kwargs,
        mm_limit,
        cudagraph_mm_encoder=True,
    )

    assert len(cudagraph_outputs) == 1
    assert_encoder_cudagraph_parity(eager_outputs, cudagraph_outputs)
