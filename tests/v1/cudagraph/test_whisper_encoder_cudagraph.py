# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol-conformance tests for Whisper's SupportsEncoderCudaGraph support.

No GPU and no weights required: these cover the pure logic of the eight
protocol methods, which is where an audio model differs from the vision models
the framework was built for. The numerics, capture, replay, padding, ownership
and fallback behaviour are covered against the real model by
``scripts/test/verify_encoder_cudagraph.py`` in the project harness.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

TOKENS_PER_AUDIO = 1500  # max_source_positions for large-v3-turbo
FRAMES = 3000  # == max_source_positions * conv total_stride (2)
N_MEL = 128
D_MODEL = 1280


class _FakeWhisper(WhisperForConditionalGeneration):
    """A WhisperForConditionalGeneration with just enough state for the
    protocol methods, built without touching nn.Module.__init__ or weights."""

    def __init__(self):  # noqa: D107 - deliberately does not call super()
        self.config = SimpleNamespace(
            max_source_positions=TOKENS_PER_AUDIO,
            num_mel_bins=N_MEL,
            d_model=D_MODEL,
        )
        self.model = SimpleNamespace(encoder=SimpleNamespace(total_stride=2))
        self.dtype = torch.float16


def _vllm_config(max_num_batched_tokens: int, max_num_seqs: int):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_encoder_input_tokens=max_num_batched_tokens,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        # Deliberately present and deliberately small: 448 is Whisper's
        # *decoder* length and must never reach the encoder budget.
        model_config=SimpleNamespace(max_model_len=448),
    )


@pytest.fixture
def model():
    return _FakeWhisper()


def test_class_declares_the_protocol():
    assert supports_encoder_cudagraph(WhisperForConditionalGeneration)


def test_config_is_audio_not_image(model):
    cfg = model.get_encoder_cudagraph_config()
    assert cfg.modalities == ["audio"]
    assert cfg.buffer_keys == ["input_features"]
    assert cfg.out_hidden_size == D_MODEL
    # Single path named "default": the default postprocess_encoder_output()
    # hardcodes that key.
    assert list(cfg.paths) == ["default"]
    assert cfg.capture_axes == ()
    assert cfg.padding_logics == {}


@pytest.mark.parametrize(
    "max_num_batched_tokens,max_num_seqs,expected_items",
    [
        (3584, 8, 2),  # vLLM's default budget: floor(3584/1500) == 2
        (24000, 16, 16),  # the benchmark protocol's pinned value
        (24000, 4, 4),  # max_num_seqs, not the token budget, is binding
        (1000, 16, 1),  # budget below one item still yields a usable bucket
    ],
)
def test_budget_range(model, max_num_batched_tokens, max_num_seqs, expected_items):
    lo, hi = model.get_encoder_cudagraph_budget_range(
        _vllm_config(max_num_batched_tokens, max_num_seqs)
    )
    assert lo == TOKENS_PER_AUDIO
    assert hi == TOKENS_PER_AUDIO * expected_items
    assert lo <= hi


def test_budget_range_ignores_max_model_len(model):
    """The regression this test exists for.

    Every vision implementor computes
    ``max_budget = min(max_num_batched_tokens, max_model_len)``. For an
    encoder-decoder model ``max_model_len`` is the decoder length -- 448 for
    Whisper -- while one audio item already costs 1500 encoder tokens, so
    copying that idiom gives ``min_budget (1500) > max_budget (448)`` and
    EncoderCudaGraphManager raises at startup.
    """
    cfg = _vllm_config(24000, 16)
    assert cfg.model_config.max_model_len < TOKENS_PER_AUDIO
    lo, hi = model.get_encoder_cudagraph_budget_range(cfg)
    assert hi > cfg.model_config.max_model_len
    assert lo <= hi


def test_budget_ladder_lands_on_whole_items(model):
    """The power-of-two ladder the manager generates must be exact item
    counts, with no ragged trailing budget."""
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    lo, hi = model.get_encoder_cudagraph_budget_range(_vllm_config(24000, 16))
    budgets = EncoderCudaGraphManager._generate_budgets(lo, hi)
    assert budgets == [1500, 3000, 6000, 12000, 24000]
    assert [b // TOKENS_PER_AUDIO for b in budgets] == [1, 2, 4, 8, 16]
    assert all(b % TOKENS_PER_AUDIO == 0 for b in budgets)


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_item_specs_are_constant_per_item(model, batch):
    feats = torch.zeros(batch, N_MEL, FRAMES)
    specs = model.get_encoder_cudagraph_item_specs({"input_features": feats})
    assert len(specs) == batch
    assert all(s.input_size == FRAMES for s in specs)
    assert all(s.output_tokens == TOKENS_PER_AUDIO for s in specs)
    assert all(s.get_path_output_tokens("default") == TOKENS_PER_AUDIO for s in specs)


def test_item_specs_accept_a_list_of_tensors(model):
    feats = [torch.zeros(N_MEL, FRAMES) for _ in range(3)]
    assert len(model.get_encoder_cudagraph_item_specs({"input_features": feats})) == 3


@pytest.mark.parametrize("indices", [[0], [2, 0], [], [1, 1]])
def test_select_items_picks_the_right_rows(model, indices):
    feats = torch.arange(4 * N_MEL * FRAMES, dtype=torch.float32).reshape(
        4, N_MEL, FRAMES
    )
    out = model.select_encoder_cudagraph_items({"input_features": feats}, indices)[
        "input_features"
    ]
    assert out.shape == (len(indices), N_MEL, FRAMES)
    for pos, src in enumerate(indices):
        assert torch.equal(out[pos], feats[src])


@pytest.mark.parametrize(
    "token_budget,max_batch_size,expected_rows",
    [
        (1500, 16, 1),
        (3000, 16, 2),
        (24000, 16, 16),
        (24000, 4, 4),  # max_batch_size clamps the budget
        (750, 16, 1),  # never zero rows: a 0-row capture would crash
    ],
)
def test_capture_inputs_shape(model, token_budget, max_batch_size, expected_rows):
    inputs = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget,
        max_batch_size,
        max_frames_per_batch=max_batch_size,  # a video concept; must be ignored
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    buf = inputs.values["input_features"]
    assert buf.shape == (expected_rows, N_MEL, FRAMES)
    assert buf.dtype == torch.float16


def test_replay_buffers_are_cast_to_model_dtype(model):
    feats = torch.zeros(2, N_MEL, FRAMES, dtype=torch.float32)
    buffers = model.prepare_encoder_cudagraph_replay_buffers(
        {"input_features": feats}, max_batch_size=16, max_frames_per_batch=16
    )
    out = buffers.values["input_features"]
    assert out.dtype == torch.float16
    assert out.shape == (2, N_MEL, FRAMES)


def test_forward_flattens_to_two_dims(model):
    """scatter_output_slices() slices dim 0 of a flat [total_tokens, hidden]
    tensor. Returning the 3-D [B, 1500, D] encoder output would scatter wrong
    slices silently rather than raise."""
    batch = 3
    fake = torch.arange(batch * TOKENS_PER_AUDIO * 4, dtype=torch.float32).reshape(
        batch, TOKENS_PER_AUDIO, 4
    )
    model.model.encoder = lambda _: fake
    out = model.encoder_cudagraph_forward({"input_features": torch.zeros(batch)})
    assert out.ndim == 2
    assert out.shape == (batch * TOKENS_PER_AUDIO, 4)
    # Row order must be item-major so consecutive 1500-row slices are items.
    assert torch.equal(out[:TOKENS_PER_AUDIO], fake[0])
    assert torch.equal(out[TOKENS_PER_AUDIO : 2 * TOKENS_PER_AUDIO], fake[1])
