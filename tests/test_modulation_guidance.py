from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from sam3ext.guidance import modulation


def _adapter_state(num_blocks=2, model_channels=2, pooled_dim=3):
    adaln_dim = model_channels * 3
    return {
        "scales": torch.arange(
            1,
            num_blocks * adaln_dim + 1,
            dtype=torch.float32,
        ).reshape(num_blocks, adaln_dim)
        / 10.0,
        "text_embedder_clip.linear_1.weight": torch.arange(
            adaln_dim * pooled_dim,
            dtype=torch.float32,
        ).reshape(adaln_dim, pooled_dim)
        / 20.0,
        "text_embedder_clip.linear_1.bias": torch.linspace(
            -0.2,
            0.2,
            adaln_dim,
        ),
        "text_embedder_clip.linear_2.weight": torch.eye(adaln_dim),
        "text_embedder_clip.linear_2.bias": torch.linspace(
            0.1,
            -0.1,
            adaln_dim,
        ),
    }


class DummyAnima:
    def __init__(
        self,
        num_blocks=2,
        model_channels=2,
        *,
        forge_layout=False,
    ):
        if forge_layout:
            self.blocks = [
                type("ForgeAnimaBlock", (), {"x_dim": model_channels})()
                for _ in range(num_blocks)
            ]
        else:
            self.blocks = [object() for _ in range(num_blocks)]
            self.model_channels = model_channels


class ModulationGuidanceTests(unittest.TestCase):
    def tearDown(self):
        modulation.clear_modulation_caches()

    def test_safe_adapter_load_keeps_only_expected_tensors(self):
        state = _adapter_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save({**state, "optimizer": {"not": "used"}}, path)

            loaded = modulation.load_adapter_cpu(path)

        self.assertEqual(
            set(loaded),
            set(modulation.EXPECTED_ADAPTER_KEYS),
        )
        for value in loaded.values():
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(value.is_contiguous())

    def test_existing_official_adapter_is_sha256_validated_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = modulation.default_adapter_path(directory)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"good")
            expected = hashlib.sha256(b"good").hexdigest()
            with (
                mock.patch.object(modulation, "ADAPTER_SIZE", 4),
                mock.patch.object(modulation, "ADAPTER_SHA256", expected),
            ):
                resolved = modulation.resolve_adapter_path(
                    directory,
                    "Auto-download official",
                )
                self.assertEqual(resolved, path.resolve())
                self.assertEqual(
                    len(modulation._VERIFIED_OFFICIAL_ADAPTERS),
                    1,
                )

                modulation.clear_modulation_caches()
                path.write_bytes(b"evil")
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    modulation.resolve_adapter_path(
                        directory,
                        "Auto-download official",
                    )

    def test_adapter_shape_validation_uses_forge_anima_structure(self):
        state = _adapter_state()
        meta = modulation.validate_adapter_for_model(
            state,
            DummyAnima(forge_layout=True),
        )

        self.assertEqual(
            meta,
            {"num_blocks": 2, "adaln_dim": 6, "pooled_dim": 3},
        )

        bad = dict(state)
        bad["scales"] = torch.zeros(1, 6)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            modulation.validate_adapter_for_model(
                bad,
                DummyAnima(forge_layout=True),
            )

    def test_projection_matches_reference_formula(self):
        state = _adapter_state()
        pooled = torch.tensor(
            [
                [0.2, -0.4, 0.6],
                [0.8, 0.1, -0.5],
                [-0.3, 0.7, 0.2],
            ]
        )
        weight = 3.0

        result, meta = modulation.project_block_modulations(
            pooled,
            state,
            DummyAnima(),
            weight,
        )

        projected = F.linear(
            pooled,
            state["text_embedder_clip.linear_1.weight"],
            state["text_embedder_clip.linear_1.bias"],
        )
        projected = F.silu(projected)
        projected = F.linear(
            projected,
            state["text_embedder_clip.linear_2.weight"],
            state["text_embedder_clip.linear_2.bias"],
        )
        expected = state["scales"] * (
            projected[0] + weight * (projected[1] - projected[2])
        ).unsqueeze(0)

        torch.testing.assert_close(result, expected)
        self.assertEqual(meta["num_blocks"], 2)

    def test_weight_zero_retains_base_modulation(self):
        state = _adapter_state()
        pooled = torch.randn(3, 3)

        result, _ = modulation.project_block_modulations(
            pooled,
            state,
            DummyAnima(),
            0.0,
        )
        base = F.linear(
            pooled[:1],
            state["text_embedder_clip.linear_1.weight"],
            state["text_embedder_clip.linear_1.bias"],
        )
        base = F.silu(base)
        base = F.linear(
            base,
            state["text_embedder_clip.linear_2.weight"],
            state["text_embedder_clip.linear_2.bias"],
        )[0]

        torch.testing.assert_close(result, state["scales"] * base)
        self.assertFalse(torch.equal(result, torch.zeros_like(result)))

    def test_hf_and_forge_clip_prefixes_are_normalized(self):
        tensor = torch.ones(2, 2)
        direct = {
            "text_model.embeddings.token_embedding.weight": tensor,
        }
        prefixed = {
            "text_encoders.clip_l.transformer."
            "text_model.embeddings.token_embedding.weight": tensor,
        }

        self.assertIs(
            modulation._normalize_clip_state_dict(direct),
            direct,
        )
        normalized = modulation._normalize_clip_state_dict(prefixed)
        self.assertIn(
            "text_model.embeddings.token_embedding.weight",
            normalized,
        )

    def test_clip_model_listing_prefers_recommended_anime_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            text_encoder = Path(directory) / "text_encoder"
            text_encoder.mkdir()
            for name in (
                "other_clip_l.safetensors",
                "Anzhc CLIP L Anime.safetensors",
                "clip_g.safetensors",
                "qwen.safetensors",
            ):
                (text_encoder / name).touch()
            (text_encoder / "ignore.ckpt").touch()

            token_key = (
                "text_model.embeddings.token_embedding.weight"
            )

            class FakeSlice:
                def __init__(self, shape):
                    self.shape = shape

                def get_shape(self):
                    return self.shape

            class FakeHandle:
                def __init__(self, path):
                    self.name = Path(path).name

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def keys(self):
                    return (
                        ["model.embed_tokens.weight"]
                        if self.name == "qwen.safetensors"
                        else [token_key]
                    )

                def get_slice(self, _key):
                    width = 1280 if self.name == "clip_g.safetensors" else 768
                    return FakeSlice((8, width))

            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = (
                lambda path, **_kwargs: FakeHandle(path)
            )
            with mock.patch.dict(
                sys.modules,
                {"safetensors": fake_safetensors},
            ):
                choices = modulation.list_clip_l_models(directory)

        self.assertEqual(choices[0], "Anzhc CLIP L Anime.safetensors")
        self.assertEqual(len(choices), 2)


if __name__ == "__main__":
    unittest.main()
