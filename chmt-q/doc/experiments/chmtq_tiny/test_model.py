import unittest

import torch

from model import ChmtQTiny, ordered_quantiles, parameter_count, safe_masked_softmax


class TinyModelTests(unittest.TestCase):
    def test_parameter_ledger(self) -> None:
        self.assertEqual(parameter_count(ChmtQTiny()), 282_838)

    def test_masked_large_logit_has_finite_zero_gradient(self) -> None:
        scores = torch.tensor([[0.0, 1000.0]], requires_grad=True)
        probabilities = safe_masked_softmax(scores, torch.tensor([[True, False]]))
        probabilities.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(probabilities.tolist(), [[1.0, 0.0]])
        self.assertEqual(scores.grad.tolist(), [[0.0, 0.0]])

    def test_full_contract_and_quantile_order(self) -> None:
        torch.manual_seed(7)
        model = ChmtQTiny().eval()
        batch = 2
        bars_mask = torch.ones(batch, 64, dtype=torch.bool)
        bars_mask[1, :5] = False
        l1_mask = torch.ones(batch, 240, dtype=torch.bool)
        level_mask = torch.ones(batch, 240, 2, 10, dtype=torch.bool)
        snapshot_mask = torch.ones(batch, 240, dtype=torch.bool)
        with torch.inference_mode():
            raw, auxiliary, diagnostics = model(
                torch.randn(batch, 64, 16),
                bars_mask,
                torch.randn(batch, 240, 16),
                l1_mask,
                torch.randn(batch, 240, 2, 10, 12),
                level_mask,
                snapshot_mask,
                torch.randn(batch, 16),
            )
        quantiles = ordered_quantiles(raw)
        self.assertEqual(raw.shape, (batch, 5, 17))
        self.assertEqual(auxiliary.shape, (batch, 5, 17))
        self.assertEqual(quantiles.shape, (batch, 5, 5, 3))
        self.assertEqual(diagnostics["memory_mask"].shape, (batch, 184))
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue((quantiles[..., 0] <= quantiles[..., 1]).all())
        self.assertTrue((quantiles[..., 1] <= quantiles[..., 2]).all())
        torch.testing.assert_close(
            diagnostics["gates"].sum(dim=-1), torch.ones(batch), rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
