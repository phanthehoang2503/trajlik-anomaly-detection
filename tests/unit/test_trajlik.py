import unittest

import torch
from torch import nn

from dcte import DCTE, MSMLoss
from ectf import EndpointConditionedTrajectoryFlow
from trajlik.model import TrajLikAD, TrajLikHead
from trajlik.normal_tail import EmpiricalTailCalibrator


def build_head():
    dcte = DCTE(
        input_dim=8,
        projection_dim=4,
        token_dim=8,
        trajectory_dim=4,
        num_steps=3,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )
    ectf = EndpointConditionedTrajectoryFlow(
        trajectory_dim=4,
        z0_dim=8,
        condition_dim=8,
        global_dim=4,
        position_dim=4,
        conditioner_hidden_dim=16,
        coupling_hidden_dim=16,
        num_blocks=2,
        num_bins=4,
    )
    msm = MSMLoss(
        input_dim=8,
        projection_dim=4,
        token_dim=8,
        trajectory_dim=4,
    )
    return TrajLikHead(
        dcte,
        ectf,
        msm,
    )


def module0_output(batch=2):
    states = torch.randn(batch, 4, 8, 4, 4)
    return {
        "states": states,
        "epsilons": torch.randn(batch, 3, 8, 4, 4),
        "deltas": states[:, 1:] - states[:, :-1],
        "a_end_coarse": torch.linalg.vector_norm(states[:, -1], dim=1),
    }


class CountingModule0(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, images, labels=None):
        self.calls += 1
        return module0_output(images.shape[0])


class TrajLikTest(unittest.TestCase):
    def test_joint_head_loss_is_finite_and_backward_works(self):
        head = build_head().train()

        output = head.training_loss(module0_output())

        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        gradient = head.dcte.tokenizer.state_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_ectf_receives_dcte_trajectory_codes_directly_during_training(self):
        head = build_head().train()
        captured = {}

        def capture_ectf_input(_module, args):
            captured["trajectory_codes"] = args[0]

        hook = head.ectf.register_forward_pre_hook(capture_ectf_input)
        try:
            output = head.training_loss(module0_output())
        finally:
            hook.remove()

        self.assertIs(
            captured["trajectory_codes"],
            output["trajectory_codes"],
        )

    def test_inference_runs_module0_once_and_returns_official_scores(self):
        module0 = CountingModule0()
        head = build_head().eval()
        calibrator = EmpiricalTailCalibrator().fit(
            torch.randn(64),
            torch.randn(64),
        )
        model = TrajLikAD(module0, head, calibrator).eval()
        images = torch.randn(2, 3, 16, 16)

        output = model(images)

        self.assertEqual(module0.calls, 1)
        self.assertFalse(module0.anchor.requires_grad)
        self.assertEqual(tuple(output["coarse_map"].shape), (2, 4, 4))
        self.assertEqual(tuple(output["pixel_map"].shape), (2, 16, 16))
        self.assertEqual(tuple(output["image_score"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
