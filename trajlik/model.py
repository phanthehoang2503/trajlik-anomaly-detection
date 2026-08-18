import torch
from torch import Tensor, nn

from dcte import DCTE, MSMLoss
from ectf import EndpointConditionedTrajectoryFlow
from trajlik.normal_tail import EmpiricalTailCalibrator
from trajlik.trajectory_batch import build_trajectory_batch


class TrajLikHead(nn.Module):
    """Joint DCTE/ECTF head for cached-normal training or online inference."""

    def __init__(
        self,
        dcte: DCTE,
        ectf: EndpointConditionedTrajectoryFlow,
        msm_loss: MSMLoss,
        lambda_msm: float = 1.0,
        flow_dequantization_std: float = 0.1,
    ):
        super().__init__()
        if dcte.trajectory_dim != ectf.trajectory_dim:
            raise ValueError("DCTE and ECTF trajectory dimensions must match")
        if flow_dequantization_std < 0.0:
            raise ValueError("flow_dequantization_std must be non-negative")
        self.dcte = dcte
        self.ectf = ectf
        self.msm_loss = msm_loss
        self.lambda_msm = lambda_msm
        self.flow_dequantization_std = float(flow_dequantization_std)

    def forward(
        self,
        module0_output,
        mask: bool | None = None,
        *,
        dequantize_flow: bool = False,
    ) -> dict[str, Tensor]:
        if mask is None:
            mask = self.training
        trajectory_batch = build_trajectory_batch(module0_output)
        dcte_output = self.dcte(trajectory_batch, mask=mask)
        flow_trajectory_codes = dcte_output["trajectory_codes"]
        if dequantize_flow and self.flow_dequantization_std > 0.0:
            flow_trajectory_codes = flow_trajectory_codes + (
                torch.randn_like(flow_trajectory_codes)
                * self.flow_dequantization_std
            )
        flow_output = self.ectf(
            flow_trajectory_codes,
            trajectory_batch["a_end_coarse"],
            trajectory_batch["z0"],
        )
        return {
            **dcte_output,
            **flow_output,
            "a_end_coarse": trajectory_batch["a_end_coarse"],
            "flow_trajectory_codes": flow_trajectory_codes,
        }

    def training_loss(self, module0_output) -> dict[str, Tensor]:
        # Per-sample LayerNorm places DCTE codes on a nearly singular shell.
        # Resampled normal-only noise gives ECTF a full-dimensional density to
        # fit, preventing an unbounded Jacobian on the LayerNorm constraints.
        output = self(
            module0_output,
            mask=True,
            dequantize_flow=True,
        )
        nll_loss = output["path_nll"].mean()
        msm_loss = self.msm_loss(
            output,
            self.dcte.tokenizer.step_embedding,
        )
        total_loss = nll_loss + self.lambda_msm * msm_loss
        return {
            **output,
            "loss": total_loss,
            "nll_loss": nll_loss,
            "msm_loss": msm_loss,
        }


class TrajLikAD(nn.Module):
    """Inference composition that calls frozen InvAD exactly once per image."""

    def __init__(
        self,
        module0: nn.Module,
        head: TrajLikHead,
        calibrator: EmpiricalTailCalibrator,
    ):
        super().__init__()
        if not calibrator.fitted:
            raise ValueError("calibrator must be fit on held-out normal scores")
        self.module0 = module0.requires_grad_(False).eval()
        self.head = head
        self.calibrator = calibrator

    def train(self, mode: bool = True):
        super().train(mode)
        self.module0.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        *,
        lambda_path: float = 1.0,
    ) -> dict[str, Tensor]:
        module0_output = self.module0(images, labels)
        head_output = self.head(module0_output, mask=False)
        score_output = self.calibrator(
            head_output["a_end_coarse"],
            head_output["path_nll"],
            output_size=tuple(images.shape[-2:]),
            lambda_path=lambda_path,
        )
        return {**module0_output, **head_output, **score_output}
