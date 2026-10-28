import torch

from si_fdjscc.train.common import finish_step


def _disabled_scaler():
    return torch.cuda.amp.GradScaler(enabled=False)


def test_finish_step_updates_on_finite_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    result = finish_step(
        parameter.square(), optimizer, _disabled_scaler(), [parameter], grad_clip=1.0
    )

    assert not result.optimizer_step_skipped
    assert torch.isfinite(result.grad_norm)
    assert parameter.item() < 2.0


def test_finish_step_skips_non_finite_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    result = finish_step(
        parameter.sqrt(), optimizer, _disabled_scaler(), [parameter], grad_clip=1.0
    )

    assert result.optimizer_step_skipped
    assert torch.isinf(result.grad_norm)
    assert parameter.item() == 0.0
