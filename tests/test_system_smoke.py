import torch

from si_fdjscc.comm import OFDMFrequencySelectiveChannel, SemanticFrequencyMapper
from si_fdjscc.losses import ReconstructionLoss
from si_fdjscc.models import ImportanceHead, SIFDJSCC, TinyConvBackbone
from si_fdjscc.train.sensitivity import channel_conditioned_ablation_target


def test_tiny_system_forward_and_reproducibility() -> None:
    torch.manual_seed(21)
    backbone = TinyConvBackbone(latent_channels=16, hidden_channels=24)
    channel = OFDMFrequencySelectiveChannel(n_taps=4, n_cp=16)
    model = SIFDJSCC(
        backbone,
        ImportanceHead(16),
        mapper=SemanticFrequencyMapper(channel.plan),
        channel=channel,
    ).eval()
    images = torch.rand(2, 3, 32, 32)
    config = {"csi_mode": "imperfect", "csi_nmse_db": -15, "channel_seed": 100}
    with torch.no_grad():
        first = model(images, torch.tensor([10.0, 10.0]), config, n_channel_draws=2)
        second = model(images, torch.tensor([10.0, 10.0]), config, n_channel_draws=2)
    assert first.shape == (2, 2, 3, 32, 32)
    assert torch.equal(first, second)


def test_detached_importance_input_isolates_encoder_gradient() -> None:
    torch.manual_seed(22)
    backbone = TinyConvBackbone(latent_channels=8, hidden_channels=12)
    channel = OFDMFrequencySelectiveChannel(n_taps=2, n_cp=16)
    model = SIFDJSCC(
        backbone,
        ImportanceHead(8),
        mapper=SemanticFrequencyMapper(channel.plan),
        channel=channel,
    ).train()
    images = torch.rand(1, 3, 16, 16)
    _, auxiliary = model(
        images,
        torch.tensor([10.0]),
        {"csi_mode": "perfect", "channel_seed": 101},
        return_aux=True,
        detach_importance_input=True,
    )

    auxiliary["importance_logits"].square().sum().backward()

    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(
        parameter.grad is not None for parameter in model.importance_head.parameters()
    )


def test_importance_head_uses_channel_context_and_returns_nonnegative_sds() -> None:
    torch.manual_seed(23)
    head = ImportanceHead(8, channel_feature_dim=11).eval()
    with torch.no_grad():
        head.channel_conditioner[-1].weight.normal_(mean=0.0, std=0.1)
    tokens = torch.randn(2, 5, 8)
    first_context = torch.randn(2, 11)
    second_context = first_context.clone()
    second_context[:, 0] += 3.0
    _, first_logits = head(tokens, first_context, return_logits=True)
    _, second_logits = head(tokens, second_context, return_logits=True)
    assert not torch.allclose(first_logits, second_logits)
    assert torch.all(head.positive_scores(first_logits) >= 0)


def test_fixed_transmission_reuses_noise_and_ablation_target_is_sparse() -> None:
    torch.manual_seed(24)
    backbone = TinyConvBackbone(latent_channels=8, hidden_channels=12)
    channel = OFDMFrequencySelectiveChannel(n_taps=2, n_cp=8)
    model = SIFDJSCC(
        backbone,
        ImportanceHead(8),
        mapper=SemanticFrequencyMapper(channel.plan),
        channel=channel,
    ).eval()
    images = torch.rand(1, 3, 16, 16)
    snr = torch.tensor([10.0])
    _, auxiliary = model(
        images,
        snr,
        {"csi_mode": "imperfect", "csi_nmse_db": -20.0},
        return_aux=True,
        n_channel_draws=2,
    )
    repeated = model.reconstruct_with_fixed_transmission(
        auxiliary["z"], auxiliary["backbone_metadata"], snr, auxiliary
    )
    assert torch.equal(repeated, auxiliary["reconstructions"])
    assert all(draw["channel_seed"] is not None for draw in auxiliary["draws"])

    sensitivity, observed, indices = channel_conditioned_ablation_target(
        model,
        images,
        snr,
        auxiliary,
        ReconstructionLoss(),
        tokens_per_batch=3,
        attenuation=0.0,
    )
    assert sensitivity.shape == observed.shape == auxiliary["importance"].shape
    assert indices.shape == (1, 3)
    assert int(observed.sum()) == 3
    assert torch.all(sensitivity >= 0)
