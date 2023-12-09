import torch

from prithvi_pytorch import PrithviEncoder, PrithviUnet, PrithviViT

CFG_PATH = "tests/Prithvi_100M_config.yaml"


@torch.no_grad()
def test_encoder():
    model = PrithviEncoder(
        cfg_path=CFG_PATH,
        ckpt_path=None,
        num_frames=1,
        in_chans=6,
        img_size=224,
    )

    x = torch.randn(2, 6, 224, 224)
    y = model(x)
    assert y.shape == (2, 197, 768)


@torch.no_grad()
def test_classifier():
    model = PrithviViT(
        num_classes=10,
        cfg_path=CFG_PATH,
        ckpt_path=None,
        in_chans=6,
        img_size=224,
        freeze_encoder=True,
    )

    x = torch.randn(2, 6, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)


@torch.no_grad()
def test_segmentation():
    model = PrithviUnet(
        num_classes=3,
        cfg_path=CFG_PATH,
        ckpt_path=None,
        in_chans=6,
        img_size=224,
        n=[2, 5, 8, 11],
        norm=True,
        decoder_channels=[256, 128, 64, 32],
        freeze_encoder=True,
    )

    x = torch.randn(2, 6, 224, 224)
    y = model(x)
    assert y.shape == (2, 3, 224, 224)
