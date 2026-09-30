"""CNET: contact estimation network used to shape the Go1 locomotion reward.

The network is a 1-D CNN followed by two GRUs and a classifier over the 16 possible
contact states of the four feet. Its architecture is reconstructed from ``CNET.ckpt``
and validated when the weights are loaded (``strict=True``), so a wrong layer layout
fails immediately instead of silently producing noise.

Reference: Thema et al., "Proprioceptive ContactNet: Towards Bridging the Sim-to-Real
Gap in Quadrupedal Contact Estimation", CROS 2026 (DOI 10.1109/CROS69211.2026.11565696).
The 48-channel, IMU-free configuration of that paper is the one shipped here:
``z = [q, qd, p_f, v_f]``, a window of 150 samples, and a 16-class output decoded into a
4-bit contact vector ordered (LF, RF, LH, RH).

Interface:

* input  ``(N, 150, 48)`` - one window of 150 environment steps per environment,
  48 features each: ``z = [q, qd, p_f, v_f]``, 4 legs x 3 values per group;
* output ``(N, 4)`` - soft contact probability per foot, ordered (LF, RF, LH, RH),
  obtained by marginalizing the 16-class head over the states in which a foot touches.

The window is expressed in environment steps, which also fixes the rate the network sees
(the published model was trained on 1 kHz logs). The training-time input normalization is
not published, so none is applied here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

CHECKPOINT_PATH = Path(__file__).parent / "CNET.ckpt"

# ---------------------------------------------------------------------------- spec
# legs in the order used by the paper, and the Unitree Go1 prefix of each one
LEG_ORDER = ("LF", "RF", "LH", "RH")
LEG_TO_GO1 = {"LF": "FL", "RF": "FR", "LH": "RL", "RH": "RR"}
JOINT_ORDER = ("hip", "thigh", "calf")
# channel groups of z = [q, qd, p_f, v_f]; each contributes 12 channels (4 legs x 3 values)
CHANNEL_GROUPS = ("joint_pos", "joint_vel", "foot_pos", "foot_vel")
# number of environment steps in one window
WINDOW_SIZE = 150
# class index -> contact of (LEG_ORDER[0], ..., LEG_ORDER[3]), most significant bit first
CLASS_MSB_FIRST = True
# the published pipeline normalizes its inputs but does not give the statistics
INPUT_MEAN: float | torch.Tensor = 0.0
INPUT_STD: float | torch.Tensor = 1.0

#: Go1 foot bodies in the order ContactNet expects
GO1_FEET = tuple(f"{LEG_TO_GO1[leg]}_foot" for leg in LEG_ORDER)

NUM_FEET = len(LEG_ORDER)
NUM_CLASSES = 2**NUM_FEET
INPUT_CHANNELS = len(CHANNEL_GROUPS) * 3 * NUM_FEET  # 48


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv-ReLU-Conv-ReLU-MaxPool, as in Fig. 1 of the paper (the pooling halves T)."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2),
    )


class CNET(nn.Module):
    """Contact estimation network: ``(N, 150, 48)`` -> logits over the 16 contact states."""

    def __init__(self, input_channels: int = INPUT_CHANNELS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.block1 = _conv_block(input_channels, 64)
        self.block2 = _conv_block(64, 128)
        self.block3 = _conv_block(128, 256)

        self.gru_early = nn.GRU(128, 64, batch_first=True)
        self.gru_late = nn.GRU(256, 64, batch_first=True)

        self.main_classifier = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(inplace=True), nn.Dropout(0.0),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.0),
            nn.Linear(256, num_classes),
        )
        self.auxi_classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Dropout(0.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of the 16 contact states for a window of shape ``(N, 150, 48)``."""
        x = x.transpose(1, 2)                        # (N, 48, T) for the 1-D convolutions
        early = self.block2(self.block1(x))          # (N, 128, T/4)
        late = self.block3(early)                    # (N, 256, T/8)
        _, h_early = self.gru_early(early.transpose(1, 2))
        _, h_late = self.gru_late(late.transpose(1, 2))
        return self.main_classifier(torch.cat([h_early[-1], h_late[-1]], dim=-1))

    def predict_contacts(self, x: torch.Tensor) -> torch.Tensor:
        """Soft contact probability per foot, ``(N, 4)`` ordered (LF, RF, LH, RH)."""
        return contact_probabilities(self.forward(x))


def load_cnet(checkpoint_path: str | Path = CHECKPOINT_PATH, device: str | torch.device = "cpu") -> CNET:
    """Load ``CNET.ckpt`` (a PyTorch Lightning checkpoint) into an evaluation-ready model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Lightning stores the network under the "model." prefix of its LightningModule
    state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}

    hparams = checkpoint.get("hyper_parameters", {})
    model = CNET(
        input_channels=hparams.get("input_channels", INPUT_CHANNELS),
        num_classes=hparams.get("num_classes", NUM_CLASSES),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def class_to_contacts(class_index: torch.Tensor, num_feet: int = NUM_FEET) -> torch.Tensor:
    """Decode class indices into per-foot contact flags, ordered like :data:`LEG_ORDER`."""
    bits = torch.arange(num_feet, device=class_index.device)
    shifts = (num_feet - 1 - bits) if CLASS_MSB_FIRST else bits
    return ((class_index.unsqueeze(-1) >> shifts) & 1).float()


def contact_probabilities(logits: torch.Tensor, num_feet: int = NUM_FEET) -> torch.Tensor:
    """Marginal probability of contact per foot, by summing the classes where it is in contact."""
    probabilities = torch.softmax(logits, dim=-1)
    classes = torch.arange(logits.shape[-1], device=logits.device)
    table = class_to_contacts(classes, num_feet)          # (num_classes, num_feet)
    return probabilities @ table
