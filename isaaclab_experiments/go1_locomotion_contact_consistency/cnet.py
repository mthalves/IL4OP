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

The window is counted in environment steps, so at 50 Hz it spans 3 s, where the published
model was trained on 150 ms of 1 kHz logs.
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
# The published pipeline standardizes its inputs and does not give the statistics, so these
# were measured instead: 40 s of Go1 locomotion in MuJoCo at 1 kHz over ten commands, from
# standing to 1 m/s, sideways and turning (tools/sim2sim, with a trained policy). 
#
# Re-measure them for another robot or another gait; the statistics are of the data, not of
# the network. The channel order is the one of this module: q, qd, p_f, v_f, leg-major.
INPUT_MEAN: torch.Tensor = torch.tensor([
    +0.1319, +0.7204, -1.5281, -0.1523, +0.6746, -1.5578,
    +0.1549, +0.9451, -1.6257, -0.1530, +0.9715, -1.5875,
    +0.0116, +0.0175, +0.0051, -0.0072, -0.0226, +0.0145,
    -0.0037, -0.0016, -0.0185, +0.0045, +0.0297, -0.0039,
    +0.1990, +0.1653, -0.2832, +0.2168, -0.1705, -0.2760,
    -0.2288, +0.1699, -0.2681, -0.2426, -0.1696, -0.2688,
    -0.0065, +0.0032, +0.0063, +0.0119, +0.0014, +0.0033,
    +0.0033, -0.0024, -0.0012, -0.0064, -0.0072, +0.0048,
])
INPUT_STD: torch.Tensor = torch.tensor([
    +0.0844, +0.2451, +0.1905, +0.0603, +0.2452, +0.1664,
    +0.0741, +0.1494, +0.1631, +0.0880, +0.1989, +0.1648,
    +1.1022, +3.1581, +3.4028, +0.7871, +2.9684, +3.2517,
    +0.8016, +1.8476, +2.7371, +0.9977, +2.3178, +2.9143,
    +0.0700, +0.0255, +0.0326, +0.0688, +0.0191, +0.0299,
    +0.0500, +0.0222, +0.0235, +0.0648, +0.0260, +0.0223,
    +0.7514, +0.2654, +0.5171, +0.7536, +0.2498, +0.4800,
    +0.6303, +0.2235, +0.4169, +0.6884, +0.2851, +0.4017,
])


def normalize(features):
    """Standardize the channels of a feature vector, or of a whole window of them.

    Takes a torch tensor or a numpy array and gives back the same kind, so the environments
    and the sim-to-sim harness normalize their inputs the one way.
    """
    if isinstance(features, torch.Tensor):
        return (features - INPUT_MEAN.to(features.device)) / INPUT_STD.to(features.device)
    return (features - INPUT_MEAN.numpy()) / INPUT_STD.numpy()

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
