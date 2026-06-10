"""Bots scriptés — étalons absolus pour l'évaluation automatique.

Contrairement à l'ELO league (relatif aux propres snapshots du learner),
battre le chase-bot est une mesure ABSOLUE de niveau : poursuite directe,
aim parfait (dans la limite d'humanisation), sprint constant, clic permanent.
Un learner correct doit atteindre ~95%+ de winrate contre lui.

Lit uniquement l'observation (mêmes informations que le réseau).
"""

import torch

RAD2DEG = 57.29577951308232


class ChaseBot:
    """Poursuite + aim + clic. act7 : [B, H, OBS_DIM] -> actions sim [B, 7]."""

    @torch.no_grad()
    def act7(self, hist: torch.Tensor) -> torch.Tensor:
        o = hist[:, -1]                                # dernier tick
        rot = (o[:, 36] * 180.0).clamp(min=1.0)        # vitesse rot max (°/tick)
        yaw_err_deg = torch.atan2(o[:, 11], o[:, 12]) * RAD2DEG
        pitch_err_deg = o[:, 13] * 90.0
        a = torch.zeros(hist.shape[0], 7, dtype=torch.float32, device=hist.device)
        a[:, 0] = (yaw_err_deg / rot).clamp(-1.0, 1.0)
        a[:, 1] = (pitch_err_deg / rot).clamp(-1.0, 1.0)
        a[:, 2] = 1.0                                  # forward
        a[:, 5] = 1.0                                  # sprint
        a[:, 6] = 1.0                                  # attack
        return a
