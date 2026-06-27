"""Bots scriptés — étalons absolus pour l'évaluation automatique.

Contrairement à l'ELO league (relatif aux propres snapshots du learner),
battre le chase-bot est une mesure ABSOLUE de niveau : poursuite directe,
aim parfait (dans la limite d'humanisation), sprint constant, clic permanent.
Un learner correct doit atteindre ~95%+ de winrate contre lui.

Lit uniquement l'observation (mêmes informations que le réseau).
"""

import torch

from .model import (
    COMBO_REHIT_ATTACK_REACH,
    COMBO_REHIT_EDGE_BRAKE_REACH,
    COUNTER_CLOSE_COUNTER_REACH,
    COUNTER_CLOSE_RECOVERY_CLICK_HURT,
    COUNTER_FAR_TRADE_REACH,
    COUNTER_HIT_SELECT_CLEAN_HURT,
    COUNTER_HIT_SELECT_CLEAN_MAX_REACH,
    COUNTER_HIT_SELECT_CLEAN_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_OWN_HURT,
)

RAD2DEG = 57.29577951308232
COMBO_REHIT_PRESS_HURT = 0.55
COMBO_REHIT_CLICK_HURT = 0.55


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

class ComboPadBot(ChaseBot):
    """Poursuit et garde la pression, mais n'attaque pas.

    Utilise comme curriculum de combo: l'adversaire reste a portee sans
    transformer chaque echange en trade hit+hit.
    """

    @torch.no_grad()
    def act7(self, hist: torch.Tensor) -> torch.Tensor:
        a = super().act7(hist)
        a[:, 6] = 0.0
        return a


class ComboSparBot(ChaseBot):
    """Active sparring bot: chases, attacks, counters, and tap-resets.

    Unlike ChaseBot, it does not hold attack from too far away and it tap-resets
    while ahead in combo. This keeps the opponent active without turning every
    close exchange into a guaranteed trade.
    """
    counter_release_hurt = 0.45

    def _counter_release_hurt(self, hist: torch.Tensor) -> torch.Tensor:
        batch = hist.shape[0]
        idx = torch.arange(batch, device=hist.device)
        easy = torch.full((batch,), 0.30, device=hist.device, dtype=hist.dtype)
        medium = torch.full_like(easy, self.counter_release_hurt)
        hard = torch.full_like(easy, self.counter_release_hurt)
        contest = torch.full_like(easy, 0.95)
        lane = idx % 4
        return torch.where(
            lane == 0,
            easy,
            torch.where(lane == 1, medium, torch.where(lane == 2, hard, contest)),
        )

    @torch.no_grad()
    def act7(self, hist: torch.Tensor) -> torch.Tensor:
        a = super().act7(hist)
        o = hist[:, -1]
        dist = o[:, 45] * 8.0
        combo_adv = o[:, 22] > o[:, 21] + 0.05
        under_combo = o[:, 21] > o[:, 22] + 0.05
        close = dist <= 3.20
        far_trade = dist > 3.35

        active_exchange = combo_adv | under_combo
        a[:, 6] = (close & active_exchange).float()
        a[far_trade, 6] = 0.0
        a[:, 3] = 0.0
        a[:, 4] = 0.0

        close_combo = combo_adv & (dist <= 3.05)
        too_close_combo = combo_adv & (dist <= 2.35)
        last_fwd = o[:, 40]
        last_sprint = o[:, 43] > 0.5
        last_release = (last_fwd <= 0.1) & (~last_sprint)
        last_tapped = (last_fwd <= 0.1) & (~last_sprint)
        hold_reset = combo_adv & (dist <= 1.65) & last_release
        tap_now = (too_close_combo & (~last_tapped)) | hold_reset
        repress = close_combo & last_tapped & (~hold_reset)
        a[tap_now, 2] = 0.0
        a[tap_now, 5] = 0.0
        a[repress, 2] = 1.0
        a[repress, 5] = 1.0

        counter_now = under_combo & (dist <= 3.15)
        a[counter_now, 6] = 1.0
        a[under_combo & (dist > 3.35), 6] = 0.0
        learning_window = under_combo & (o[:, 21] > self._counter_release_hurt(hist))
        a[learning_window, 6] = 0.0
        a[learning_window, 2] = torch.where(
            dist[learning_window] > 2.75,
            torch.ones_like(a[learning_window, 2]),
            torch.zeros_like(a[learning_window, 2]),
        )
        a[learning_window, 5] = (a[learning_window, 2] > 0.5).float()
        return a


class ComboRehitBot(ComboSparBot):
    """Re-hit drill opponent.

    It still chases, aims, tap-resets when ahead, and counters if the learner
    misses the re-hit window. While it is freshly hurt, it holds fire long
    enough for the learner to practice a real W/Z release rechain instead of
    learning only "trade then counter".
    """
    counter_release_hurt = 0.30

    def _counter_release_hurt(self, hist: torch.Tensor) -> float | torch.Tensor:
        return self.counter_release_hurt

    @torch.no_grad()
    def act7(self, hist: torch.Tensor) -> torch.Tensor:
        a = super().act7(hist)
        o = hist[:, -1]
        dist = o[:, 45] * 8.0
        own_hurt = o[:, 21]
        opp_hurt = o[:, 22]
        combo_adv = opp_hurt > own_hurt + 0.05
        under_combo = own_hurt > opp_hurt + 0.05

        # If the bot is the victim of a fresh learner hit, do not immediately
        # steal the turn back. It starts punishing after the legal re-hit window
        # has clearly been missed.
        learning_window = under_combo & (own_hurt > self._counter_release_hurt(hist))
        a[learning_window, 6] = 0.0
        a[learning_window, 2] = torch.where(
            dist[learning_window] > 2.75,
            torch.ones_like(a[learning_window, 2]),
            torch.zeros_like(a[learning_window, 2]),
        )
        a[learning_window, 5] = (a[learning_window, 2] > 0.5).float()

        # When the bot is ahead, use the same one-tick-earlier click timing as
        # the learner; obs hurt=11 becomes 10 before attacks are resolved.
        wait_own_rehit = combo_adv & (opp_hurt > COMBO_REHIT_CLICK_HURT)
        a[wait_own_rehit, 6] = 0.0
        press_own_rehit = (
            combo_adv
            & (opp_hurt <= COMBO_REHIT_PRESS_HURT)
            & (opp_hurt > COMBO_REHIT_CLICK_HURT)
            & (dist <= 3.05)
        )
        a[press_own_rehit, 2] = 1.0
        a[press_own_rehit, 5] = 1.0
        return a


class ComboPressureBot(ComboRehitBot):
    """Intermediate active drill between re-hit pad and full spar.

    Half the rows leave a very short legal re-hit window, while the other half
    contests on the first legal re-hit tick. This keeps the drill learnable
    while forcing transfer toward active spar instead of solving only pad timing.
    """
    counter_release_hurt = 0.45

    def _counter_release_hurt(self, hist: torch.Tensor) -> torch.Tensor:
        batch = hist.shape[0]
        idx = torch.arange(batch, device=hist.device)
        easy = torch.full((batch,), 0.30,
                          device=hist.device, dtype=hist.dtype)
        hard = torch.full_like(easy, COMBO_REHIT_CLICK_HURT)
        return torch.where((idx % 2) == 0, easy, hard)


class ComboChaseBot(ComboSparBot):
    """Aggressive live-like chaser for combo transfer.

    It rushes and contests more often than ComboSparBot, but it avoids the
    permanent neutral pre-fire of ChaseBot. That keeps the drill beatable:
    the learner must open, tap-reset/re-hit, and counter while being comboed
    instead of learning to accept one-hit trades.
    """

    def _counter_release_hurt(self, hist: torch.Tensor) -> torch.Tensor:
        batch = hist.shape[0]
        idx = torch.arange(batch, device=hist.device)
        easy = torch.full((batch,), 0.30, device=hist.device, dtype=hist.dtype)
        medium = torch.full_like(easy, 0.45)
        hard = torch.full_like(easy, COMBO_REHIT_CLICK_HURT)
        contest = torch.full_like(easy, 0.90)
        lane = idx % 4
        return torch.where(
            lane == 0,
            easy,
            torch.where(lane == 1, medium, torch.where(lane == 2, hard, contest)),
        )

    @torch.no_grad()
    def act7(self, hist: torch.Tensor) -> torch.Tensor:
        a = super().act7(hist)
        o = hist[:, -1]
        batch = hist.shape[0]
        idx = torch.arange(batch, device=hist.device)
        lane = idx % 4
        dist = o[:, 45] * 8.0
        combo_adv = o[:, 22] > o[:, 21] + 0.05
        under_combo = o[:, 21] > o[:, 22] + 0.05
        neutral = (~combo_adv) & (~under_combo)

        # A subset of rows probes very close neutral openings. Other rows leave
        # the first hit to the learner, so the curriculum still teaches clean
        # openers instead of only symmetric pre-click trades.
        neutral_probe = neutral & (dist <= 2.85) & ((lane == 1) | (lane == 3))
        a[neutral_probe, 6] = 1.0

        # If the bot is behind and the learner lets it close again, contest
        # only after the lane-specific re-hit window. The previous fixed 0.70
        # threshold stole the turn while the learner was still practicing a
        # legal rechain, which over-trained countering instead of combos.
        early_contest = (
            under_combo
            & (dist <= 3.25)
            & (o[:, 21] <= self._counter_release_hurt(hist))
        )
        a[early_contest, 6] = 1.0

        # Edge lanes punish overrunning the re-hit pocket. The learner must
        # brake/release before reliable re-hit reach instead of sprinting into
        # the active chaser and accepting a stolen contact trade.
        edge_contest = (
            under_combo
            & (lane == 3)
            & (o[:, 21] <= self._counter_release_hurt(hist))
            & (dist > COMBO_REHIT_ATTACK_REACH)
            & (dist <= torch.minimum(
                torch.as_tensor(COMBO_REHIT_EDGE_BRAKE_REACH, device=hist.device, dtype=dist.dtype),
                torch.as_tensor(COUNTER_FAR_TRADE_REACH, device=hist.device, dtype=dist.dtype),
            ))
        )
        a[edge_contest, 6] = 1.0
        return a


class ComboCounterBot(ComboChaseBot):
    """Active recovery drill for escaping a combo without teaching backpedal.

    The bot starts exchanges aggressively, keeps forward sprint pressure while
    ahead, then alternates between short legal recovery pockets and contested
    pockets. The learner has to hit-select the open pocket instead of holding
    click, backing away, or accepting a trade.
    """

    def _counter_release_hurt(self, hist: torch.Tensor) -> torch.Tensor:
        batch = hist.shape[0]
        idx = torch.arange(batch, device=hist.device)
        easy = torch.full((batch,), 0.25, device=hist.device, dtype=hist.dtype)
        medium = torch.full_like(easy, 0.38)
        hard = torch.full_like(easy, COMBO_REHIT_CLICK_HURT)
        contest = torch.full_like(easy, 0.78)
        lane = idx % 4
        return torch.where(
            lane == 0,
            easy,
            torch.where(lane == 1, medium, torch.where(lane == 2, hard, contest)),
        )

    @torch.no_grad()
    def act7(self, hist: torch.Tensor) -> torch.Tensor:
        a = super().act7(hist)
        o = hist[:, -1]
        batch = hist.shape[0]
        idx = torch.arange(batch, device=hist.device)
        lane = idx % 4
        dist = o[:, 45] * 8.0
        combo_adv = o[:, 22] > o[:, 21] + 0.05
        under_combo = o[:, 21] > o[:, 22] + 0.05
        neutral = (~combo_adv) & (~under_combo)

        close = dist <= 3.20
        far_trade = dist > COUNTER_FAR_TRADE_REACH

        opener = neutral & (dist <= 3.05)
        a[opener, 6] = 1.0
        a[far_trade, 6] = 0.0

        bot_ahead = combo_adv & close
        recovery_lane = (lane == 0) | (lane == 1) | (lane == 2)
        close_counter_recovery_window = (
            bot_ahead
            & recovery_lane
            & (o[:, 22] <= COUNTER_CLOSE_RECOVERY_CLICK_HURT)
            & (dist >= 1.80)
            & (dist <= COUNTER_CLOSE_COUNTER_REACH)
        )
        hit_select_recovery_window = (
            bot_ahead
            & recovery_lane
            & (o[:, 22] >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
            & (o[:, 22] <= COUNTER_HIT_SELECT_CLEAN_HURT)
            & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
            & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
        )
        fresh_combo = bot_ahead & (o[:, 22] > 0.45) & (~hit_select_recovery_window)
        patient_recovery_window = (
            bot_ahead
            & (o[:, 22] <= 0.45)
            & ((lane == 0) | (lane == 1))
            & (dist >= 2.20)
        )
        delayed_recovery_window = (
            bot_ahead
            & (o[:, 22] <= 0.32)
            & (lane == 2)
            & (dist >= 2.35)
            & (dist <= 3.15)
        )
        recovery_window = (
            close_counter_recovery_window
            | hit_select_recovery_window
            | patient_recovery_window
            | delayed_recovery_window
        )
        contest_window = (
            bot_ahead
            & (o[:, 22] <= 0.45)
            & (lane == 3)
        )
        overclose_reset = combo_adv & (dist <= 2.20)
        drive = combo_adv & (dist > 2.65) & (dist <= COUNTER_FAR_TRADE_REACH) & (~hit_select_recovery_window)

        a[fresh_combo, 6] = 1.0
        a[recovery_window, 6] = 0.0
        a[contest_window, 6] = 1.0
        a[close_counter_recovery_window, 2] = 0.0
        a[close_counter_recovery_window, 5] = 0.0
        a[hit_select_recovery_window, 2] = 0.0
        a[hit_select_recovery_window, 5] = 0.0
        a[drive, 2] = 1.0
        a[drive, 5] = 1.0
        a[overclose_reset, 2] = 0.0
        a[overclose_reset, 5] = 0.0

        # If the learner gets the turn, preserve the active spar behaviour:
        # give some rows a re-hit window and others a contest window.
        recontest = under_combo & (dist <= 3.20) & (o[:, 21] <= self._counter_release_hurt(hist))
        a[recontest & (lane >= 2), 6] = 1.0
        a[under_combo & (lane < 2) & (o[:, 21] > self._counter_release_hurt(hist)), 6] = 0.0
        return a
