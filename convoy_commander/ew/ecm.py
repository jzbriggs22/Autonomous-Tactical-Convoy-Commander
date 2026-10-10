"""Electronic countermeasures (ECM) state for vehicles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ECMState:
    """Per-vehicle ECM state.

    Frequency hopping reduces the effective jammer degradation by
    ``freq_hop_loss_reduction``.  Adaptive power boost further mitigates
    loss by increasing transmit power.
    """

    freq_hopping_active: bool = False
    freq_hop_loss_reduction: float = 0.6
    adaptive_power_boost: float = 1.0  # [1.0, 2.0]

    def effective_jammer_degradation(self, raw_degradation: float) -> float:
        """Apply ECM to reduce jammer effectiveness.

        When frequency hopping is active::

            eff = 1.0 + (raw - 1.0) * freq_hop_loss_reduction / adaptive_power_boost

        Returns *raw_degradation* unchanged if ECM is inactive.
        """
        if not self.freq_hopping_active or raw_degradation <= 1.0:
            return raw_degradation
        reduced = (raw_degradation - 1.0) * self.freq_hop_loss_reduction
        if self.adaptive_power_boost > 1.0:
            reduced /= self.adaptive_power_boost
        return 1.0 + reduced
