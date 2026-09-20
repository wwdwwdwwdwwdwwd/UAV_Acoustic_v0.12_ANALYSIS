from __future__ import annotations
import numpy as np


class AngleEMA:
    """Very small baseline smoother for azimuth/elevation.

    Replace with Kalman/particle tracking once real trajectories are available.
    """
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.state: np.ndarray | None = None

    def update(self, az_deg: float, el_deg: float) -> tuple[float,float]:
        z = np.array([az_deg, el_deg], dtype=float)
        if self.state is None:
            self.state = z
        else:
            # Wrap azimuth residual to [-180,180]
            daz = ((z[0]-self.state[0]+180) % 360) - 180
            self.state[0] = (self.state[0] + self.alpha*daz + 180) % 360 - 180
            self.state[1] = (1-self.alpha)*self.state[1] + self.alpha*z[1]
        return float(self.state[0]), float(self.state[1])
