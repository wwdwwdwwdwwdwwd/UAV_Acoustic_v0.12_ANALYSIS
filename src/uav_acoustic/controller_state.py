"""Pure two-Enter controller state machine with automatic active recovery."""
from __future__ import annotations


class ControllerState:
    def __init__(self, recovery_s=3.0):
        self.state = "WAIT_FOR_CALIBRATION"
        self.baseline_ready = False
        self.detection_started = False
        self.paused = False
        self.prior_state = None
        self.healthy_since = None
        self.recovery_s = float(recovery_s)

    def set_state(self, new_state, reason):
        previous = self.state
        self.state = str(new_state)
        if previous == self.state:
            return None
        return {"previous_state": previous, "new_state": self.state, "reason": str(reason)}

    def enter(self):
        if self.state == "WAIT_FOR_CALIBRATION":
            self.set_state("NOISE_CALIBRATION", "first_enter")
            return "START_CALIBRATION"
        if self.state == "WAIT_FOR_MEASUREMENT" and self.baseline_ready and not self.detection_started:
            self.detection_started = True
            self.set_state("SEARCH_IDLE", "second_enter")
            return "START_DETECTION"
        return "IGNORED"

    def calibration_complete(self, success):
        self.baseline_ready = bool(success)
        self.set_state("WAIT_FOR_MEASUREMENT" if success else "WAIT_FOR_CALIBRATION", "calibration_complete" if success else "calibration_failed")

    def recalibrate(self):
        self.baseline_ready = False
        self.detection_started = False
        self.paused = False
        self.set_state("WAIT_FOR_CALIBRATION", "operator_recalibrate")

    def pause(self):
        if not self.detection_started:
            return
        self.paused = not self.paused
        self.set_state("PAUSED" if self.paused else "SEARCH_IDLE", "operator_pause" if self.paused else "operator_resume")

    def acquisition_health(self, healthy, now):
        if not healthy:
            if self.state != "ACQ_UNHEALTHY":
                self.prior_state = self.state
            self.set_state("ACQ_UNHEALTHY", "acquisition_unhealthy")
            self.healthy_since = None
            return
        if self.state == "ACQ_UNHEALTHY":
            if self.healthy_since is None:
                self.healthy_since = now
            elif now - self.healthy_since >= self.recovery_s:
                if self.detection_started and self.baseline_ready:
                    recovered = "SEARCH_IDLE"
                elif self.baseline_ready:
                    recovered = "WAIT_FOR_MEASUREMENT"
                else:
                    recovered = "WAIT_FOR_CALIBRATION"
                self.set_state(recovered, "acquisition_recovered")
                self.healthy_since = None
