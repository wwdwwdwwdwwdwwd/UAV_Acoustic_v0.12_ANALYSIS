"""Shared offline/realtime AZ x EL normalized multiband spatial search."""
from __future__ import annotations

import time
import numpy as np

from ..coordinates import az_el_to_unit


class NormalizedMultibandAzElSearch:
    """Far-field direction-grid search with v0.3.1's aggregation semantics."""

    def __init__(self, mic_xyz: np.ndarray, frequencies_hz: np.ndarray,
                 band_local_indices: list[np.ndarray], band_weights: np.ndarray,
                 config: dict | None = None, *, sound_speed_mps: float = 343.0):
        xyz = np.asarray(mic_xyz, dtype=float)
        self.xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
        self.frequency = np.asarray(frequencies_hz, dtype=float)
        self.band_local_indices = [np.asarray(value, dtype=int) for value in band_local_indices]
        self.band_weights = np.asarray(band_weights, dtype=float)
        self.c = float(sound_speed_mps)
        self._steering_cache: dict[tuple, np.ndarray] = {}
        self._dynamic_caches: dict[tuple[float,...],dict[tuple,np.ndarray]] = {}
        cfg = config or {}
        self.fov = tuple(map(float, cfg.get("fov_deg", (-90.0, 90.0))))
        self.el_fov = tuple(map(float, cfg.get("el_fov_deg", (-45.0, 45.0))))
        self.coarse_step = float(cfg.get("acquire_coarse_step_deg", 2.0))
        self.fine_step = float(cfg.get("fine_step_deg", 0.2))
        self.fine_radius = float(cfg.get("fine_radius_deg", 2.0))
        self.edge_margin = float(cfg.get("edge_margin_deg", 0.5))
        if config is not None:
            # Keep fixed-grid exponential construction out of the first measured frame.
            coarse_az = self._axis(self.fov[0], self.fov[1], self.coarse_step)
            coarse_el = self._axis(self.el_fov[0], self.el_fov[1], self.coarse_step)
            az_mesh, el_mesh = np.meshgrid(coarse_az, coarse_el, indexing="xy")
            key_2d = ("azel", self.fov, self.el_fov, self.coarse_step)
            self._steering_cache[key_2d] = self.steering(az_mesh.ravel(), el_mesh.ravel())
            key_1d = ("az", self.fov, self.coarse_step)
            self._steering_cache[key_1d] = self.steering(coarse_az, np.zeros_like(coarse_az))

    def steering(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray,
                 frequencies_hz: np.ndarray | None = None) -> np.ndarray:
        directions = az_el_to_unit(azimuth_deg, elevation_deg).reshape(-1, 3)
        delay = directions @ self.xyz.T / self.c
        frequency=self.frequency if frequencies_hz is None else np.asarray(frequencies_hz,float)
        return np.exp(
            -2j * np.pi * delay[:, :, None] * frequency[None, None, :]
        ).astype(np.complex64)

    def power(self, steering: np.ndarray, target_spectra: np.ndarray,
              *, direction_batch: int = 512, tf_weights: np.ndarray | None = None) -> np.ndarray:
        """Exact v0.3.1 per-TF, per-band, temporal-median map in bounded memory."""
        spectra = np.asarray(target_spectra)
        denominator = len(self.xyz) * np.sum(np.abs(spectra) ** 2, axis=1)
        result = np.empty(steering.shape[0], dtype=float)
        for start in range(0, steering.shape[0], int(direction_batch)):
            stop = min(start + int(direction_batch), steering.shape[0])
            beam = np.einsum("dmf,kmf->kdf", steering[start:stop], spectra, optimize=True)
            normalized = np.abs(beam) ** 2 / (denominator[:, None, :] + 1e-30)
            if tf_weights is None:
                band_maps = np.stack([np.mean(normalized[:, :, index],axis=2) for index in self.band_local_indices],axis=1)
                per_subframe=np.average(band_maps,axis=1,weights=self.band_weights)
            else:
                weights=np.asarray(tf_weights,float)
                if weights.shape != (spectra.shape[0],spectra.shape[2]): raise ValueError("TF weights shape mismatch")
                per_subframe=np.sum(normalized*weights[:,None,:],axis=2)/(np.sum(weights,axis=1)[:,None]+1e-30)
            result[start:stop] = np.median(per_subframe, axis=0)
        return result

    # Compatibility names let existing v0.3.1 normalization regressions exercise
    # this shared implementation without maintaining a second beamformer.
    def _steering(self, azimuth_deg: np.ndarray,
                  elevation_deg: np.ndarray | None = None) -> np.ndarray:
        azimuth = np.asarray(azimuth_deg, dtype=float)
        elevation = np.zeros_like(azimuth) if elevation_deg is None else np.asarray(elevation_deg, dtype=float)
        return self.steering(azimuth, elevation)

    def _power(self, steering: np.ndarray, target_spectra: np.ndarray) -> np.ndarray:
        return self.power(steering, target_spectra)

    @staticmethod
    def _axis(start: float, stop: float, step: float) -> np.ndarray:
        # Anchor regular samples at zero so symmetric FOVs always contain EL=0.
        # Preserve non-aligned FOV endpoints as explicit boundary candidates.
        first = np.ceil(start / step) * step
        regular = np.arange(first, stop + step * 0.25, step)
        return np.unique(np.round(np.concatenate(([start], regular, [stop])), 10))

    def _evaluate_grid(self, target_spectra: np.ndarray, azimuth: np.ndarray,
                       elevation: np.ndarray, *, cache_key: tuple | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        az_mesh, el_mesh = np.meshgrid(azimuth, elevation, indexing="xy")
        steering = self._steering_cache.get(cache_key) if cache_key is not None else None
        if steering is None:
            steering = self.steering(az_mesh.ravel(), el_mesh.ravel())
            if cache_key is not None:
                self._steering_cache[cache_key] = steering
        values = self.power(steering, target_spectra).reshape(len(elevation), len(azimuth))
        return az_mesh, el_mesh, values

    def search_1d(self, target_spectra: np.ndarray, *, az_fov=(-90.0, 90.0),
                  coarse_step_deg=2.0, fine_step_deg=0.2,
                  fine_radius_deg=2.0) -> dict:
        started = time.perf_counter()
        coarse_az = self._axis(float(az_fov[0]), float(az_fov[1]), float(coarse_step_deg))
        key = ("az", tuple(map(float, az_fov)), float(coarse_step_deg))
        steering = self._steering_cache.get(key)
        if steering is None:
            steering = self.steering(coarse_az, np.zeros_like(coarse_az))
            self._steering_cache[key] = steering
        coarse = self.power(steering, target_spectra)
        centre = float(coarse_az[int(np.argmax(coarse))])
        fine_az = self._axis(max(float(az_fov[0]), centre - float(fine_radius_deg)),
                             min(float(az_fov[1]), centre + float(fine_radius_deg)),
                             float(fine_step_deg))
        fine = self.power(self.steering(fine_az, np.zeros_like(fine_az)), target_spectra)
        peak_index = int(np.argmax(fine)); peak = float(fine[peak_index])
        return {
            "azimuth_deg": float(fine_az[peak_index]), "elevation_deg": 0.0,
            "spatial_coherence": float(np.clip(peak, 0.0, 1.0)),
            "beam_prominence_db": float(10 * np.log10((peak + 1e-30) /
                                                        (float(np.median(coarse)) + 1e-30))),
            "processing_ms": (time.perf_counter() - started) * 1000.0,
        }

    def search_2d(self, target_spectra: np.ndarray, *, az_fov=(-90.0, 90.0),
                  el_fov=(-45.0, 45.0), coarse_step_deg=2.0,
                  fine_step_deg=0.2, fine_radius_deg=2.0) -> dict:
        started = time.perf_counter()
        coarse_az = self._axis(float(az_fov[0]), float(az_fov[1]), float(coarse_step_deg))
        coarse_el = self._axis(float(el_fov[0]), float(el_fov[1]), float(coarse_step_deg))
        key = ("azel", tuple(map(float, az_fov)), tuple(map(float, el_fov)),
               float(coarse_step_deg))
        az_mesh, el_mesh, coarse = self._evaluate_grid(
            target_spectra, coarse_az, coarse_el, cache_key=key
        )
        coarse_index = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
        centre_az = float(az_mesh[coarse_index]); centre_el = float(el_mesh[coarse_index])
        fine_az = self._axis(max(float(az_fov[0]), centre_az - float(fine_radius_deg)),
                             min(float(az_fov[1]), centre_az + float(fine_radius_deg)),
                             float(fine_step_deg))
        fine_el = self._axis(max(float(el_fov[0]), centre_el - float(fine_radius_deg)),
                             min(float(el_fov[1]), centre_el + float(fine_radius_deg)),
                             float(fine_step_deg))
        fine_az_mesh, fine_el_mesh, fine = self._evaluate_grid(target_spectra, fine_az, fine_el)
        fine_index = np.unravel_index(int(np.argmax(fine)), fine.shape)
        peak = float(fine[fine_index])
        az_slice = fine[fine_index[0], :]
        threshold = peak * 10 ** (-3 / 10)
        left = fine_index[1]; right = fine_index[1]
        while left > 0 and az_slice[left - 1] >= threshold:
            left -= 1
        while right + 1 < len(az_slice) and az_slice[right + 1] >= threshold:
            right += 1
        return {
            "azimuth_deg": float(fine_az_mesh[fine_index]),
            "elevation_deg": float(fine_el_mesh[fine_index]),
            "coarse_peak_az_deg": centre_az, "coarse_peak_el_deg": centre_el,
            "spatial_coherence": float(np.clip(peak, 0.0, 1.0)),
            "beam_prominence_db": float(10 * np.log10((peak + 1e-30) /
                                                        (float(np.median(coarse)) + 1e-30))),
            "mainlobe_width_deg": float(fine_az[right] - fine_az[left]),
            "second_peak_ratio_db": None,
            "beam_peak_power": peak,
            "fine_azimuth_deg": fine_az, "fine_elevation_deg": fine_el,
            "fine_response": fine / max(peak, 1e-30),
            "processing_ms": (time.perf_counter() - started) * 1000.0,
        }

    def estimate(self, target_spectra: np.ndarray, previous_az: float | None,
                 force_reacquire: bool = False) -> dict:
        """Joint 2D measurement plus an EL=0 reference used only for logging."""
        result = self.search_2d(
            target_spectra, az_fov=self.fov, el_fov=self.el_fov,
            coarse_step_deg=self.coarse_step, fine_step_deg=self.fine_step,
            fine_radius_deg=self.fine_radius,
        )
        reference = self.search_1d(
            target_spectra, az_fov=self.fov, coarse_step_deg=self.coarse_step,
            fine_step_deg=self.fine_step, fine_radius_deg=self.fine_radius,
        )
        if force_reacquire:
            mode = "REACQUIRE"
        else:
            mode = "ACQUIRE" if previous_az is None else "TRACK"
        azimuth = float(result["azimuth_deg"]); elevation = float(result["elevation_deg"])
        edge = bool(
            abs(azimuth - self.fov[0]) <= self.edge_margin or
            abs(azimuth - self.fov[1]) <= self.edge_margin or
            abs(elevation - self.el_fov[0]) <= self.edge_margin or
            abs(elevation - self.el_fov[1]) <= self.edge_margin
        )
        return {
            **result, "az_1d_reference_deg": float(reference["azimuth_deg"]),
            "search_mode": mode, "edge_of_fov": edge,
        }

    def estimate_dynamic(self, target_spectra: np.ndarray, frequencies_hz: np.ndarray,
                         tf_weights: np.ndarray, previous_az: float | None,
                         force_reacquire: bool=False) -> dict:
        """Same AZ/EL grids and normalization, with a shared dynamic TF layout."""
        old_frequency=self.frequency; old_bands=self.band_local_indices; old_weights=self.band_weights
        self.frequency=np.asarray(frequencies_hz,float)
        self.band_local_indices=[np.arange(len(self.frequency))]; self.band_weights=np.ones(1)
        original_power=self.power
        self.power=lambda steering,spectra,**kw: original_power(steering,spectra,tf_weights=tf_weights,**kw)
        # Dynamic layouts must not reuse fixed-frequency steering matrices.
        cache=self._steering_cache
        layout=tuple(np.round(self.frequency,6))
        self._steering_cache=self._dynamic_caches.setdefault(layout,{})
        try: return self.estimate(target_spectra,previous_az,force_reacquire)
        finally:
            self.power=original_power; self.frequency=old_frequency; self.band_local_indices=old_bands
            self.band_weights=old_weights; self._steering_cache=cache
