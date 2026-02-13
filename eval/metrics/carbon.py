"""Carbon footprint tracking via CodeCarbon.

Provides a context manager that wraps ``codecarbon.EmissionsTracker`` and
returns a ``CarbonResult`` populated after the block exits.  Degrades
gracefully to a no-op when ``codecarbon`` is not installed.

Typical usage inside an experiment::

    from eval.metrics.carbon import track_emissions

    with track_emissions("rainrag_eval") as carbon:
        results = engine.query(...)

    metrics = carbon.as_metrics()   # dict[str, float], empty if unavailable
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Generator


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CarbonResult:
    """Emission measurements collected for one tracked block.

    Attributes:
        available: ``True`` when codecarbon is installed and tracking succeeded.
        emissions_kg: Total CO₂-equivalent in kilograms.
        energy_kwh: Total energy consumed in kWh.
        cpu_power_w: Mean CPU power draw in watts (if reported by codecarbon).
        duration_s: Duration of the tracked block in seconds.
    """

    available: bool = False
    emissions_kg: float | None = None
    energy_kwh: float | None = None
    cpu_power_w: float | None = None
    duration_s: float | None = None

    def as_metrics(self) -> dict[str, float]:
        """Return a flat dict suitable for ``mlflow_tracking.log_metrics()``.

        Returns an empty dict when tracking is unavailable or all values are
        ``None``.
        """
        if not self.available:
            return {}
        out: dict[str, float] = {}
        if self.emissions_kg is not None:
            out["carbon.emissions_kg"] = self.emissions_kg
            out["carbon.emissions_g"] = self.emissions_kg * 1_000
        if self.energy_kwh is not None:
            out["carbon.energy_kwh"] = self.energy_kwh
        if self.cpu_power_w is not None:
            out["carbon.cpu_power_w"] = self.cpu_power_w
        if self.duration_s is not None:
            out["carbon.duration_s"] = self.duration_s
        return out


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def track_emissions(
    project_name: str = "rainrag_eval",
    country_iso_code: str = "USA",
    log_level: str = "error",
) -> Generator[CarbonResult, None, None]:
    """Context manager that tracks CO₂ emissions for the enclosed block.

    Args:
        project_name: Logical name for the CodeCarbon project.
        country_iso_code: ISO-3166 alpha-3 code used for carbon intensity lookup.
        log_level: CodeCarbon internal log level ("error" suppresses most output).

    Yields:
        A :class:`CarbonResult` that is populated once the ``with`` block exits.
        If ``codecarbon`` is not installed, ``result.available`` is ``False``
        and ``result.as_metrics()`` returns an empty dict.

    Example::

        with track_emissions("my_experiment") as carbon:
            do_heavy_work()

        print(carbon.emissions_kg)   # None if unavailable
        mlflow.log_metrics(carbon.as_metrics())
    """
    try:
        from codecarbon import EmissionsTracker  # type: ignore[import]
    except ImportError:
        # Graceful no-op: yield an empty result without blocking the caller
        result = CarbonResult(available=False)
        yield result
        return

    result = CarbonResult(available=True)
    tracker = EmissionsTracker(
        project_name=project_name,
        country_iso_code=country_iso_code,
        log_level=log_level,
        save_to_file=False,
        save_to_api=False,
        save_to_logger=False,
    )
    tracker.start()
    try:
        yield result
    finally:
        try:
            emissions_kg = tracker.stop()
            if emissions_kg is not None:
                result.emissions_kg = float(emissions_kg)

            # Richer metrics from EmissionsData (codecarbon >= 2.3)
            data = getattr(tracker, "final_emissions_data", None)
            if data is not None:
                _energy = getattr(data, "energy_consumed", None)
                if _energy is not None:
                    result.energy_kwh = float(_energy)

                _cpu = getattr(data, "cpu_power", None)
                if _cpu is not None:
                    result.cpu_power_w = float(_cpu)

                _dur = getattr(data, "duration", None)
                if _dur is not None:
                    result.duration_s = float(_dur)
        except Exception:
            # Never let tracking errors propagate into experiment code
            result.available = False
