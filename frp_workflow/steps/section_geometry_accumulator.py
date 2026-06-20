from __future__ import annotations

"""Append-only accumulator for section geometry outputs."""

from dataclasses import dataclass


@dataclass(slots=True)
class SectionGeometryAccumulator:
    centers_xyz: list[tuple[float, float, float]]
    centers_xyz_id: list[tuple[float, float, float]]
    concentricity_list: list[float]

    def append_od_center(self, center: tuple[float, float, float]) -> None:
        self.centers_xyz.append(center)

    def append_id_center(self, center: tuple[float, float, float]) -> None:
        self.centers_xyz_id.append(center)

    def append_concentricity(self, value: float) -> None:
        self.concentricity_list.append(value)


__all__ = ["SectionGeometryAccumulator"]
