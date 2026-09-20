"""Mathematical engine for GRIDPOINT.

The optimizer places k warehouses so that high-demand neighbourhoods are, on
average, closer to a warehouse.  It uses a weighted k-median approach:
  1. Start with well-spread candidate locations.
  2. Assign every neighbourhood to the nearest warehouse.
  3. Move each warehouse towards the weighted geometric median of its demand.
  4. Repeat and retain the best of several starts.

All public functions use latitude/longitude values and distances in kilometres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class CostBreakdown:
    """Cost and service metrics for one proposed warehouse network."""

    weighted_distance_km: float
    delivery_cost: float
    average_distance_km: float
    served_orders: float
    total_orders: float
    coverage_percent: float
    distances_km: np.ndarray


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres; supports NumPy broadcasting."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_matrix(points: np.ndarray, warehouses: np.ndarray) -> np.ndarray:
    """Return an n-by-k matrix of straight-line distances in kilometres."""
    return haversine_km(
        points[:, 0, None],
        points[:, 1, None],
        warehouses[None, :, 0],
        warehouses[None, :, 1],
    )


def assign_nearest(points: np.ndarray, warehouses: np.ndarray) -> np.ndarray:
    """Assign each neighbourhood to the closest warehouse."""
    return distance_matrix(points, warehouses).argmin(axis=1)


def assign_with_limits(
    points: np.ndarray,
    orders: np.ndarray,
    warehouses: np.ndarray,
    capacity: float | None = None,
    max_radius: float | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Assign demand while respecting optional capacity and service-radius limits.

    The largest demand centres are allocated first. This makes the trade-off
    explicit: a tight capacity or radius can leave some demand unserved.
    """
    distances = distance_matrix(points, warehouses)
    assignment = np.full(len(points), -1, dtype=int)
    loads = np.zeros(len(warehouses), dtype=float)
    unserved: list[int] = []

    for point_idx in np.argsort(-orders):
        for warehouse_idx in np.argsort(distances[point_idx]):
            distance = distances[point_idx, warehouse_idx]
            if max_radius is not None and distance > max_radius:
                break
            if capacity is not None and loads[warehouse_idx] + orders[point_idx] > capacity:
                continue
            assignment[point_idx] = warehouse_idx
            loads[warehouse_idx] += orders[point_idx]
            break
        if assignment[point_idx] == -1:
            unserved.append(int(point_idx))
    return assignment, unserved


def evaluate_network(
    points: np.ndarray,
    orders: np.ndarray,
    warehouses: np.ndarray,
    assignment: np.ndarray,
    cost_per_km_per_order: float,
) -> CostBreakdown:
    """Calculate delivery cost, average distance, and coverage for an assignment."""
    distances = np.full(len(points), np.nan, dtype=float)
    served = assignment >= 0
    if served.any():
        distances[served] = haversine_km(
            points[served, 0],
            points[served, 1],
            warehouses[assignment[served], 0],
            warehouses[assignment[served], 1],
        )

    served_orders = float(orders[served].sum())
    total_orders = float(orders.sum())
    weighted_distance = float(np.nansum(distances * orders))
    return CostBreakdown(
        weighted_distance_km=weighted_distance,
        delivery_cost=weighted_distance * cost_per_km_per_order,
        average_distance_km=weighted_distance / served_orders if served_orders else 0.0,
        served_orders=served_orders,
        total_orders=total_orders,
        coverage_percent=100 * served_orders / total_orders if total_orders else 0.0,
        distances_km=distances,
    )


def _to_local_xy(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Project nearby latitude/longitude values to local kilometres (north, east)."""
    north = np.radians(points[:, 0] - origin[0]) * EARTH_RADIUS_KM
    east = np.radians(points[:, 1] - origin[1]) * EARTH_RADIUS_KM * np.cos(np.radians(origin[0]))
    return np.column_stack((north, east))


def _from_local_xy(xy: np.ndarray, origin: np.ndarray) -> np.ndarray:
    lat = origin[0] + np.degrees(xy[:, 0] / EARTH_RADIUS_KM)
    lon = origin[1] + np.degrees(xy[:, 1] / (EARTH_RADIUS_KM * np.cos(np.radians(origin[0]))))
    return np.column_stack((lat, lon))


def _weighted_geometric_median(points: np.ndarray, weights: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Weiszfeld's algorithm in a local kilometre coordinate system."""
    centre = start.copy()
    for _ in range(100):
        distances = np.linalg.norm(points - centre, axis=1)
        if (distances < 1e-9).any():
            return points[distances.argmin()].copy()
        inverse_distance_weights = weights / distances
        updated = (points * inverse_distance_weights[:, None]).sum(axis=0) / inverse_distance_weights.sum()
        if np.max(np.abs(updated - centre)) < 1e-6:
            return updated
        centre = updated
    return centre


def _kmeans_plus_plus_start(points: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Demand-weighted, spread-out starting locations."""
    first = rng.choice(len(points), p=weights / weights.sum())
    centres = [points[first]]
    for _ in range(1, k):
        squared_distance = ((points[:, None, :] - np.asarray(centres)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        probabilities = squared_distance * weights
        if probabilities.sum() <= 0:
            probabilities = np.full(len(points), 1 / len(points))
        else:
            probabilities = probabilities / probabilities.sum()
        centres.append(points[rng.choice(len(points), p=probabilities)])
    return np.asarray(centres, dtype=float)


def optimize_warehouses(
    points: np.ndarray,
    orders: np.ndarray,
    warehouse_count: int,
    restarts: int = 18,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return optimized warehouse coordinates and the nearest assignment.

    The local projection keeps the optimization stable in kilometres, while the
    final assignment still uses great-circle (Haversine) distances.
    """
    points = np.asarray(points, dtype=float)
    orders = np.asarray(orders, dtype=float)
    if len(points) == 0 or len(points) != len(orders):
        raise ValueError("Points and orders must be non-empty arrays of the same length.")
    if (orders <= 0).any():
        raise ValueError("Every neighbourhood must have a positive order count.")

    k = max(1, min(int(warehouse_count), len(points)))
    origin = points.mean(axis=0)
    local_points = _to_local_xy(points, origin)
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for _ in range(restarts):
        centres = _kmeans_plus_plus_start(local_points, orders, k, rng)
        for _ in range(100):
            membership = ((local_points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            updated = centres.copy()
            for warehouse_idx in range(k):
                in_cluster = membership == warehouse_idx
                if in_cluster.any():
                    updated[warehouse_idx] = _weighted_geometric_median(
                        local_points[in_cluster], orders[in_cluster], centres[warehouse_idx]
                    )
            if np.allclose(updated, centres, atol=1e-6):
                centres = updated
                break
            centres = updated

        warehouses = _from_local_xy(centres, origin)
        assignment = assign_nearest(points, warehouses)
        score = evaluate_network(points, orders, warehouses, assignment, 1.0).weighted_distance_km
        if best is None or score < best[0]:
            best = (score, warehouses, assignment)

    assert best is not None
    return best[1], best[2]


def suggested_existing_sites(points: np.ndarray, orders: np.ndarray, warehouse_count: int) -> np.ndarray:
    """Choose spread-out demand points for a neutral starting comparison scenario."""
    # A deterministic nearest-neighbour baseline is fairer and easier to explain than random sites.
    first = int(np.argmax(orders))
    chosen = [first]
    while len(chosen) < min(warehouse_count, len(points)):
        distances = distance_matrix(points, points[chosen]).min(axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances * (orders / orders.max()))))
    return np.asarray(chosen, dtype=int)


def tradeoff_curve(
    points: np.ndarray,
    orders: np.ndarray,
    max_warehouses: int,
    cost_per_km_per_order: float,
    daily_facility_cost: float,
) -> pd.DataFrame:
    """Compare delivery and daily facility costs as the network size changes."""
    rows = []
    for warehouse_count in range(1, max_warehouses + 1):
        warehouses, assignment = optimize_warehouses(points, orders, warehouse_count, restarts=8)
        delivery = evaluate_network(points, orders, warehouses, assignment, cost_per_km_per_order).delivery_cost
        facility = warehouse_count * daily_facility_cost
        rows.append(
            {
                "warehouses": warehouse_count,
                "delivery_cost": delivery,
                "facility_cost": facility,
                "total_daily_cost": delivery + facility,
            }
        )
    return pd.DataFrame(rows)
