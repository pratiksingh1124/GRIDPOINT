"""Decision-support summaries for the GRIDPOINT interface.

The optimization engine answers *where* warehouses should be located.  This
module translates a proposed network into the questions an operations lead
needs to answer before approving it: Is coverage complete? Which service zone
is most stretched? Where should we validate assumptions first?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer import CostBreakdown


def operation_pulse(
    data: pd.DataFrame,
    warehouses: np.ndarray,
    assignment: np.ndarray,
    result: CostBreakdown,
    capacity: float | None,
    service_radius: float | None,
) -> tuple[dict[str, float | int | str | None], pd.DataFrame, list[str]]:
    """Summarise workload, coverage and the highest-priority validation work."""
    demand = data["orders"].to_numpy(dtype=float)
    served = assignment >= 0
    unserved_orders = int(demand[~served].sum())
    top_count = min(3, len(data))
    top_three_share = 100 * float(data.nlargest(top_count, "orders")["orders"].sum()) / float(demand.sum())

    rows: list[dict[str, float | int | str]] = []
    for warehouse_idx in range(len(warehouses)):
        in_zone = assignment == warehouse_idx
        zone_orders = int(demand[in_zone].sum())
        zone_distances = result.distances_km[in_zone]
        longest_delivery = float(np.nanmax(zone_distances)) if len(zone_distances) else 0.0
        utilisation = 100 * zone_orders / capacity if capacity else None
        if utilisation is not None and utilisation >= 90:
            status = "Capacity risk"
        elif service_radius is not None and longest_delivery >= 0.9 * service_radius:
            status = "Radius watch"
        elif zone_orders == 0:
            status = "No demand assigned"
        else:
            status = "On track"
        rows.append(
            {
                "service zone": f"Warehouse {warehouse_idx + 1}",
                "orders/day": zone_orders,
                "neighbourhoods": int(in_zone.sum()),
                "longest delivery (km)": round(longest_delivery, 1),
                "capacity used": f"{utilisation:.0f}%" if utilisation is not None else "Not set",
                "status": status,
            }
        )
    zone_summary = pd.DataFrame(rows)

    if served.any():
        served_indices = np.flatnonzero(served)
        farthest_idx = int(served_indices[np.nanargmax(result.distances_km[served])])
        farthest_name = str(data.iloc[farthest_idx]["name"])
        farthest_distance = float(result.distances_km[farthest_idx])
    else:
        farthest_name, farthest_distance = "No neighbourhood served", 0.0

    actions: list[str] = []
    if not served.all():
        unserved_names = ", ".join(data.loc[~served, "name"].tolist())
        actions.append(f"Restore coverage for {unserved_orders:,} daily orders in: {unserved_names}.")
    else:
        actions.append("Coverage is complete under the rules selected for this scenario. Validate road travel times before committing to sites.")

    if capacity:
        busiest = zone_summary.loc[zone_summary["orders/day"].idxmax()]
        busiest_utilisation = 100 * float(busiest["orders/day"]) / capacity
        if busiest_utilisation >= 90:
            actions.append(f"Protect capacity at {busiest['service zone']}: it is planned at {busiest_utilisation:.0f}% utilisation.")
        elif busiest_utilisation <= 60:
            actions.append(f"{busiest['service zone']} is the busiest at {busiest_utilisation:.0f}% utilisation; test peak-day demand before sizing the facility.")

    if service_radius and farthest_distance >= 0.9 * service_radius:
        actions.append(f"Validate {farthest_name} first: its estimated {farthest_distance:.1f} km delivery is close to the {service_radius:.0f} km service limit.")
    if top_three_share >= 45:
        actions.append(f"Demand is concentrated: the top {top_count} neighbourhoods account for {top_three_share:.0f}% of daily orders. Protect service levels in those zones first.")

    pulse: dict[str, float | int | str | None] = {
        "coverage_percent": result.coverage_percent,
        "unserved_orders": unserved_orders,
        "top_three_share": top_three_share,
        "farthest_name": farthest_name,
        "farthest_distance": farthest_distance,
        "max_utilisation": max(
            (100 * float(row["orders/day"]) / capacity for row in rows),
            default=None,
        ) if capacity else None,
    }
    return pulse, zone_summary, actions


def decision_brief_markdown(
    warehouse_count: int,
    current_result: CostBreakdown,
    optimized_result: CostBreakdown,
    pulse: dict[str, float | int | str | None],
    actions: list[str],
) -> str:
    """Create a portable, plain-language scenario brief for project stakeholders."""
    saving = current_result.delivery_cost - optimized_result.delivery_cost
    saving_label = "reduces" if saving >= 0 else "increases"
    return "\n".join(
        [
            "# GRIDPOINT decision brief",
            "",
            "## Recommended scenario",
            f"Plan for **{warehouse_count} warehouse(s)**. This scenario {saving_label} estimated delivery cost by **₹{abs(saving):,.0f} per day** versus the selected current-site baseline.",
            f"It covers **{optimized_result.coverage_percent:.0f}%** of listed daily demand with an average straight-line delivery distance of **{optimized_result.average_distance_km:.2f} km**.",
            "",
            "## Operating signals",
            f"- Top-three demand concentration: **{float(pulse['top_three_share']):.0f}%** of daily orders",
            f"- Farthest served area: **{pulse['farthest_name']}** at **{float(pulse['farthest_distance']):.1f} km**",
            f"- Unserved daily orders: **{int(pulse['unserved_orders']):,}**",
            "",
            "## Validate before commitment",
            *[f"- {action}" for action in actions],
            "",
            "*GRIDPOINT uses straight-line distance for fast scenario planning. Validate routes, traffic, property availability and staffing costs before making a final site decision.*",
        ]
    )
