# GRIDPOINT

> A decision-support application for planning warehouse locations from neighbourhood-level demand.

GRIDPOINT helps operations teams explore where to place one or more warehouses, understand delivery-distance trade-offs, and compare a proposed network with an existing one. It combines an interactive Streamlit interface with demand-weighted location optimization to turn a simple CSV of demand points into a decision-ready planning scenario.

> **Planning note:** GRIDPOINT is a prototype for scenario analysis. Its distance estimates are straight-line (great-circle) distances, not road-network travel times. Validate final site decisions with routing, traffic, property, capacity, labour, and operating-cost data.

## What it does

- Maps demand centres, warehouse locations, and warehouse-to-neighbourhood allocations
- Recommends warehouse locations that minimize demand-weighted delivery distance
- Compares an existing network against a recommended network
- Estimates delivery-cost and facility-cost trade-offs
- Applies optional daily-capacity and maximum-service-radius rules
- Highlights coverage gaps and operational risk indicators
- Supports editing in-app data or uploading a CSV
- Exports recommended neighbourhood assignments for further analysis

## How it works

GRIDPOINT uses a demand-weighted **k-median** approach:

1. It creates multiple well-distributed starting warehouse configurations using demand-weighted k-means++ initialization.
2. Each neighbourhood is assigned to its nearest warehouse.
3. Warehouse positions are refined toward the weighted geometric median of their assigned demand using Weiszfeld updates.
4. The best result across several optimization runs is retained.

Distances are calculated with the Haversine formula. The optimizer works in a local kilometre projection for stable location updates, then evaluates the final plan using great-circle distances.

## Quick start

### Prerequisites

- Python 3.10 or later
- pip

### Install and run

```bash
git clone https://github.com/pratiksingh1124/GRIDPOINT.git
cd GRIDPOINT
python -m pip install -r requirements.txt
streamlit run GRIDPOINT.py
```

Streamlit will display a local URL in the terminal. Open it in a browser to begin exploring scenarios.

## Using the app

1. Start with the included Bengaluru demo data or upload a CSV.
2. Review or edit demand data in the in-app table.
3. Choose the number of warehouses and enter a delivery cost per kilometre per order.
4. Optionally select current warehouse sites to establish a comparison baseline.
5. Add capacity or service-radius constraints if needed.
6. Select **Find best locations** and review the recommended network, economics, and operations views.
7. Export assignments for reporting or downstream analysis.

## Input data

Upload a CSV with the following required columns:

| Column | Description | Example |
| --- | --- | --- |
| `name` | Unique demand-centre or neighbourhood name | `Koramangala` |
| `lat` | Latitude in decimal degrees | `12.9352` |
| `lon` | Longitude in decimal degrees | `77.6245` |
| `orders` | Positive daily order volume | `320` |

Example:

```csv
name,lat,lon,orders
Koramangala,12.9352,77.6245,320
Indiranagar,12.9784,77.6408,280
```

Invalid coordinates, blank names, non-positive order volumes, and duplicate names are excluded during validation. Provide at least two valid demand centres.

## Project structure

| File | Purpose |
| --- | --- |
| `GRIDPOINT.py` | Streamlit application, visualizations, input handling, and user interface |
| `optimizer.py` | Location optimization, assignment logic, distance calculations, and cost evaluation |
| `briefing.py` | Decision briefing and operational-pulse helpers |
| `sample_data.csv` | Illustrative Bengaluru demand dataset |
| `requirements.txt` | Python dependencies |
| `DEPLOYMENT_CHECKLIST.md` | Deployment preparation checklist |

## Technology

- Python
- Streamlit
- Pandas and NumPy
- PyDeck
- Plotly

## Data and modelling considerations

The included Bengaluru dataset is illustrative and intended for demonstration. Before using GRIDPOINT for a real operating decision, replace it with verified demand data and assess factors outside this prototype, including:

- road-network travel distance and travel time
- time-of-day traffic and delivery windows
- property availability, rent, and fit-out cost
- warehouse capacity, labour, and fleet availability
- service-level commitments and delivery-zone constraints
- demand seasonality and projected growth

## Team

- **Pratik Singh** — Project lead: concept development, application integration, optimization workflow, deployment, documentation, and final presentation
- **Bicky Jaiswal** — Development and research support: testing scenarios, output review, implementation support, and presentation preparation
- **Sachin Sah** — Research and validation support: domain research, documentation, presentation creation, and final review
- **Sameer Ray** — Ideation, project feedback, and final review

## Acknowledgement

An AI coding assistant was used to help draft and explain portions of the code. The project team remains responsible for understanding, testing, integrating, and presenting the work.
