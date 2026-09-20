# GRIDPOINT - Warehouse Location Optimization Platform

GRIDPOINT helps an e-commerce company decide where to locate one or more warehouses. It maps demand, recommends locations that reduce order-weighted delivery distance, assigns neighbourhoods to warehouses, and compares the recommended plan with the current network.

## What it demonstrates

- Upload or edit neighbourhood data: `name`, `lat`, `lon`, `orders`
- Interactive map of demand, warehouse locations, and allocation lines
- Weighted k-median optimization: busy neighbourhoods influence the recommendation more
- Current-vs-recommended delivery-cost comparison
- Optional warehouse capacity and maximum service-radius rules
- Delivery-cost versus daily facility-cost trade-off
- CSV export of the recommended assignments

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Input data format

```csv
name,lat,lon,orders
Koramangala,12.9352,77.6245,320
Indiranagar,12.9784,77.6408,280
```

`orders` must be a positive daily order count. Coordinates must be latitude and longitude in decimal degrees.

## How the optimization works

1. GRIDPOINT begins with several well-spread warehouse candidates.
2. Each neighbourhood is assigned to the nearest warehouse.
3. Each warehouse moves toward the weighted geometric median of the neighbourhoods it serves. Higher-order neighbourhoods exert more pull.
4. The app repeats this from several starting layouts and retains the lowest weighted delivery distance.

The app uses Haversine (great-circle) distance. This is an excellent fast planning estimate, but real rollouts should incorporate roads, traffic, delivery time, property availability, and operating costs.

## Demo script

1. Open the Bengaluru demo data and show the demand bubbles.
2. Set three warehouses and select **Find best locations**.
3. Compare the maps and daily cost figures.
4. Enable a tight service radius or capacity to demonstrate real-world constraints and the coverage warning.
5. Open the trade-off tab and explain why the cheapest delivery plan may not be the cheapest total operating plan.

## Technology and AI disclosure

- Python, Streamlit, Pandas, NumPy, and PyDeck
- The mathematical approach is a weighted k-median clustering method using demand-weighted k-means++ initialization and Weiszfeld updates for geometric medians.
- An AI coding assistant was used to help draft and explain code. The team is responsible for understanding, testing, integrating, and presenting the work.

## Before submission

- Add your team members' names and repository link.
- Record a 2-3 minute demo video.
- Confirm the event rules about code written during the official hackathon window, library disclosure, and AI-assistant disclosure.
