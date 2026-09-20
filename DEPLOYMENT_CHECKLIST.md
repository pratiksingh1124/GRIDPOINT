# GRIDPOINT deployment checklist

Before publishing a new Streamlit Cloud version:

1. Commit `GRIDPOINT.py`, `optimizer.py`, `briefing.py`, `sample_data.csv`, `requirements.txt`, and `.streamlit/config.toml` to the repository.
2. In the Streamlit Cloud app settings, set the main file path to `GRIDPOINT.py`.
3. Reboot the app once after the dependency update so `pydeck` is installed.
4. Open the live URL in a private browser window and confirm that **Bengaluru demo data** starts without a file error.
5. Select **Find best locations**, then check the recommended network, network economics, and Operations pulse tabs.

The app is a planning tool. Validate road travel times, traffic, property availability, staffing, and operating costs before selecting a real warehouse site.
