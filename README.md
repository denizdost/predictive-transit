<div align="center">

🚇 Terminal Sivas

Next-Gen Predictive Transit & Simulation Engine

Stop guessing. Start knowing. Terminal Sivas is a full-scale transit prediction engine powered by Machine Learning, real-time environmental data, and deterministic spatial simulations.

Explore Features • Installation • Tech Stack

</div>

⚡ Overview

Traditional transit apps rely on static schedules. Terminal Sivas relies on data. By combining historical transit logs, live weather conditions, and traffic density, our backend ML models dynamically predict exactly when your bus will arrive and how crowded it will be.

Coupled with a gorgeous, corporate-grade "Metro" UI, the frontend doesn't just show you numbers—it visualizes the entire city network using an advanced simulation engine mapped to real-world roads via OSRM.

:sparkles: Key Features

🔮 ML-Powered Predictions: Arrival delays and crowd levels (passenger flow) are dynamically calculated using predictive models trained on weather, traffic, and time-of-day data.

🗺️ Real-Road OSRM Routing: Buses don't fly. Our interactive map engine snaps transit routes to the actual street network of Sivas for pixel-perfect geospatial simulations.

⏳ The "Time Machine": Drag the interactive timeline slider to simulate the exact position of the entire transit fleet up to 60 minutes into the past or future.

🔄 Smart Journey Planner: Click anywhere on the map to automatically calculate optimal multi-leg walking/transit routes, including transfer penalties and automated routing.

🎨 Metro-Grade UI: A sleek, highly accessible, responsive frontend inspired by world-class metro networks, featuring Dark Mode and real-time condition overlays.

🛡️ Offline/Demo Fallback: If the API goes down, the frontend seamlessly shifts into a localized, deterministic mock-generation mode so your demos never break.

:gear: Under the Hood

The architecture is split into a robust Python data pipeline and a lightning-fast static frontend.

Backend & Data Pipeline:

API: FastAPI (api/main.py)

Machine Learning: Scikit-Learn / Pandas (model/train.py)

Data Sources: Weather observations, stop arrivals, passenger flows, and bus trips (data/*.csv)

Frontend & Visuals:

Core: Vanilla HTML5 / JavaScript (Zero-build pipeline)

Styling: Tailwind CSS (Custom Metro Red/Blue theme)

Maps: Leaflet.js with Carto basemaps

Routing: Open Source Routing Machine (OSRM) API

:rocket: Getting Started

We've made bootstrapping the entire stack as painless as possible.

Prerequisites

Python 3.9+

Pip

1. Clone the Repository

git clone [https://github.com/yourusername/terminal-sivas.git](https://github.com/yourusername/terminal-sivas.git)
cd terminal-sivas


2. Run the Setup Script

We've included a handy shell script that sets up your virtual environment, installs dependencies, trains the initial ML models, and boots up the FastAPI server.

chmod +x setup_and_run.sh
./setup_and_run.sh


(Alternatively, you can manually run pip install -r requirements.txt, execute python model/train.py, and then start the API with uvicorn api.main:app --reload)

3. Launch the Frontend

The frontend requires no build steps! Simply open frontend/index.html in your favorite modern web browser, or serve it via a simple HTTP server:

python -m http.server 8080 --directory frontend


Navigate to http://localhost:8080 and start exploring the network!

:file_folder: Project Structure

terminal-sivas/
├── api/
│   └── main.py                 # FastAPI endpoints for predictions
├── data/
│   ├── bus_stops.csv           # Geospatial stop data
│   ├── bus_trips.csv           # Historical route logs
│   ├── passenger_flow.csv      # Crowd density datasets
│   ├── stop_arrivals.csv       # Arrival timing data
│   └── weather_observations.csv# Meteorological factors
├── frontend/
│   └── index.html              # The complete SPA commuter portal
├── model/
│   └── train.py                # ML model training scripts
├── requirements.txt            # Python dependencies
└── setup_and_run.sh            # One-click bootstrap script


:handshake: Contributing

<i>Built for the commuters of tomorrow.</i>




<b>Terminal Sivas</b>
</div>
