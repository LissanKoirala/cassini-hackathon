"""API Wrapper for NOAA Aurora 30 Minute Forecast."""

import logging
import threading
import time
from typing import Optional

import requests
from requests import RequestException

APIUrl = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

_LOGGER = logging.getLogger("aurora")


class AuroraForecast:
    forecast_dict = {}
    last_update_time = None
    lock = threading.Lock()

    def __init__(self, session: Optional[requests.Session] = None):
        """Initialize and test the session."""

        if session:
            self._session = session
        else:
            self._session = requests.Session()

    def close(self):
        self._session.close()

    def get_forecast_data(self, longitude: float, latitude: float):
        """Return a forecast probability for the given coordinates."""

        # acquire the lock to ensure that only one request is processed at a time
        with AuroraForecast.lock:
            longitude = (
                longitude % 360
            )  # Convert -180 to 180 to 360 longitudinal values

            # Check if the forecast data is older than 5 minutes
            if AuroraForecast.last_update_time is None or (
                AuroraForecast.last_update_time
                and time.monotonic() - AuroraForecast.last_update_time > 5 * 60
            ):
                AuroraForecast.forecast_dict = {}

                _LOGGER.debug("Fetching forecast data from NOAA")
                try:
                    resp = self._session.get(APIUrl, timeout=15)
                    resp.raise_for_status()
                    forecast_data = resp.json()

                    for forecast_item in forecast_data["coordinates"]:
                        if forecast_item[2] > 0:
                            AuroraForecast.forecast_dict[
                                forecast_item[0], forecast_item[1]
                            ] = forecast_item[2]

                    # update the time of the last update
                    AuroraForecast.last_update_time = time.monotonic()
                    _LOGGER.debug("Successfully fetched forecast data from NOAA")

                except RequestException as error:
                    _LOGGER.debug("Error fetching forecast from NOAA: %s", error)

            probability = AuroraForecast.forecast_dict.get(
                (round(longitude), round(latitude)), 0
            )
            _LOGGER.debug(
                "Forecast probability: %s at (long, lat) = (%s, %s)",
                probability,
                round(longitude),
                round(latitude),
            )
            return probability



if __name__ == "__main__":
    aurora = AuroraForecast()
    probability = aurora.get_forecast_data(79.119764, 97.415467)
    print(f"Aurora probability at (24.94, 60.17): {probability}%")
    aurora.close()
