# filepath: ET_prediction_project/src/ET_model.py

import torch
import torch.nn as nn
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.components.sensor import SensorEntity
from homeassistant.helpers.event import async_track_time_change
from datetime import datetime, time
import asyncio
from .irrigation_scheduler import IrrigationScheduler

DOMAIN = "et_irrigation"
VALVE_ID = "CAE5-982E-004B-1200"
OWM_API_KEY = "1ddb5488cc1d4b7ad483113a50514bc0"

async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the ET Irrigation component."""
    hass.data[DOMAIN] = {
        'scheduler': IrrigationScheduler(VALVE_ID, OWM_API_KEY)
    }
    return True

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the ET Prediction and irrigation control entities."""
    model = ETModel(input_size=3, hidden_size=10, output_size=1)
    model.eval()
    
    # Add both prediction and irrigation control entities
    async_add_entities([
        ETPredictionSensor(model),
        IrrigationControl(hass.data[DOMAIN]['scheduler'])
    ])

    # Setup scheduled irrigation checks
    async def check_irrigation(now):
        et_value = model.predict(torch.tensor([[1.0, 2.0, 3.0]])).item()
        hass.data[DOMAIN]['scheduler'].check_schedule(et_value)

    # Run at 7:00 AM and 7:00 PM
    async_track_time_change(hass, check_irrigation, hour=[7, 19], minute=0, second=0)

class ETModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ETModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

class ETPredictionSensor(SensorEntity):
    def __init__(self, model):
        self._model = model
        self._state = None
        self._attr_name = "ET Prediction"
        self._attr_unique_id = "et_prediction_sensor"
        self._attr_device_class = None
        self._attr_native_unit_of_measurement = "mm"

    @property
    def native_value(self):
        return self._state

    async def async_update(self):
        """Update sensor with new ET prediction."""
        input_data = torch.tensor([[1.0, 2.0, 3.0]])  # Replace with real data
        prediction = self._model.predict(input_data)
        self._state = round(prediction.item(), 2)

class IrrigationControl(Entity):
    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._state = "idle"
        self._attr_name = "Irrigation Control"
        self._attr_unique_id = "irrigation_control"
        self._attr_icon = "mdi:sprinkler"

    @property
    def state(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return {
            "valve_id": self._scheduler.valve_id,
            "next_scheduled": "07:00 AM" if datetime.now().hour < 12 else "07:00 PM"
        }

    async def async_update(self):
        """Update irrigation control status."""
        # This would be called periodically to update state
        pass

    async def async_turn_on(self):
        """Manual override to turn on irrigation."""
        if self._scheduler.control_valve(True):
            self._state = "active"
            self.async_write_ha_state()

    async def async_turn_off(self):
        """Manual override to turn off irrigation."""
        if self._scheduler.control_valve(False):
            self._state = "idle"
            self.async_write_ha_state()
