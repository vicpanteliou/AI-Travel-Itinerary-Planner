import pytest
from unittest.mock import Mock, patch, MagicMock
from src.mini2.app import (
    TravelAgent,
    get_weather_forecast,
    GraphState
)


def test_parse_request_node():
    """Test that parse_request_node correctly extracts city, days, and interests."""
    # Mock the LLM response
    mock_response = Mock()
    mock_response.content = "City: Paris\nDays: 3\nInterests: museums, cafes"

    agent = TravelAgent()
    agent.llm = Mock()
    agent.llm.invoke.return_value = mock_response

    state = {"user_request": "I want to go to Paris for 3 days to visit museums and cafes"}
    result = agent.parse_request_node(state)

    assert result["city"] == "Paris"
    assert result["days"] == 3
    assert result["interests"] == "museums, cafes"


def test_decide_activity_type_node():
    """Test that decide_activity_type_node returns valid activity preference."""
    mock_response = Mock()
    mock_response.content = "INDOOR"

    agent = TravelAgent()
    agent.llm = Mock()
    agent.llm.invoke.return_value = mock_response

    state = {
        "city": "London",
        "days": 3,
        "weather_data": "Day 1: Rain, 15°C | Day 2: Rain, 14°C"
    }
    result = agent.decide_activity_type_node(state)

    assert result["activity_preference"] in ["INDOOR", "OUTDOOR", "BOTH"]
    assert result["activity_preference"] == "INDOOR"


def test_check_activity_quality():
    """Test that check_activity_quality returns correct routing decision."""
    mock_response = Mock()
    mock_response.content = "SUFFICIENT"

    agent = TravelAgent()
    agent.llm = Mock()
    agent.llm.invoke.return_value = mock_response

    state = {
        "city": "Rome",
        "activities": ["Museums: Visit the Colosseum and Vatican Museums"],
        "search_iterations": 0
    }
    result = agent.check_activity_quality(state)

    assert result in ["activities", "generate"]
    assert result == "generate"


def test_get_weather_forecast_parsing():
    """Test that get_weather_forecast correctly parses API data."""
    # Mock API response - API returns 3-hour forecasts, so 8 items per day
    # We need items at indices 0, 8, 16 for days 1, 2, 3
    mock_list = [None] * 24  # Create enough items
    mock_list[0] = {'weather': [{'main': 'Clear'}], 'main': {'temp': 25.5}}
    mock_list[8] = {'weather': [{'main': 'Rain'}], 'main': {'temp': 18.2}}
    mock_list[16] = {'weather': [{'main': 'Clouds'}], 'main': {'temp': 22.0}}

    # Fill in the None values with dummy data
    for i in range(24):
        if mock_list[i] is None:
            mock_list[i] = {'weather': [{'main': 'Clear'}], 'main': {'temp': 20.0}}

    mock_api_response = {'list': mock_list}

    mock_response = Mock()
    mock_response.json.return_value = mock_api_response
    mock_response.raise_for_status = Mock()

    with patch('src.mini2.app.requests.get') as mock_get, \
         patch.dict('os.environ', {'OPENWEATHERMAP_API_KEY': 'test_key'}):
        mock_get.return_value = mock_response

        result = get_weather_forecast.invoke({"city": "Athens", "days": 3})

        assert "Day 1: Clear, 25.5°C" in result
        assert "Day 2: Rain, 18.2°C" in result
        assert "Day 3: Clouds, 22.0°C" in result
