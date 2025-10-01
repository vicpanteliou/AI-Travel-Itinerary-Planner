# MiniProject2 - The AI Travel Itinerary Planner - Solution Code

AI travel agent that creates weather-aware itineraries using LangGraph.

## Workflow

```
parse → weather → decide → activities → [quality check] → generate
                                            ↓
                                      (loop if needed)
```

1. **parse**: Extract city, days, interests
2. **weather**: Fetch forecast (OpenWeatherMap)
3. **decide**: LLM determines INDOOR/OUTDOOR/BOTH based on weather
4. **activities**: Search POIs (DuckDuckGo), modified by weather preference
5. **quality check**: LLM validates specificity, loops if needed (max 2x)
6. **generate**: Create final itinerary

## Bonus Features
- Async parallel searches for activities categories
- Self-correcting quality loop

## Setup

Install dependencies:
```bash
poetry install
```

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_key
OPENAI_ENDPOINT=your_endpoint
OPENAI_MODEL_NAME=gpt-4-turbo
OPENWEATHERMAP_API_KEY=your_key
```

## Usage

```python
from src.mini2.app import TravelAgent

agent = TravelAgent(max_search_iterations=2)
itinerary = agent.plan_trip("I want to go to Warsaw for 5 days")
print(itinerary)
```

Or run directly:
```bash
poetry run python src/mini2/app.py
```

## Testing

```bash
poetry run pytest tests/test_app.py -v
```
