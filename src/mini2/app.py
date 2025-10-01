import requests
import asyncio
from langchain.tools import tool
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import Dict

load_dotenv()

class GraphState(TypedDict):
    user_request: str      # Input: "I want to go to Athens for 5 days"
    city: str              # Parsed: "Athens"
    days: int              # Parsed: 5
    interests: str         # Parsed: "food"
    weather_data: str     # From weather tool
    activity_preference: str  # Decision: "INDOOR", "OUTDOOR", or "BOTH"
    activities: List[str]  # From activities tool
    final_itinerary: str   # From LLM generation
    search_iterations: int  # Track number of search loops


@tool
def get_weather_forecast(city: str, days: int) -> str:
    """Get weather forecast for a city using OpenWeatherMap API."""
    
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Error: OPENWEATHERMAP_API_KEY not set"
    
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        forecasts = []
        max_days = min(days, 5)  # API only provides 5 days

        for i in range(max_days):
            # API returns 3-hour intervals (8 data points per day), so i * 8 jumps to same time each day
            if i * 8 < len(data['list']):
                forecast = data['list'][i * 8]
                weather = forecast['weather'][0]['main']
                temp = forecast['main']['temp']
                forecasts.append(f"Day {i+1}: {weather}, {temp}°C")
            else:
                forecasts.append(f"Day {i+1}: Could not get forecast")
        
        return " | ".join(forecasts)
    
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
async def find_points_of_interest(city: str, category: str) -> str:
    """Find attractions in a city by category using DuckDuckGo search."""

    search = DuckDuckGoSearchRun()

    query = f"best {category} in {city}"

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, search.run, query)
        return results
    except Exception as e:
        return f"Error searching for {category}: {str(e)}"


class TravelAgent:
    """AI-powered travel assistant that generates personalized itineraries."""

    def __init__(self, max_search_iterations: int = 2):
        """Initialize the TravelAgent with LLM and workflow configuration."""
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
            temperature=0
        )
        self.max_search_iterations = max_search_iterations
        self.app = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        workflow.add_node("parse", self.parse_request_node)
        workflow.add_node("weather", self.weather_node)
        workflow.add_node("decide", self.decide_activity_type_node)
        workflow.add_node("activities", self.activities_node)
        workflow.add_node("generate", self.generate_itinerary_node)

        workflow.set_entry_point("parse")

        # Flow: parse → weather → decide → activities → (check quality) → loop or generate
        workflow.add_edge("parse", "weather")
        workflow.add_edge("weather", "decide")
        workflow.add_edge("decide", "activities")

        # Activities has conditional routing (loop or continue)
        workflow.add_conditional_edges(
            "activities",
            self.check_activity_quality,
            {
                "activities": "activities",  # Loop back for refinement
                "generate": "generate"       # Continue to final generation
            }
        )

        workflow.add_edge("generate", END)

        return workflow.compile()

    def parse_request_node(self, state: GraphState) -> Dict:
        """Extract city, days, and interests from user request."""
        print("Parsing request...")
        prompt = f"""Extract the following from this travel request: "{state['user_request']}"
        Return ONLY in this format:
        City: <city name>
        Days: <number>
        Interests: <comma-separated interests>"""

        response = self.llm.invoke(prompt)
        print(f"LLM response: {response.content}")
        # Parse the LLM response
        lines = [line.strip() for line in response.content.split('\n') if line.strip()]

        parsed_data = {}
        for line in lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                parsed_data[key.lower()] = value

        result = {
            "city": parsed_data.get("city", "Unknown"),
            "days": int(parsed_data.get("days", 5)),
            "interests": parsed_data.get("interests", "sightseeing")
        }
        print(f"Parsed: {result}")
        return result

    def weather_node(self, state: GraphState) -> Dict:
        """Fetch weather data."""
        print(f"Fetching weather for {state['city']}...")
        weather = get_weather_forecast.invoke({
            "city": state["city"],
            "days": state["days"]
        })
        print(f"Weather: {weather}")
        return {"weather_data": weather}

    def decide_activity_type_node(self, state: GraphState) -> Dict:
        """Decide indoor/outdoor activities based on weather."""
        print("Deciding activity type based on weather...")
        prompt = f"""Based on this weather forecast: {state['weather_data']}

        Should we prioritize INDOOR, OUTDOOR, or BOTH activities?
        Consider rain, extreme temperatures, etc.
        Answer with only one word: INDOOR, OUTDOOR, or BOTH"""

        response = self.llm.invoke(prompt)
        decision = response.content.strip().upper()

        # Ensure valid response
        if decision not in ["INDOOR", "OUTDOOR", "BOTH"]:
            decision = "BOTH"

        print(f"Activity preference: {decision}")
        return {"activity_preference": decision}

    async def activities_node(self, state: GraphState) -> Dict:
        """Fetch activities based on interests and weather preference."""
        preference = state.get("activity_preference", "BOTH")
        current_iteration = state.get("search_iterations", 0) + 1

        print(f"Fetching activities (iteration {current_iteration}) for {state['interests']} (preference: {preference})...")
        interests_list = [i.strip() for i in state["interests"].split(',')]

        # Modify search based on weather preference
        modifier = ""
        if preference == "INDOOR":
            modifier = "indoor "
        elif preference == "OUTDOOR":
            modifier = "outdoor "

        search_tasks = []
        for interest in interests_list:
            search_category = f"{modifier}{interest}" if modifier else interest
            print(f"Searching for {search_category}...")
            task = find_points_of_interest.ainvoke({
                "city": state["city"],
                "category": search_category
            })
            search_tasks.append((interest, task))

        # Execute all searches in parallel
        results = await asyncio.gather(*[task for _, task in search_tasks])

        # Combine interests with results
        activities = [f"{interest.title()}: {result}" for (interest, _), result in zip(search_tasks, results)]

        print("Activities done")
        return {"activities": activities, "search_iterations": current_iteration}

    def check_activity_quality(self, state: GraphState) -> str:
        """Check if activities are specific enough."""
        current_iteration = state.get("search_iterations", 0)
        max_iterations = self.max_search_iterations

        print("\n=== CHECKING ACTIVITY QUALITY ===")
        print(f"Current iteration: {current_iteration}/{max_iterations}")
        print(f"Activities to evaluate: {state['activities']}")

        if current_iteration >= max_iterations:
            print(f"→ Max iterations ({max_iterations}) reached, proceeding with current results...\n")
            return "generate"

        prompt = f"""Review these activities for {state['city']}:
        {' '.join(state['activities'])}

        Do these include SOME specific venue/place names? Be lenient - if you can identify at least a few actual names (like restaurants, bars, temples, trails), answer SUFFICIENT.
        If it's all generic descriptions with NO specific names, answer NEED_MORE.
        Answer only: SUFFICIENT or NEED_MORE"""

        response = self.llm.invoke(prompt)
        decision = response.content.strip().upper()

        print(f"LLM Response: '{response.content}'")
        print(f"Decision: {decision}")

        if "NEED_MORE" in decision:
            print("→ Activities need refinement, searching again...\n")
            return "activities"  # Loop back

        print("→ Activities are sufficient, proceeding to itinerary generation...\n")
        return "generate"  # Move forward

    def generate_itinerary_node(self, state: GraphState) -> Dict:
        """Generate final itinerary using LLM."""
        print("Generating itinerary...")

        prompt = f"""Create a {state['days']}-day travel itinerary for {state['city']}.

        Weather: {state['weather_data']}
        Activities: {' '.join(state['activities'])}

        Create a detailed, practical itinerary that:
        - Balances activities with rest time
        - Includes specific timing and logistics
        - Suggests backup options for bad weather
        - Uses actual place names from the activities data provided


        Return ONLY in this format:
        # {state['days']}-Day Itinerary for {state['city']}

        ## Day 1: [Weather]
        - **Morning**: [Activity with specific place name]
        - **Lunch**: [Restaurant name and cuisine type]
        - **Afternoon**: [Activity with specific place name]
        - **Evening**: [Activity/Dinner with specific place name]
        - **Weather**: [Expected conditions and recommendations]
        - **Backup Plan**: [Alternative if weather is bad]

        ## Day 2: [Weather]
        [Same structure...]

        [Repeat for all {state['days']} days]

        """

        response = self.llm.invoke(prompt)
        print("Itinerary complete!")
        return {"final_itinerary": response.content}

    async def plan_trip(self, user_request: str) -> str:
        """Main method to generate a travel itinerary from a user request."""
        print("Starting workflow...")
        result = await self.app.ainvoke({"user_request": user_request})
        return result.get("final_itinerary", "No itinerary generated")


if __name__ == "__main__":
    agent = TravelAgent(max_search_iterations=2)

    user_request = "I want to go to Kyoto for 5 days. i want recommendations for restaurants, bars and attractions. Also maybe a day for hiking"
    itinerary = asyncio.run(agent.plan_trip(user_request))

    print("\n=== FINAL RESULT ===")
    print(itinerary)

