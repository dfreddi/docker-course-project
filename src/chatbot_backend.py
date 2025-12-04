from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import uvicorn
import argparse


LLM_URL = "http://model_container:8000/v1/"
WEATHER_FORCAST_URL = "https://api.open-meteo.com/v1/forecast"

parser = argparse.ArgumentParser(description='Run Chatbot FastAPI server')
parser.add_argument('--llm-name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Name of the LLM to use')
args = parser.parse_args()


class Chatbot:
    def __init__(self, llm_model_name):
        self.llm = OpenAI(base_url=LLM_URL, api_key="EMPTY")
        self.llm_model_name = llm_model_name

    def get_weather_info(self) -> pd.DataFrame:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        params = {
            "latitude": 44.1391,
            "longitude": 12.2431,
            "hourly": ["temperature_2m", "precipitation_probability", "rain", "showers", "snowfall"],
            "timezone": "auto",
            "past_days": 1
        }
        responses = openmeteo.weather_api(WEATHER_FORCAST_URL, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation_probability = hourly.Variables(1).ValuesAsNumpy()
        hourly_rain = hourly.Variables(2).ValuesAsNumpy()
        hourly_showers = hourly.Variables(3).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["precipitation_probability"] = hourly_precipitation_probability
        hourly_data["rain"] = hourly_rain
        hourly_data["showers"] = hourly_showers
        hourly_data["snowfall"] = hourly_snowfall

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        return hourly_dataframe

    def get_formatted_weather_data(self) -> str:
        weather_info = self.get_weather_info()
        df = weather_info.iloc[::2].copy()  # only get info for every 2 hours
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.tz_convert('Europe/Rome')
        df['date_only'] = df['date'].dt.date
        grouped = df.groupby('date_only')  # group rows by day

        # Format the groups into LLM digestible data
        formatted_weather_info = ""
        for date, group in grouped:
            group = group.drop(columns=['date_only'])
            formatted_weather_info += f"## {date.strftime('%A, %B %d, %Y')}\n"
            for _, row in group.iterrows():
                formatted_row = f"Time-{row['date'].strftime('%H:%M')};"
                formatted_row += f"Temperature-{round(row['temperature_2m'], 1)}°C;"
                formatted_row += f"Precipitation Probability-{row['precipitation_probability']}%;"
                formatted_row += f"Rain-{round(row['rain'], 1)}mm;"
                formatted_row += f"Showers-{round(row['showers'], 1)}mm;"
                formatted_row += f"Snowfall-{round(row['snowfall'], 1)}mm"
                formatted_weather_info += formatted_row + "\n"
            formatted_weather_info += "\n"
        return formatted_weather_info

    def get_current_time_string(self) -> str:
        return pd.Timestamp.now(tz='Europe/Rome').strftime('%A, %B %d, %Y')

    async def answer_message(self, question, chat_history):
        # Prepare the system prompt and messages
        system_prompt = self.get_system_prompt()
        messages = self.prepare_messages(system_prompt, chat_history, question)

        # Call the LLM and yield the response chunks
        async for chunk in self.llm.chat.completions.create(
            model=self.llm_model_name,
            messages=messages,
            temperature=0.8,
            stream=True
        ):
            yield chunk

    def get_system_prompt(self):
        with open('./prompts/system_prompt_template.txt', 'r') as file:
            system_prompt_template = file.read()
        return system_prompt_template.format(
            current_time=self.get_current_time_string(),
            weather_forecast=self.get_formatted_weather_data()
        )

    def prepare_messages(self, system_prompt, chat_history, question):
        messages = [{"role": "system", "content": system_prompt}]
        for user_message, bot_response in zip(chat_history[::2], chat_history[1::2]):
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_response})
        messages.append({"role": "user", "content": question})
        return messages

# --- FastAPI ---
app = FastAPI()
chatbot = Chatbot(llm_model_name=args.llm_name)

class AnswerMessageRequest(BaseModel):
    history: list[str]
    last_message: str

@app.post("/answer_message")
def answer_message_endpoint(request: AnswerMessageRequest):
    answer = chatbot.answer_message(request.history, request.last_message)
    return {"answer": answer}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker compose healthcheck"""
    return {"status": "healthy"}

if __name__ == '__main__':
    uvicorn.run("chatbot_backend:app", host="0.0.0.0", port=8001, reload=True)