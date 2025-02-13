# Pirate Weather Chatbot

Simple toy project for the containerization course.

## Features

- Get current weather information
- Receive weather forecasts (up to 7 days in the future)
- Enjoy pirate-themed chatbot responses

## Containers structure

Three containers:
- Model container: runs the LLM and exposes generic openai-style restful apis
- Chatbot container: handles retrieval of weather information and calls the model container for chatbot responses
- Webapp container: runs the webapp

![Pirate Weather Chatbot Diagram](./images/pirate-weather-chatbot.png)

## Limitations
The bot only reports weather in Cesena for simplicity. Reporting information for different locations would require function calling, which would overcomplicate the project.