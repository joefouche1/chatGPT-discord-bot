import os
import discord
from datetime import datetime
import requests_cache
from utils.log import logger

# Create a cached session for requests
session = requests_cache.CachedSession('hogcache', expire_after=360)

async def get_weather(channel, city: str):
    """Get weather information for a city."""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    api_key = os.getenv('WEATHER_API_KEY')
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    
    try:
        response = session.get(complete_url)
        x = response.json()

        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"]
            current_temperature_fahrenheit = str(
                round((current_temperature - 273.15) * 9 / 5 + 32))
            current_pressure = y["pressure"]
            current_pressure_mmHg = str(round(current_pressure * 0.750062))
            current_humidity = y["humidity"]
            z = x["weather"]
            weather_description = z[0]["description"]
            
            embed = discord.Embed(
                title=f"Weather in {city}",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Description",
                            value=f"**{weather_description}**", inline=False)
            embed.add_field(
                name="Temperature(F)", value=f"**{current_temperature_fahrenheit}°F**", inline=False)
            embed.add_field(name="Humidity(%)",
                            value=f"**{current_humidity}%**", inline=False)
            embed.add_field(name="Atmospheric Pressure(mmHg)",
                            value=f"**{current_pressure_mmHg}mmHg**", inline=False)
            
            icon = z[0]["icon"]
            embed.set_thumbnail(
                url=f"https://openweathermap.org/img/w/{icon}.png")
            
            await channel.send(embed=embed)
        else:
            await channel.send("City not found. Try format: Pittsburgh OR Pittsburgh,PA,US OR London,UK")
            
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        await channel.send("Error getting weather information. Please try again later.") 