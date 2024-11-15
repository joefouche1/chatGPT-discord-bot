import os
import random
import discord
import requests_cache
from utils.log import logger

# Create a cached session for requests
session = requests_cache.CachedSession('hogcache', expire_after=360)

async def get_news(channel, query: str = None):
    """Get news articles based on query."""
    api_key = os.getenv('NEWS_API_KEY')
    
    # Build the query string
    q_params = query if query else "pig OR bacon OR hog"
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={q_params}&"
        "sortBy=relevancy&"
        "searchIn=title,description&"
        f"apiKey={api_key}"
    )
    
    logger.info(f"Getting news for query: {q_params}")
    
    try:
        response = session.get(url)
        data = response.json()
        
        if not data.get("articles"):
            await channel.send("No news articles found.")
            return
            
        art = None
        # If searching for pig-related news, try to find relevant articles
        if "pig" in q_params.lower():
            for _ in range(10):
                art = random.choice(data["articles"])
                adl = art['description'].lower()
                if 'pig' in adl or 'hog' in adl or 'bacon' in adl:
                    break
        else:
            art = random.choice(data["articles"])
            
        try:
            embed = discord.Embed(title=art['title'])
            embed.add_field(name='Description',
                            value=art['description'], inline=False)
            embed.add_field(name='Link', value=art['url'], inline=False)
            if art.get('urlToImage'):
                embed.set_thumbnail(url=art['urlToImage'])
            await channel.send(embed=embed)
        except KeyError:
            await channel.send("Data incomplete in article.")
            
    except Exception as e:
        logger.error(f"Error while retrieving news: {e}")
        await channel.send("Error while retrieving news. Please try again later.") 