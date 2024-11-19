import os
import json
import aiohttp
import requests_cache
from datetime import datetime, timedelta
from utils.log import logger
import discord
import asyncio

session = requests_cache.CachedSession('hogcache2', expire_after=360)

async def parse_sports_query(query: str, client):
    """Parse natural language query into structured data using LLM"""
    current_time = datetime.now()
    
    messages = [
        {"role": "system", "content": f"""
        You are an expert sports assistant. The current date and time is: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        
        Extract the following information from queries for sports scores:
        - Sport (baseball, basketball, football, etc.)
        - League (nba, wnba, ncaab, nfl, ncaaf, nhl, epl, mls etc.)
        
        For football queries:
        - If the team is an NFL team (Bengals, Steelers, Browns, Ravens, Bills, Patriots, etc.), set league to "nfl"
        - Only use "ncaaf" for explicit college football teams or when "college" is mentioned
        
        - Team name or identifier. If ambiguous, consider what sports are in season based on current date.
        - Date (convert relative timeframes to YYYY-MM-DD format based on current time provided above)
        
        For date parsing rules:
        - "today" = current date
        - "yesterday" = current date - 1 day
        - "last night" = if current time is before 11am, use yesterday, otherwise use today
        - "last weekend" = most recent Saturday
        - If no date specified, assume today
        
        NFL teams reference:
        AFC North: Bengals, Browns, Ravens, Steelers
        AFC South: Colts, Jaguars, Texans, Titans
        AFC East: Bills, Dolphins, Patriots, Jets
        AFC West: Broncos, Chiefs, Raiders, Chargers
        NFC North: Bears, Lions, Packers, Vikings
        NFC South: Falcons, Panthers, Saints, Buccaneers
        NFC East: Cowboys, Giants, Eagles, Commanders
        NFC West: Cardinals, Rams, 49ers, Seahawks
        
        Respond with only a JSON object, no other text. Format:
        {{"sport": "...", "league": "...", "team": "...", "date": "YYYY-MM-DD"}}
        Make all values lowercase.
        """},
        {"role": "user", "content": query}
    ]
    
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    response = completion.choices[0].message.content
    
    # Clean up the response in case there's any extra text
    json_str = response.strip()
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    if json_str.endswith('```'):
        json_str = json_str[:-3]
    
    parsed = json.loads(json_str)
    
    # Convert the date string to datetime object
    if "date" in parsed:
        parsed["date"] = datetime.strptime(parsed["date"], "%Y-%m-%d")
    else:
        parsed["date"] = current_time
    
    return parsed

def build_espn_request(parsed_query, live=False):
    """Build ESPN API request from parsed query"""
    if live:
        # Use live score API for SPORTSNOW
        LIVE_ENDPOINTS = {
            "nba" : "basketball_nba",
            "nfl" : "americanfootball_nfl",
            "nhl" : "icehockey_nhl",
            "mlb" : "baseball_mlb",
            "wnba" : "basketball_wnba",
            "ncaab" : "basketball_ncaab",
            "ncaaf" : "americanfootball_ncaaf"
        }
        return {
            "url": f"http://api.joefu.net/espn/{LIVE_ENDPOINTS[parsed_query['league'].lower()]}",
            "params": {}  # Live API doesn't need date params
        }
    
    # Original ESPN endpoints for historical data
    ENDPOINTS = {
        "mlb": "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
        "nba": "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        "ncaab": "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50",
        "ncaaf": "http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80",
        "nfl": "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        "nhl": "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
        "wnba": "http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
        "ncaaw": "http://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard",
        "epl": "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
        "mls": "http://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard"
    }
    
    details = {
        "url": ENDPOINTS[parsed_query["league"]],
        "params": {
            "dates": parsed_query["date"].strftime("%Y%m%d")
        }
    }

    return details

def clean_espn_response(scores_json):
    """Strip unnecessary metadata from ESPN API response"""
    if not scores_json:
        return {}
        
    cleaned = {}
    
    if "leagues" in scores_json:
        cleaned["leagues"] = [{
            "name": league.get("name"),
            "abbreviation": league.get("abbreviation")
        } for league in scores_json.get("leagues", [])]
    
    if "events" in scores_json:
        cleaned["events"] = []
        for event in scores_json.get("events", []):
            cleaned_event = {
                "name": event.get("name"),
                "date": event.get("date"),
                "status": event.get("status", {}).get("type", {}).get("state"),
                "competitions": []
            }
            
            for competition in event.get("competitions", []):
                cleaned_competition = {
                    "competitors": [{
                        "team": {
                            "name": comp.get("team", {}).get("name"),
                            "abbreviation": comp.get("team", {}).get("abbreviation")
                        },
                        "score": comp.get("score"),
                        "winner": comp.get("winner")
                    } for comp in competition.get("competitors", [])]
                }
                cleaned_event["competitions"].append(cleaned_competition)
                
            cleaned["events"].append(cleaned_event)
    
    return cleaned

async def format_sports_response(result: str, user_name: str = None, query: str = None) -> discord.Embed:
    """Format sports response into a consistent embed style."""
    # Extract heading if provided with ## prefix, otherwise leave empty
    short_heading = ""
    if result.split('\n')[0].startswith('##'):
        short_heading = result.split('\n')[0].lstrip('#').strip()
        # Remove heading from result
        result = '\n'.join(result.split('\n')[1:])
    
    # Create an embed for the response
    embed = discord.Embed(
        title=f"🏆 Megapig Sports Update ",
        description=result,
        color=discord.Color.green()
    )
    
    if user_name and query:
        embed.set_footer(text=f"Requested by {user_name} (query: {query})")
    
    return embed

async def get_sports_score(query: str, client=None, live=False) -> str:
    """Get sports scores from the query."""
    try:
        if client is None:
            raise Exception("OpenAI client not provided")
        
        # Check if query contains NOW or LIVE to override the live parameter
        if any(word in query.upper() for word in ["NOW", "LIVE"]):
            live = True
            # Remove NOW/LIVE from query to not confuse the parser
            query = query.upper().replace("NOW", "").replace("LIVE", "").strip()
            
        # Parse the natural language query
        parsed = await parse_sports_query(query, client)
        logger.info(f"Parsed {'live' if live else 'historical'} query: {parsed}")
        
        # Build and execute API request with retry
        request = build_espn_request(parsed, live=live)
        response = None
        try:
            logger.info(f"{'Live' if live else 'Historical'} request: {request}")
            response = session.get(request["url"], params=request["params"])
        except Exception as e:
            logger.warning(f"First API attempt failed: {e}, retrying once...")
            try:
                await asyncio.sleep(1)
                response = session.get(request["url"], params=request["params"])
            except Exception as retry_e:
                logger.error(f"Retry also failed: {retry_e}")
                raise retry_e
        
        if not response:
            raise Exception("Failed to get response from API")
            
        rjson = response.json()
        if not live:
            scores = clean_espn_response(rjson) 
        else:
            scores = rjson
        
        # Adjust system prompt based on whether this is live or historical
        system_prompt = f"""
            You are Megapig... 
            
            You are providing the family some {'live' if live else ''} sports updates. 
            This data was extracted to satisfy a query.
            The exact query was: "{query}".
            
            {'For live games, focus on the current score, time remaining, and any recent significant events.' if live else ''}
            Format the game scores and/or schedules most consistent with the query into a witty response using markdown.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Scores: {json.dumps(scores)}"}
        ]
        
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Failed to get sports score: {e}")
        raise Exception(f"Failed to get sports score: {e}")