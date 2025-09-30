"""
Game Notification System
Monitors sports schedules and notifies channels when games are starting
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import discord
from utils.log import logger


class GameNotificationManager:
    """Manages game notifications for teams across channels"""

    def __init__(self, bot_client, config_file: str = "game_subscriptions.json"):
        self.client = bot_client
        self.config_file = config_file
        self.subscriptions: Dict[str, List[Dict]] = {}  # channel_id -> list of team subscriptions
        self.notified_games: Set[str] = set()  # Track games we've already notified about
        self.check_interval = 900  # Check every 15 minutes
        self.notification_window = 30  # Notify when game starts within 30 minutes
        self.load_subscriptions()

    def load_subscriptions(self):
        """Load subscriptions from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.subscriptions = json.load(f)
                logger.info(f"Loaded {len(self.subscriptions)} channel subscriptions")
            else:
                self.subscriptions = {}
                logger.info("No existing subscriptions found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading subscriptions: {e}")
            self.subscriptions = {}

    def save_subscriptions(self):
        """Save subscriptions to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.subscriptions, f, indent=2)
            logger.info("Saved subscriptions to file")
        except Exception as e:
            logger.error(f"Error saving subscriptions: {e}")

    def add_subscription(self, channel_id: str, team_name: str, league: str,
                        lead_time_minutes: int = 30) -> bool:
        """
        Add a team subscription to a channel

        Args:
            channel_id: Discord channel ID
            team_name: Team name (e.g., "Boise State")
            league: League code (e.g., "ncaaf")
            lead_time_minutes: How many minutes before game to notify
        """
        if channel_id not in self.subscriptions:
            self.subscriptions[channel_id] = []

        # Check if subscription already exists
        for sub in self.subscriptions[channel_id]:
            if sub['team'].lower() == team_name.lower() and sub['league'].lower() == league.lower():
                logger.info(f"Subscription already exists for {team_name} in channel {channel_id}")
                return False

        subscription = {
            'team': team_name,
            'league': league.lower(),
            'lead_time_minutes': lead_time_minutes,
            'added_at': datetime.now().isoformat()
        }

        self.subscriptions[channel_id].append(subscription)
        self.save_subscriptions()
        logger.info(f"Added subscription: {team_name} ({league}) in channel {channel_id}")
        return True

    def remove_subscription(self, channel_id: str, team_name: str, league: str) -> bool:
        """Remove a team subscription from a channel"""
        if channel_id not in self.subscriptions:
            return False

        original_count = len(self.subscriptions[channel_id])
        self.subscriptions[channel_id] = [
            sub for sub in self.subscriptions[channel_id]
            if not (sub['team'].lower() == team_name.lower() and
                   sub['league'].lower() == league.lower())
        ]

        if len(self.subscriptions[channel_id]) < original_count:
            if not self.subscriptions[channel_id]:
                del self.subscriptions[channel_id]
            self.save_subscriptions()
            logger.info(f"Removed subscription: {team_name} ({league}) from channel {channel_id}")
            return True

        return False

    def get_channel_subscriptions(self, channel_id: str) -> List[Dict]:
        """Get all subscriptions for a channel"""
        return self.subscriptions.get(channel_id, [])

    def get_all_subscriptions(self) -> Dict[str, List[Dict]]:
        """Get all subscriptions"""
        return self.subscriptions

    async def check_game_schedule(self, team_name: str, league: str, date: datetime) -> Optional[Dict]:
        """
        Check if a team has a game on the specified date
        Returns game info if found, None otherwise
        """
        try:
            # Build ESPN API request
            api_host = os.getenv('SPORTS_API_HOST', 'site.api.espn.com')

            ENDPOINTS = {
                "mlb": f"http://{api_host}/apis/site/v2/sports/baseball/mlb/scoreboard",
                "nba": f"http://{api_host}/apis/site/v2/sports/basketball/nba/scoreboard",
                "ncaab": f"http://{api_host}/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50",
                "ncaaf": f"http://{api_host}/apis/site/v2/sports/football/college-football/scoreboard?groups=80",
                "nfl": f"http://{api_host}/apis/site/v2/sports/football/nfl/scoreboard",
                "nhl": f"http://{api_host}/apis/site/v2/sports/hockey/nhl/scoreboard",
                "wnba": f"http://{api_host}/apis/site/v2/sports/basketball/wnba/scoreboard",
            }

            if league not in ENDPOINTS:
                logger.warning(f"League {league} not supported")
                return None

            url = ENDPOINTS[league]
            params = {"dates": date.strftime("%Y%m%d")}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"ESPN API returned status {resp.status}")
                        return None

                    data = await resp.json()

            # Search for the team in the events
            if 'events' not in data:
                return None

            for event in data['events']:
                # Check if this team is in the game
                for competition in event.get('competitions', []):
                    for competitor in competition.get('competitors', []):
                        team = competitor.get('team', {})
                        if team_name.lower() in team.get('name', '').lower() or \
                           team_name.lower() in team.get('displayName', '').lower():
                            # Found the team! Return game info
                            return {
                                'event_id': event.get('id'),
                                'name': event.get('name'),
                                'date': event.get('date'),
                                'status': event.get('status', {}).get('type', {}).get('state'),
                                'competition': competition,
                                'team': team.get('displayName')
                            }

            return None

        except Exception as e:
            logger.error(f"Error checking game schedule: {e}")
            return None

    async def create_game_notification(self, channel_id: str, game_info: Dict,
                                      subscription: Dict) -> Optional[discord.Embed]:
        """Create a notification embed for a game"""
        try:
            competition = game_info['competition']
            competitors = competition.get('competitors', [])

            if len(competitors) < 2:
                return None

            # Determine home/away
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)

            if not home_team or not away_team:
                home_team, away_team = competitors[0], competitors[1]

            home_name = home_team.get('team', {}).get('displayName', 'Unknown')
            away_name = away_team.get('team', {}).get('displayName', 'Unknown')

            # Parse game time
            game_time = datetime.fromisoformat(game_info['date'].replace('Z', '+00:00'))

            # Create embed
            embed = discord.Embed(
                title="🏈 Game Starting Soon!",
                description=f"**{away_name}** @ **{home_name}**",
                color=discord.Color.orange(),
                timestamp=game_time
            )

            # Add game details
            venue = competition.get('venue', {})
            if venue:
                embed.add_field(
                    name="Location",
                    value=f"{venue.get('fullName', 'Unknown')}",
                    inline=False
                )

            # Add broadcast info if available
            broadcasts = competition.get('broadcasts', [])
            if broadcasts:
                broadcast_names = [b.get('names', [''])[0] for b in broadcasts if b.get('names')]
                if broadcast_names:
                    embed.add_field(
                        name="Watch on",
                        value=", ".join(broadcast_names),
                        inline=False
                    )

            # Add records if available
            home_record = home_team.get('records', [{}])[0].get('summary', '')
            away_record = away_team.get('records', [{}])[0].get('summary', '')

            if home_record or away_record:
                records = f"**{away_name}**: {away_record or 'N/A'}\n**{home_name}**: {home_record or 'N/A'}"
                embed.add_field(name="Records", value=records, inline=False)

            embed.set_footer(text=f"Subscription: {subscription['team']} | Game time")

            return embed

        except Exception as e:
            logger.error(f"Error creating game notification: {e}")
            return None

    async def check_and_notify(self):
        """Check all subscriptions and send notifications for upcoming games"""
        now = datetime.now()

        for channel_id, subs in self.subscriptions.items():
            try:
                # Get the Discord channel
                channel = self.client.get_channel(int(channel_id))
                if not channel:
                    logger.warning(f"Channel {channel_id} not found, skipping")
                    continue

                for subscription in subs:
                    team = subscription['team']
                    league = subscription['league']
                    lead_time = subscription.get('lead_time_minutes', 30)

                    # Check today's schedule
                    game_info = await self.check_game_schedule(team, league, now)

                    if not game_info:
                        # Also check tomorrow in case game is late tonight
                        tomorrow = now + timedelta(days=1)
                        game_info = await self.check_game_schedule(team, league, tomorrow)

                    if game_info:
                        # Parse game time
                        game_time = datetime.fromisoformat(game_info['date'].replace('Z', '+00:00'))
                        # Make timezone aware for comparison
                        if game_time.tzinfo is None:
                            game_time = game_time.replace(tzinfo=None)
                        if now.tzinfo is None:
                            now = now.replace(tzinfo=None)

                        time_until_game = (game_time - now).total_seconds() / 60  # minutes

                        # Create unique game ID to track notifications
                        game_id = f"{channel_id}_{game_info['event_id']}"

                        # Check if we should notify
                        if 0 <= time_until_game <= lead_time and game_id not in self.notified_games:
                            # Send notification
                            embed = await self.create_game_notification(channel_id, game_info, subscription)
                            if embed:
                                await channel.send(
                                    f"🚨 **{team}** game starting in {int(time_until_game)} minutes!",
                                    embed=embed
                                )
                                self.notified_games.add(game_id)
                                logger.info(f"Sent game notification for {team} to channel {channel_id}")

                        # Clean up old notified games (remove games older than 12 hours)
                        if time_until_game < -720:  # Game was more than 12 hours ago
                            self.notified_games.discard(game_id)

            except Exception as e:
                logger.error(f"Error checking notifications for channel {channel_id}: {e}")

    async def start_monitoring(self):
        """Start the background monitoring task"""
        logger.info("Starting game notification monitoring")
        while True:
            try:
                await self.check_and_notify()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)