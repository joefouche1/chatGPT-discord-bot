import json
from datetime import datetime
from typing import List, Dict

def extract_match_info(json_file_path: str) -> List[Dict]:
    """
    Extract basic match information from ESPN scoreboard JSON data.
    """
    print(f"Opening file: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"JSON loaded. Top-level keys: {list(data.keys())}")
    matches = []
    
    # Navigate through the JSON structure to find events
    leagues = data.get('leagues', [])
    print(f"Found {len(leagues)} leagues")
    
    for league in leagues:
        events = league.get('events', [])
        print(f"Found {len(events)} events in league: {league.get('name', 'Unknown')}")
        
        for event in events:
            match_info = {}
            
            # Get basic game info
            match_info['game_id'] = event.get('id')
            match_info['start_time'] = datetime.fromisoformat(
                event.get('date', '').replace('Z', '+00:00')
            ).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get team information
            competitions = event.get('competitions', [{}])
            competitors = competitions[0].get('competitors', [])
            print(f"Game ID: {match_info['game_id']}, Found {len(competitors)} competitors")
            
            for team in competitors:
                team_type = team.get('homeAway', '')
                ranking = team.get('curatedRank', {}).get('current')
                
                team_info = {
                    'name': team.get('team', {}).get('name'),
                    'ranking': ranking if ranking != 99 else None
                }
                print(f"Team found: {team_info['name']} ({team_type})")
                
                match_info[f'{team_type}_team'] = team_info
            
            # Get broadcast information
            broadcasts = competitions[0].get('broadcasts', [])
            match_info['broadcast'] = [b.get('names', [])[0] for b in broadcasts] if broadcasts else []
            print(f"Broadcast info: {match_info['broadcast']}")
            
            matches.append(match_info)
    
    print(f"\nTotal matches processed: {len(matches)}")
    return matches

# Example usage:
if __name__ == "__main__":
    matches = extract_match_info("scoreboard.json")
    if not matches:
        print("No matches found!")
    else:
        print("\nMatch Details:")
        for match in matches:
            print(f"\nGame: {match.get('home_team', {}).get('name')} vs {match.get('away_team', {}).get('name')}")
            print(f"Start Time: {match['start_time']}")
            print(f"Broadcast: {', '.join(match['broadcast'])}")