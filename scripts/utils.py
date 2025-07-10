import json
import os
import pandas as pd

def load_shot_data(directory, limit=None):
    """
    Loads all StatsBomb JSON event files from a directory, processes each match to compute
    goal difference, determines if the shooting team is home, adds player position and match_id,
    and also extracts assist type and number of prior passes, then returns a list of shot events
    with these additional features.

    Args:
        directory (str): Path to the directory containing event JSON files (events subfolder).
        limit (int, optional): Maximum number of match files to process. Defaults to None (all files).

    Returns:
        list: List of shot dictionaries with added keys:
            'match_id', 'goal_difference', 'is_home', 'position',
            'assist_type', 'n_prev_passes'
    """
    shots_list = []
    parent_dir = os.path.dirname(directory.rstrip('/'))
    lineup_dir = os.path.join(parent_dir, 'lineups')

    # Gather event filenames
    filenames = [fn for fn in os.listdir(directory) if fn.endswith('.json')]
    if limit:
        filenames = filenames[:limit]

    for filename in filenames:
        match_id   = os.path.splitext(filename)[0]
        event_path = os.path.join(directory, filename)
        lineup_path= os.path.join(lineup_dir, f"{match_id}.json")

        # Skip if lineup file doesn't exist
        if not os.path.exists(lineup_path):
            continue

        # Load event data
        with open(event_path, 'r') as f:
            data   = json.load(f)
            events = data.get('events') if isinstance(data, dict) and 'events' in data else data

        # Load lineup data
        with open(lineup_path, 'r') as f:
            lineup_data = json.load(f)

        # Identify home/away
        home_team_id = lineup_data[0].get('team_id')
        away_team_id = lineup_data[1].get('team_id')

        # Map player_id → position
        player_position_map = {}
        for team in lineup_data:
            for player in team.get('lineup', []):
                pid      = player.get('player_id')
                position = player.get('position_name', 'Unknown')
                player_position_map[pid] = position

        # Sort events
        events.sort(key=lambda e: (e.get('period'), e.get('timestamp')))

        # Track live score
        home_score = away_score = 0

        # Window size for pre-shot sequence
        k = 5

        for idx, event in enumerate(events):
            etype = event.get('type', {}).get('name')

            # Update score
            if etype == 'Shot' and event.get('shot', {}).get('outcome', {}).get('name') == 'Goal':
                tid = event.get('team', {}).get('id')
                if tid == home_team_id:
                    home_score += 1
                else:
                    away_score += 1
            elif etype == 'Own Goal Against':
                tid = event.get('team', {}).get('id')
                if tid == home_team_id:
                    away_score += 1
                else:
                    home_score += 1

            # Compute assist_type from previous event
            assist_type = 'None'
            if idx > 0:
                prev = events[idx - 1]
                if prev.get('type', {}).get('name') == 'Pass' and prev.get('pass', {}).get('shot_assist', False):
                    assist_type = prev['pass'].get('type', {}).get('name', 'None')

            # Count number of passes in the last k events
            start = max(0, idx - k)
            window = events[start:idx]
            n_prev_passes = sum(1 for e in window if e.get('type', {}).get('name') == 'Pass')

            # When it's a shot, build the shot record
            if etype == 'Shot':
                tid      = event.get('team', {}).get('id')
                is_home  = (tid == home_team_id)
                goal_diff= (home_score - away_score) if is_home else (away_score - home_score)
                pid      = event.get('player', {}).get('id')
                pos      = player_position_map.get(pid, 'Unknown')

                shot = {
                    **event,
                    'match_id':        match_id,
                    'goal_difference': goal_diff,
                    'is_home':         is_home,
                    'position':        pos,
                    'assist_type':     assist_type,
                    'n_prev_passes':   n_prev_passes
                }
                shots_list.append(shot)

    return shots_list


def get_freeze_frame(shot_id, freeze_frame):
    """
    Converts a shot's freeze frame data into a structured DataFrame.

    Args:
        shot_id: Identifier for the shot.
        freeze_frame (list): Freeze frame data from a shot event.

    Returns:
        pd.DataFrame: DataFrame with columns 'shot_id', 'player_id', 'x', 'y', 'teammate', 'goalkeeper'.
    """
    if not freeze_frame:
        return pd.DataFrame()

    rows = []
    for p in freeze_frame:
        rows.append({
            'shot_id':    shot_id,
            'player_id':  p.get('player', {}).get('id'),
            'x':          p.get('location', [None, None])[0],
            'y':          p.get('location', [None, None])[1],
            'teammate':   p.get('teammate', False),
            'goalkeeper': p.get('position', {}).get('name') == 'Goalkeeper'
        })

    return pd.DataFrame(rows)