import numpy as np
from scipy.optimize import linear_sum_assignment
from formation_detection.formation_template import formations
from collections import Counter

class FormationDetector:
    def __init__(self, pitch_length=23.32, pitch_width=68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.formations = formations  # Dictionary of formation templates

    def detect_formation_for_team(self, player_positions):
        
        #find the best-fit formation.
                
        if len(player_positions) < 10:
            return ("Unknown", float("inf"))
        
        player_positions = np.array(player_positions)
        best_formation = "Unknown"
        min_cost = float("inf")
        
        for formation_name, template_positions in self.formations.items():
            template_positions = np.array(template_positions)
            if len(template_positions) != len(player_positions):
                continue

            # Create cost matrix based on Euclidean distance
            cost_matrix = np.linalg.norm(player_positions[:, None] - template_positions[None, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()

            if total_cost < min_cost:
                min_cost = total_cost
                best_formation = formation_name
        
        return best_formation, min_cost

def add_formation_with_goalkeeper(formation_detector, tracks):
   
    TARGET_OUTFIELD = 10  # number of outfield players
    num_frames = len(tracks['players'])
    if 'team_formations' not in tracks:
        tracks['team_formations'] = [dict() for _ in range(num_frames)]
    
    for frame_num in range(num_frames):
        # Build a dictionary for each team: team_id -> {"goalkeeper": pos, "outfield": []}
        teams = {}
        # Process outfield players from tracks["players"]
        for player_id, data in tracks['players'][frame_num].items():
            team_id = data.get('team')
            pos = data.get('transformed_position')
            if team_id is None or pos is None:
                continue
            if team_id not in teams:
                teams[team_id] = {"goalkeeper": None, "outfield": []}
            teams[team_id]["outfield"].append(pos)
        
        # Process goalkeepers from tracks["goalkeepers"]
        for gk_id, data in tracks['goalkeepers'][frame_num].items():
            team_id = data.get('team')
            pos = data.get('transformed_position')
            if team_id is None or pos is None:
                continue
            if team_id not in teams:
                teams[team_id] = {"goalkeeper": None, "outfield": []}
            teams[team_id]["goalkeeper"] = pos
        
        # For each team, decide on formation
        for team_id, group in teams.items():
            goalkeeper_pos = group["goalkeeper"]
            outfield_positions = group["outfield"]
            formation_str = "Unknown"
            if goalkeeper_pos is not None:
                # Check if we have at least TARGET_OUTFIELD outfield players
                if len(outfield_positions) == TARGET_OUTFIELD:
                    selected_positions = outfield_positions
                elif len(outfield_positions) > TARGET_OUTFIELD:
                    sorted_positions = sorted(outfield_positions, key=lambda pos: pos[0])
                    start_index = (len(sorted_positions) - TARGET_OUTFIELD) // 2
                    selected_positions = sorted_positions[start_index:start_index+TARGET_OUTFIELD]
                else:
                    selected_positions = None

                if selected_positions is not None and len(selected_positions) == TARGET_OUTFIELD:
                    # Detect formation for outfield players (e.g., "4-3-3")
                    outfield_formation, cost = formation_detector.detect_formation_for_team(selected_positions)
                    # Prepend "1-" to denote the goalkeeper is always 1.
                    formation_str = f"1-{outfield_formation}"
                else:
                    formation_str = "Unknown"
            else:
                formation_str = "Unknown"

            # Store formation for the team in this frame.
            tracks['team_formations'][frame_num][team_id] = formation_str
            #  update each player's formation field in both players and goalkeepers.
            for player_id, data in tracks['players'][frame_num].items():
                if data.get('team') == team_id:
                    data['formation'] = formation_str
            for gk_id, data in tracks['goalkeepers'][frame_num].items():
                if data.get('team') == team_id:
                    data['formation'] = formation_str
    return tracks

def smooth_team_formations(tracks):
    
    num_frames = len(tracks['team_formations'])
    # First, gather all valid formations per team
    team_formations_all = {}
    for frame_num in range(num_frames):
        formation_dict = tracks['team_formations'][frame_num]
        for team_id, formation in formation_dict.items():
            if formation != "Unknown":
                team_formations_all.setdefault(team_id, []).append(formation)
    
    # Compute majority formation for each team (if available)
    team_majority = {}
    for team_id, formations in team_formations_all.items():
        if formations:
            most_common, count = Counter(formations).most_common(1)[0]
            team_majority[team_id] = most_common
        else:
            team_majority[team_id] = "Unknown"
    
    # update each frame: if a team's formation is "Unknown", replace it with the majority
    for frame_num in range(num_frames):
        formation_dict = tracks['team_formations'][frame_num]
        for team_id in formation_dict:
            if formation_dict[team_id] == "Unknown":
                formation_dict[team_id] = team_majority.get(team_id, "Unknown")
    
    return tracks
