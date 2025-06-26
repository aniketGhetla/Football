from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from formation_cnn_utils import generate_team_heatmap, load_formation_model, predict_formation_from_heatmap
from formation_smoothing import stabilize_formations
from tactical_mini_map import TacticalMiniMap



def main():
    # Read Video Frames
    video_frames = read_video('input_videos/clip4.mp4', resize_to=(1920, 1080))

    mini_map_generator = TacticalMiniMap(frame_width=1920, frame_height=1080,x_shift_meters=40)

    # Initialize Tracker and Get Tracks
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
    tracker.add_position_to_tracks(tracks)

    # Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=False)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    num_frames = min(len(video_frames), len(tracks['players']))
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    # CNN-based Formation Detection
    formation_model = load_formation_model('models/best_formation_model.pth')
    formation_labels = {
        0: "3-4-3", 1: "3-5-1", 2: "3-5-2",
        3: "4-2-3-1", 4: "4-2-4", 5: "4-3-3",
        6: "4-4-2", 7: "4-5-1", 8: "5-3-2"
    }

    team_formations_per_frame = []
    last_known_formations = {1: "Unknown", 2: "Unknown"}

    for frame_num in range(num_frames):
        team_positions = {1: [], 2: []}
        for player_id, player_data in tracks['players'][frame_num].items():
            team_id = player_data.get('team')
            pos = player_data.get('position_transformed')
            if team_id and pos:
                team_positions[team_id].append(pos)

        frame_formations = {}
        for team_id, positions in team_positions.items():
            if len(positions) >= 1:
                heatmap = generate_team_heatmap(positions)
                formation_idx = predict_formation_from_heatmap(formation_model, heatmap)
                formation_label = formation_labels.get(formation_idx, last_known_formations.get(team_id, "Unknown"))
                frame_formations[team_id] = formation_label
                last_known_formations[team_id] = formation_label
            else:
                frame_formations[team_id] = last_known_formations.get(team_id, "Unknown")
        team_formations_per_frame.append(frame_formations)

    # Stabilize formations
    fps = 24
    frames_per_two_minutes = fps * 60 * 2
    team_formations_per_frame = stabilize_formations(team_formations_per_frame, min_persist=frames_per_two_minutes)
    tracks['team_formations'] = team_formations_per_frame

    
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks['players'])

    # Forward fill formations
    team_formations = tracks.get('team_formations', [{}])
    if len(team_formations) < len(output_video_frames):
        last_known = team_formations[-1] if team_formations else {1: "Unknown", 2: "Unknown"}
        team_formations.extend([last_known] * (len(output_video_frames) - len(team_formations)))

    # Frame-by-Frame: Add Tactical Map + Formation Labels
    for i, frame in enumerate(output_video_frames):
        frame = frame.copy()

        # Add Mini-map 
        player_tracks = tracks['players'][i]
        ball_track = tracks['ball'][i] if 'ball' in tracks and i < len(tracks['ball']) else {}
        frame = mini_map_generator.draw_mini_map(frame, player_tracks, ball_track)

        # Draw Formation Labels
        formation_dict = team_formations[i]
        team_colors = {1: "Team 1", 2: "Team 2"}
        seen_teams = set()

        for player in player_tracks.values():
            team = player.get('team')
            color = player.get('team_color')
            if isinstance(team, (int, np.integer)) and isinstance(color, str):
                team = int(team)
                if team in [1, 2] and team not in seen_teams:
                    team_colors[team] = color.capitalize()
                    seen_teams.add(team)
            if len(seen_teams) == 2:
                break

        formation_text_1 = f"{team_colors[1]} formation: {formation_dict.get(1, 'Unknown')}"
        formation_text_2 = f"{team_colors[2]} formation: {formation_dict.get(2, 'Unknown')}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 850), (570, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, formation_text_1, (50, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, formation_text_2, (50, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        output_video_frames[i] = frame

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
