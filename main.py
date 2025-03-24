from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
#from formation_detection import FormationDetector, add_formation_with_goalkeeper, smooth_team_formations
from formation_cnn_utils import generate_team_heatmap, load_formation_model, predict_formation_from_heatmap

def main():
    # 1. Read Video Frames
    video_frames = read_video('input_videos/ars6.mp4',  resize_to=(1920, 1080))

    # 2. Initialize Tracker and Get Tracks
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
    tracker.add_position_to_tracks(tracks)

    # 3. Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=False)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 4. View Transformer (adds transformed_position to tracks)
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 5. Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 6. Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # 7. Assign Player Teams (make sure goalkeepers get a team too if needed)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    num_frames = min(len(video_frames), len(tracks['players']))
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        # (Ensure that goalkeepers are assigned a team elsewhere or update here if needed.)

    # 8. Assign Ball Acquisition
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

    

    # 9. CNN-based Formation Detection
    formation_model = load_formation_model('models/best_formation_model.pth')  # pretrained model path
    formation_labels = {0: "3-3-4",
    1: "3-4-3",
    2: "4-2-3-1",
    3: "4-2-4",
    4: "4-3-3",
    5: "4-4-2",
    6: "4-5-1",
    7: "5-3-2"}

    team_formations_per_frame = []
    for frame_num in range(num_frames):
        team_positions = {1: [], 2: []}

        for player_id, player_data in tracks['players'][frame_num].items():
            team_id = player_data.get('team')
            pos = player_data.get('position_transformed')
            if team_id and pos:
                team_positions[team_id].append(pos)

        frame_formations = {}
        for team_id, positions in team_positions.items():
            if len(positions) >= 7:
                print(f"Frame {frame_num} → Team {team_id} → Positions: {len(positions)}")

                heatmap = generate_team_heatmap(positions)
                formation_idx = predict_formation_from_heatmap(formation_model, heatmap)
                formation_label = formation_labels.get(formation_idx, "Unknown")
                frame_formations[team_id] = formation_label
            else:
                frame_formations[team_id] = "Unknown"

        team_formations_per_frame.append(frame_formations)

    # Store it in tracks
    tracks['team_formations'] = team_formations_per_frame

    # 10. Draw Output (annotations)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
 
    
    # Draw Formation Labels in Bottom-Left Corner in Bold Black:
    for i, frame in enumerate(output_video_frames):
        # Get a set of team IDs from players and goalkeepers for this frame.
        teams_in_frame = set()
        for data in list(tracks['players'][i].values()) + list(tracks['goalkeepers'][i].values()):
            team_id = data.get('team')
            if team_id is not None:
                teams_in_frame.add(team_id)
        # Get formation info if available, otherwise default to "Unknown"
        formation_dict = tracks.get('team_formations', [{}])[i]
        y_offset = frame.shape[0] - (30 * len(teams_in_frame)) - 10
        for team_id in sorted(teams_in_frame):
            formation = formation_dict.get(team_id, "Unknown")
            text = f"Team {team_id} formation: {formation}"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30
        output_video_frames[i] = frame

    # 11. Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # 12. Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # 13. Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
