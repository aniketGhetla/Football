import os
import pickle
from utils import read_video
from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import cv2
import numpy as np

def export_tracks(clip_path, output_tracks_path):
    # 1. Read video frames
    frames = read_video(clip_path)
    if len(frames) == 0:
        print(f"No frames found in {clip_path}")
        return
    
    # 2. Initialize Tracker (using your model path)
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub=False, stub_path=output_tracks_path)
    tracker.add_position_to_tracks(tracks)
    
    # 3. Estimate and adjust for camera movement
    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames, read_from_stub=False)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # 4. Transform positions to field view
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # 5. Speed and distance (optional for dataset generation, but included)
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # 6. Assign team colors and IDs
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])
    num_frames = min(len(frames), len(tracks['players']))
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # 7. (Optional) Assign ball possession if desired using PlayerBallAssigner...
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        if 'ball' in tracks and 1 in tracks['ball'][frame_num]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    
    # 8. Save tracks to output_tracks_path
    os.makedirs(os.path.dirname(output_tracks_path), exist_ok=True)
    with open(output_tracks_path, 'wb') as f:
        pickle.dump(tracks, f)
    print(f"Tracks saved to {output_tracks_path}")

if __name__ == '__main__':
    # List of your video clip filenames (make sure these exist in your input_videos folder)
    clips = ['clip1.mp4', 'ars1.mp4', 'ars4.mp4','ars6.mp4','dfb2.mp4','dfb5.mp4','dfb8.mp4','dfb12.mp4','dfb11.mp4']
    input_folder = 'input_videos'
    output_folder = 'tracks'
    os.makedirs(output_folder, exist_ok=True)
    
    for clip in clips:
        clip_path = os.path.join(input_folder, clip)
        output_tracks = os.path.join(output_folder, clip.split('.')[0] + '_tracks.pkl')
        print(f"Processing {clip} ...")
        export_tracks(clip_path, output_tracks)
