import os
import csv
import pickle
import cv2
from formation_cnn_utils import generate_team_heatmap

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_row(row):
    clip_name = row['\ufeffclip_name'].strip()
    frame_start = int(row['frame_start'])
    frame_end = int(row['frame_end'])
    team_id = int(row['team_id'])
    formation = row['formation'].strip()

    # Build the tracks file path (assumes tracks are saved in the "tracks" folder)
    track_file = os.path.join('tracks', clip_name.split('.')[0] + '_tracks.pkl')
    if not os.path.exists(track_file):
        print(f"[Missing File] Tracks file {track_file} not found for {clip_name}")
        return

    with open(track_file, 'rb') as f:
        tracks = pickle.load(f)

    # Sanity check for player frames
    total_frames = len(tracks.get('players', []))
    if total_frames == 0:
        print(f"[Empty] No player data in tracks for {clip_name}")
        return

    # Process every frame in the segment
    for frame_index in range(frame_start, frame_end + 1):
        if frame_index >= total_frames:
            print(f"[Out of Range] {clip_name} has only {total_frames} frames. Requested: {frame_index}. Skipping...")
            continue

        frame_players = tracks['players'][frame_index]

        positions = []
        for player_id, player_data in frame_players.items():
            if player_data.get('team') == team_id:
                pos = player_data.get('position_transformed')
                if pos is not None:
                    positions.append(pos)

        if len(positions) == 0:
            print(f"[No Positions] No valid positions for team {team_id} in {clip_name} at frame {frame_index}")
            continue

        # Generate the heatmap from the positions
        heatmap = generate_team_heatmap(positions)

        # Save the heatmap image under dataset/<formation>/
        output_folder = os.path.join('dataset2', formation)
        ensure_dir(output_folder)
        output_filename = os.path.join(
            output_folder, 
            f"{clip_name.split('.')[0]}_team{team_id}_frame{frame_index}.png"
        )
        cv2.imwrite(output_filename, heatmap)
        print(f"Saved heatmap to {output_filename}")

def main():
    csv_filename = 'formation_labels.csv'
    if not os.path.exists(csv_filename):
        print(f"{csv_filename} not found!")
        return

    # Adjust delimiter if needed (here we assume standard CSV)
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            process_row(row)

if __name__ == '__main__':
    main()
