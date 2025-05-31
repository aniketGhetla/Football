import cv2
import sys 
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_rate = 24  # frames per second
    
    def add_speed_and_distance_to_tracks(self, tracks):
        window_size = int(self.frame_rate / 2)  # 0.5 second window
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
        
            number_of_frames = len(object_tracks)
        
            total_distance = {}
        
            # Step 1: Frame-by-frame distance accumulation
            for frame_num in range(number_of_frames - 1):
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[frame_num + 1]:
                        continue
        
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[frame_num + 1][track_id]['position_transformed']
        
                    if start_position is None or end_position is None:
                        continue
        
                    distance_covered = measure_distance(start_position, end_position)
        
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
        
                    total_distance[object][track_id] += distance_covered
        
                    object_tracks[frame_num][track_id]['distance'] = total_distance[object][track_id]
        
            # Step 2: Smoothed Speed Calculation (windowed)
            for frame_num in range(number_of_frames - window_size):
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[frame_num + window_size]:
                        continue
        
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[frame_num + window_size][track_id]['position_transformed']
        
                    if start_position is None or end_position is None:
                        continue
        
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = window_size / self.frame_rate  # Smoothing window time
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6
        
                    object_tracks[frame_num][track_id]['speed'] = speed_km_per_hour
        
        
    def draw_speed_and_distance(self, frames, player_tracks_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            if frame_num >= len(player_tracks_per_frame):
                continue

            player_tracks = player_tracks_per_frame[frame_num]
            
            for player_id, track_info in player_tracks.items():
                if "speed" in track_info:
                    speed = track_info.get('speed', None)
                    distance = track_info.get('distance', None)
                    if speed is None or distance is None:
                        continue

                    bbox = track_info['bbox']
                    position = get_foot_position(bbox)
                    position = list(position)
                    position[1] += 40  # offset for display

                    position = tuple(map(int, position))
                    cv2.putText(frame, f"{speed:.2f} km/h", position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)
        
        return output_frames
