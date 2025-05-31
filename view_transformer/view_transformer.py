import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68  # meters
        court_length = 23.32  # meters

        # Original field boundary (corners of the pitch in the image)
        self.pixel_vertices = np.array([
            [90, 1000], 
            [245, 255], 
            [890, 240], 
            [1620, 895]
        ])
        
        # Target field size in meters (FIFA standard pitch is ~68m width)
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Get perspective transform matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        
        # ---- RELAXED VERSION ----
        # COMMENT OUT boundary check — Allow all points to be transformed
        # is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        # if not is_inside:
        #     return None

        # Proceed to transform without rejecting
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        
        previous_positions = {}
    
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
    
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
    
                        # Smoothing
                        if track_id in previous_positions:
                            prev_pos = np.array(previous_positions[track_id])
                            position_transformed = (0.8 * prev_pos + 0.2 * np.array(position_transformed)).tolist()
    
                        previous_positions[track_id] = position_transformed
    
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
