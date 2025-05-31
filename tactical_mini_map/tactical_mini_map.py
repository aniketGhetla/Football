import cv2
import numpy as np

class TacticalMiniMap:
    def __init__(self, frame_width, frame_height, x_shift_meters=5):
        self.mini_map_width = 400
        self.mini_map_height = int(self.mini_map_width * (68 / 105))  # Maintain real pitch ratio

        self.pitch_length = 105  # meters (standard)
        self.pitch_width = 68    # meters

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Shift in meters to fix mapping error (positive = right shift, negative = left shift)
        self.x_shift_meters = x_shift_meters

    def draw_mini_map(self, frame, player_tracks, ball_track):
        # Create transparent mini-map
        mini_map = np.zeros((self.mini_map_height, self.mini_map_width, 4), dtype=np.uint8)

        # Draw pitch lines
        center_x = self.mini_map_width // 2
        center_y = self.mini_map_height // 2
        cv2.line(mini_map, (center_x, 0), (center_x, self.mini_map_height), (255, 255, 255, 255), 1)

        # Center circle
        radius_pixels = int((9.15 / 105) * self.mini_map_width)
        cv2.circle(mini_map, (center_x, center_y), radius_pixels, (255, 255, 255, 255), 1)

        # Penalty boxes
        penalty_box_length = 16.5
        penalty_box_width = 40.3
        pbox_length_px = int((penalty_box_length / 105) * self.mini_map_width)
        pbox_width_px = int((penalty_box_width / 68) * self.mini_map_height)

        # Left Penalty Box
        cv2.rectangle(mini_map, (0, (self.mini_map_height - pbox_width_px) // 2),
                      (pbox_length_px, (self.mini_map_height + pbox_width_px) // 2),
                      (255, 255, 255, 255), 1)

        # Right Penalty Box
        cv2.rectangle(mini_map, (self.mini_map_width - pbox_length_px, (self.mini_map_height - pbox_width_px) // 2),
                      (self.mini_map_width, (self.mini_map_height + pbox_width_px) // 2),
                      (255, 255, 255, 255), 1)

        # Draw players
        for track_info in player_tracks.values():
            pos = track_info.get('position_transformed', None)
            team_color = track_info.get('team_color', (0, 255, 0))  # fallback green

            if pos is None:
                continue

            x_meters, y_meters = pos
            # Apply X shift correction
            mini_x = int(((x_meters + self.x_shift_meters) / 105) * self.mini_map_width)
            mini_y = int((y_meters / 68) * self.mini_map_height)

            # Clip inside mini-map bounds
            mini_x = np.clip(mini_x, 0, self.mini_map_width - 1)
            mini_y = np.clip(mini_y, 0, self.mini_map_height - 1)

            color_with_alpha = (*team_color, 255)
            cv2.circle(mini_map, (mini_x, mini_y), 5, color_with_alpha, -1)

        # Draw ball
        if ball_track:
            ball_pos = ball_track.get(1, {}).get('position_transformed', None)
            if ball_pos is not None:
                x_meters, y_meters = ball_pos
                mini_x = int(((x_meters + self.x_shift_meters) / 105) * self.mini_map_width)
                mini_y = int((y_meters / 68) * self.mini_map_height)
                mini_x = np.clip(mini_x, 0, self.mini_map_width - 1)
                mini_y = np.clip(mini_y, 0, self.mini_map_height - 1)
                cv2.circle(mini_map, (mini_x, mini_y), 3, (0, 140, 255, 255), -1)  # Neon Orange


        # Add white border
        border_thickness = 5
        bordered_map = cv2.copyMakeBorder(
            mini_map,
            border_thickness, border_thickness,
            border_thickness, border_thickness,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255, 255]
        )

        # Position bottom-center
        bordered_height, bordered_width = bordered_map.shape[:2]
        x_offset = (self.frame_width - bordered_width) // 2
        y_offset = self.frame_height - bordered_height - 20

        frame = self.overlay_transparent(frame, bordered_map, x_offset, y_offset)

        return frame

    def overlay_transparent(self, background, overlay, x, y):
        """Overlay transparent mini_map onto the frame."""
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))

        mask = cv2.merge((a, a, a))

        h, w, _ = overlay.shape

        roi = background[y:y+h, x:x+w]

        img1_bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(overlay_rgb, mask)

        dst = cv2.add(img1_bg, img2_fg)

        background[y:y+h, x:x+w] = dst

        return background
