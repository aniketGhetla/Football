from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image to a 2D array
        image_2d = image.reshape(-1, 3)
        
        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract the region of interest using the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Check if the extracted image is empty
        if image.size == 0:
            raise ValueError(f"Empty image extracted from bbox: {bbox}. Please check the bounding box coordinates.")
        
        # Get the top half of the image
        top_half_image = image[0:int(image.shape[0] / 2), :]
        # Fallback to the full image if top_half_image is empty
        if top_half_image.size == 0:
            top_half_image = image

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to match the top half image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster by checking the corners of the image
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id
def closest_color(self, requested_color):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]