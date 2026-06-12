import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def _get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def _get_player_color(self, frame, bbox):
        crop = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half = crop[: crop.shape[0] // 2, :]
        kmeans = self._get_clustering_model(top_half)
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - bg_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = [
            self._get_player_color(frame, info["bbox"])
            for info in player_detections.values()
        ]
        if len(player_colors) < 2:
            return
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.kmeans.fit(player_colors)
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self._get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
