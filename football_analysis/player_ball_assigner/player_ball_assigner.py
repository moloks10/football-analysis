from football_analysis.utils.bbox_utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_distance=70):
        self.max_distance = max_distance

    def assign_ball_to_player(self, players, ball_bbox):
        ball_center = get_center_of_bbox(ball_bbox)
        min_dist = float("inf")
        assigned_player = -1
        for player_id, player in players.items():
            bbox = player["bbox"]
            dist = min(
                measure_distance((bbox[0], bbox[3]), ball_center),
                measure_distance((bbox[2], bbox[3]), ball_center),
            )
            if dist < self.max_distance and dist < min_dist:
                min_dist = dist
                assigned_player = player_id
        return assigned_player
