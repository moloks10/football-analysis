import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


class HeatmapGenerator:
    PITCH_LENGTH = 23.32
    PITCH_WIDTH = 68.0

    def generate(self, tracks, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        team_positions = {1: [], 2: []}

        for frame_tracks in tracks["players"]:
            for info in frame_tracks.values():
                pos = info.get("position_transformed")
                team = info.get("team")
                if pos is None or team not in team_positions:
                    continue
                team_positions[team].append(pos)

        for team_id, positions in team_positions.items():
            if len(positions) < 5:
                continue
            arr = np.array(positions)
            self._render(arr[:, 0], arr[:, 1], team_id, output_dir)

    def _render(self, x, y, team_id, output_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("#4CAF50")
        ax.set_xlim(0, self.PITCH_LENGTH)
        ax.set_ylim(0, self.PITCH_WIDTH)

        rect = patches.Rectangle(
            (0, 0), self.PITCH_LENGTH, self.PITCH_WIDTH,
            linewidth=2, edgecolor="white", facecolor="none",
        )
        ax.add_patch(rect)
        ax.axvline(x=self.PITCH_LENGTH / 2, color="white", linewidth=1.5)

        try:
            kde = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[0 : self.PITCH_LENGTH : 100j, 0 : self.PITCH_WIDTH : 100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.6, cmap="hot")
        except Exception:
            ax.scatter(x, y, alpha=0.1, s=5, c="red")

        ax.set_title(f"Team {team_id} Heatmap", fontsize=16, color="white")
        fig.patch.set_facecolor("#2d5a27")
        plt.savefig(
            os.path.join(output_dir, f"team_{team_id}_heatmap.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
