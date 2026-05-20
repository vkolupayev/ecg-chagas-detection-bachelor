import numpy as np


class Recurrence:
    def __init__(self, signal, D, d, norm):
        self.signal = signal
        self.D = D
        self.d = d
        self.norm = norm
        self.signal_state = self._create_signal_states()
        self.distance_matrix = self._calculate_distance_matrix()

    def recalculate_distance_matrix(self, D, d, norm):
        self.D = D
        self.d = d
        self.norm = norm
        self.signal_state = self._create_signal_states()
        self.distance_matrix = self._calculate_distance_matrix()

    def _create_signal_states(self):
        # ar tikrai?
        m = len(self.signal) - (self.D - 1) * self.d
        s = (m, self.D)
        ss = np.zeros(s)

        for i in range(m):
            for k in range(self.D):
                ss[i][k] = self.signal[i + k * self.d]
        return ss

    def _calculate_distance_matrix(self):
        dists_idx_no_diag = []
        m = len(self.signal_state)
        for i in range(m):
            for j in range(m):
                if i != j:
                    dist = self.calculate_dist(
                        self.signal_state[i], self.signal_state[j], norm=self.norm
                    )
                    idx = [i, j]
                    dists_idx_no_diag.append([dist, idx])
        return dists_idx_no_diag

    def get_recurrence_coord(self, percent, tolerance=0.1, max_iter=1000, diag=True):
        only_dists = [dist[0] for dist in self.distance_matrix]

        r, calibrated_percent = self._approximate_r(
            only_dists, percent, tolerance, max_iter
        )

        x_coords = [idx[1][0] for idx in self.distance_matrix if idx[0] <= r]
        y_coords = [idx[1][1] for idx in self.distance_matrix if idx[0] <= r]

        # append diagonal coordinates
        if diag:
            m = len(self.signal_state)
            x_coords += list(range(m))
            y_coords += list(range(m))
        return x_coords, y_coords, r, calibrated_percent

    def _approximate_r(self, dists, percent, tolerance, max_iter):
        lower_b = percent - tolerance
        upper_b = percent + tolerance
        r_min = np.min(dists)
        r_max = np.max(dists)

        dists = np.array(dists)
        ratio = percent / 100.0
        current_r = ratio * r_max
        current_percent = self.calculate_percent_below(dists, current_r)

        iter = 0
        # fix tolerance
        while True:
            if (current_percent > lower_b) and (current_percent < upper_b):
                break

            if iter > max_iter:
                print(
                    "Sorry, max iter limit reached; tolerance probably too low, will return last estimated percent"
                )
                break

            if current_percent < lower_b:
                r_min = current_r
                delta_r = r_max - r_min
            elif current_percent > upper_b:
                r_max = current_r
                delta_r = -(r_max - r_min)

            # previous iteration is important - count iter only if it get nowhere
            iter += 1
            current_r += 0.5 * delta_r
            current_percent = self.calculate_percent_below(dists, current_r)

        return current_r, current_percent

    @staticmethod
    def calculate_dist(y_i, y_j, norm):
        u = y_i - y_j
        if norm == "L2":
            dist = np.sqrt(np.sum(np.square(u)))
        elif norm == "L1":
            dist = np.sum(np.abs(u))
        elif norm == "Linf":
            dist = np.max(np.abs(u))
        return dist

    @staticmethod
    def calculate_percent_below(dists, r):
        n = len(dists)
        percent_with_r = (len(dists[dists <= r]) / n) * 100.0
        return percent_with_r
