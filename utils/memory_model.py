import numpy

class MemoryModel:
    def __init__(self):
        self.dt = 1

    def lif(self, data, count, v_th, tau_init, time_scale, tau_scale, v_rest, i_rest):
        v_init: float = data['v']
        i_init: float = data['i']
        tau: float = data['tau']
        spike: list[list[float]] = data['spike']
        weights, times = spike
        elapsed_time = times[-1] - times[-2]
        t = numpy.arange(0, elapsed_time + 1, self.dt)
        v = numpy.zeros_like(t, dtype=float)
        i = numpy.zeros_like(t, dtype=float)
        s = numpy.zeros_like(t, dtype=float)
        v[0] = v_init
        i[0] = i_init
        s[0] = weights[-1]
        elapsed_time = elapsed_time * time_scale
        mem_tau = tau + (1 - numpy.exp(-elapsed_time)) / (1 + numpy.exp(-elapsed_time))
        tau = (mem_tau + tau_init) * tau_scale
        for j in range(1, len(t)):
            di = (-i[j-1] + s[j-1] - i_rest) * self.dt / tau
            i[j] = i[j-1] + di
            dv = (-(v[j-1] - v_rest) + i[j]) * self.dt / tau
            v[j] = v[j-1] + dv
            if v[j] >= v_th:
                return {'fire': True, 'v': v_rest, 'i': i_rest, 'tau': mem_tau, 'spike': [[1], [count]]}
        return {'fire': False, 'v': v[-1], 'i': i[-1], 'tau': mem_tau, 'spike': spike}

    def stimulate(self, distance, parent_data, child_data, tau_init, tau_scale, bond_scale):
        tau = (0.5 * (parent_data['tau'] + child_data['tau']) + tau_init) * tau_scale
        _, parent_times = parent_data['spike']
        _, child_times = child_data['spike']
        bond_score = self._calculate_bond_score(parent_times, child_times, tau) * bond_scale
        stimulus = distance * bond_score
        return stimulus
    
    def _calculate_bond_score(self, A, B, tau):
        A = numpy.array(A)
        B = numpy.array(B)
        D = numpy.abs(A[:, numpy.newaxis] - B[numpy.newaxis, :])
        n, m = D.shape
        C = numpy.zeros((n, m))
        C[0, 0] = D[0, 0]
        C[1:, 0] = numpy.cumsum(D[1:, 0]) + C[0, 0]
        C[0, 1:] = numpy.cumsum(D[0, 1:]) + C[0, 0]
        for i in range(1, n):
            for j in range(1, m):
                C[i, j] = D[i, j] + min(C[i-1, j], C[i, j-1], C[i-1, j-1])
        i, j = numpy.array(C.shape) - 1
        path = [(i, j)]
        while i > 0 and j > 0:
            step_costs = [C[i-1, j], C[i, j-1], C[i-1, j-1]]
            step_index = numpy.argmin(step_costs)
            if step_index == 0:
                i -= 1
            elif step_index == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
            path.append((i, j))
        while i > 0:
            i -= 1
            path.append((i, j))
        while j > 0:
            j -= 1
            path.append((i, j))
        path = path[::-1]
        score = 0
        for i in path:
            distance = abs(A[i[0]] - B[i[1]])
            score += numpy.exp(-int(distance) / tau)
        score = 1 / (1 + numpy.exp(-score))
        return score