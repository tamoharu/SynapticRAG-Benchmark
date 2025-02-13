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
                return {'fire': v[j], 'v': v_rest, 'i': i_rest, 'tau': mem_tau, 'spike': [[1], [count]]}
        return {'fire': -1, 'v': v[-1], 'i': i[-1], 'tau': mem_tau, 'spike': spike}

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
        W = numpy.exp(-numpy.abs(A[:, numpy.newaxis] - B[numpy.newaxis, :])/tau)
        n, m = W.shape
        L = numpy.zeros((n, m))
        L[0, 0] = W[0, 0]
        L[1:, 0] = numpy.cumsum(W[1:, 0]) + L[0, 0]
        L[0, 1:] = numpy.cumsum(W[0, 1:]) + L[0, 0]
        for i in range(1, n):
            for j in range(1, m):
                L[i, j] = W[i, j] + min(L[i-1, j], L[i, j-1], L[i-1, j-1])
        score = L[-1, -1]
        score = 1 / (1 + numpy.exp(-score))
        return score