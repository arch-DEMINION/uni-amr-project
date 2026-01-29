import numpy as np

class residual_dynamics:
    def __init__(self, time : float = 0, starting_x : np.array = np.zeros(6), etah : float = 1, g: float = 9.81, gain : float = 100, w_size = 10, threshold = 0.01) -> None:

        self.time = time
        self.x = starting_x
        self.pre_x = starting_x
        self.etah = etah
        self.gain = gain
        self.g = g
        self.w_size = w_size
        self.threshold = threshold

        self.r = 0
        self.integral = 0
        self.E0 = 0.5 * starting_x[1::2].T @ starting_x[1::2] / np.pow(self.etah, 2)
        self.pre_u = np.zeros(3)

        self.r_history = [self.r]
        self.mean_history = [0]
    
    def update(self, x : np.array, u : np.array, t : float) -> float:
        Dt = (t - self.time)*0.01 # compute delta time for integrate
        _, x2 = x[0::2], x[1::2] # split the new state in positions and velocity
        pre_x1, pre_x2 = self.x[0::2], self.x[1::2] # split the old state in positions and velocity
        pre_pre_x1, pre_pre_x2 = self.pre_x[0::2], self.pre_x[1::2] # split the old state in positions and velocity

        # compute the energy of the system using new state
        E = 0.5 * x2.T@x2 / np.pow(self.etah, 2)

        # compute the integral using old state and control input
        self.integral += 0.5 * Dt * ( pre_pre_x1.T @ pre_pre_x2 - pre_pre_x2.T @ self.pre_u - self.g*pre_pre_x2[2] + self.r_history[max(len(self.r_history)-2, 0)] +\
                                          pre_x1.T @     pre_x2 -     pre_x2.T @ u          - self.g*    pre_x2[2] + self.r)

        self.integral = min(self.integral, 1e3)

        # update the residual signal
        self.r = min(self.gain * (E - self.integral - self.E0), 1e3)

        self.pre_x = self.x
        self.x = x
        self.time = t
        self.r_history.append(np.pow(self.r, 2))
        #print(f"[R]: {self.sliding_window_mean(self.w_size):0.4f}, {self.IsPerturbed()}")
        return self.r
    
    def sliding_window_mean(self, w_size : int = 1) -> float:
        l = max(0, len(self.r_history) - w_size)
        self.mean_history.append( sum(self.r_history[l:-1])/(len(self.r_history) - l) )
        return self.mean_history[-1]
    
    def IsPerturbed(self) -> bool:
        if len(self.mean_history) <= self.w_size: return True

        return sum( [int(e > self.threshold) for e in self.mean_history[-self.w_size: -1]] ) >= self.w_size*0.5
