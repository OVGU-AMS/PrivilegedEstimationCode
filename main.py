"""

"""

import numpy as np
import matplotlib.pyplot as plt
import plotting as plot_funcs
import key_stream as keyed_num_gen

SIM_TIMESTEPS = 100

"""
 
  ######   ######## 
 ##    ##     ##    
 ##           ##    
 ##   ####    ##    
 ##    ##     ##    
 ##    ##     ##    
  ######      ##    
 
"""

class GroundTruth:
    def __init__(self, F, Q, init_state):
        self.F = F
        self.Q = Q
        self.state = init_state
        return
    
    def update(self):
        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), self.Q)
        self.state = self.F@self.state + w
        return self.state

"""
 
  ######  ######## ##    ##  ######   #######  ########  
 ##    ## ##       ###   ## ##    ## ##     ## ##     ## 
 ##       ##       ####  ## ##       ##     ## ##     ## 
  ######  ######   ## ## ##  ######  ##     ## ########  
       ## ##       ##  ####       ## ##     ## ##   ##   
 ##    ## ##       ##   ### ##    ## ##     ## ##    ##  
  ######  ######## ##    ##  ######   #######  ##     ## 
 
"""

class SensorAbs:
    def measure(self, ground_truth):
        raise NotImplementedError

class SensorNoEvents(SensorAbs):
    def __init__(self, H, R):
        self.H = H
        self.R = R
        return
    
    def measure(self, ground_truth):
        v = np.random.multivariate_normal(np.array([0, 0]), self.R)
        return self.H@ground_truth + v

class SensorWithEventsAbs(SensorNoEvents):
    def measure(self, ground_truth):
        measurement = super().measure(ground_truth)
        measurement_to_send = None
        if self.is_event(measurement):
            measurement_to_send = measurement
        return measurement_to_send
    
    def is_event(self, measurement):
        raise NotImplementedError

class SensorRandEvent(SensorWithEventsAbs):
    def __init__(self, H, R, Z, generator):
        self.H = H
        self.R = R
        self.Z = Z
        self.generator = generator
        return
    
    def is_event(self, measurement):
        u = self.generator.next()
        prob_not_send = np.e ** (-0.5 * (measurement - u)@np.linalg.inv(self.Z)@(measurement - u))
        urand = np.random.uniform()

        print('u=', u, 'urand', urand, 'prob_not_send=', prob_not_send, (-0.5 * (measurement - u)@self.Z@(measurement - u)))

        to_send = True
        if urand <= prob_not_send:
            to_send = False
        return to_send

class SensorTwoEvents(SensorWithEventsAbs):
    def __init__(self, F, Q, H, R, init_x, init_P, Z, s, generator):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.Z = Z
        self.generator = generator
        self.last_sent = None

        # For local filter and trigger mean
        self.trig_x = init_x
        self.trig_P = init_P

        # Compute sending probabilities
        self.s = s
        # TODO get probabilities of sending with Z or not

        return
    
    # TODO predict and optionally update local filter in overridden measure method

    def is_event(self, measurement):
        # Always send the first measurement
        if self.last_sent == None:
            return True
        
        # TODO here need to use a local estimate as mean and choose which trigger to use with generator
        u = self.generator.next()
        prob_not_send = np.e ** (-0.5 * (measurement - u)@np.linalg.inv(self.Z)@(measurement - u))
        urand = np.random.uniform()

        to_send = True
        if urand <= prob_not_send:
            to_send = False
        return to_send

"""
 
 ######## #### ##       ######## ######## ########   ######  
 ##        ##  ##          ##    ##       ##     ## ##    ## 
 ##        ##  ##          ##    ##       ##     ## ##       
 ######    ##  ##          ##    ######   ########   ######  
 ##        ##  ##          ##    ##       ##   ##         ## 
 ##        ##  ##          ##    ##       ##    ##  ##    ## 
 ##       #### ########    ##    ######## ##     ##  ######  
 
"""

class Filter:
    def predict(self):
        raise NotImplementedError
    
    def update(self, measurement):
        raise NotImplementedError

class BaseFilter(Filter):
    def __init__(self, F, Q, H, R, init_state, init_cov):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        return
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return self.x, self.P
    
    def update(self, measurement):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return self.x, self.P

class UPFilterNoUpdate(BaseFilter):
    def update(self, measurement):
        if measurement is not None:
            super().update(measurement)
        return self.x, self.P

class PFilter(BaseFilter):
    def __init__(self, F, Q, H, R, init_state, init_cov, Z, generator):
        super().__init__(F, Q, H, R, init_state, init_cov)
        self.Z = Z
        self.generator = generator
        return
    
    def update(self, measurement):
        R = self.R
        if measurement is None:
            u = self.generator.next()
            measurement = u
            R = self.R + self.Z
        
        S = (self.H@self.P@self.H.T) + R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)

        return self.x, self.P
 
"""
 
 ##     ##    ###    #### ##    ## 
 ###   ###   ## ##    ##  ###   ## 
 #### ####  ##   ##   ##  ####  ## 
 ## ### ## ##     ##  ##  ## ## ## 
 ##     ## #########  ##  ##  #### 
 ##     ## ##     ##  ##  ##   ### 
 ##     ## ##     ## #### ##    ## 
 
"""

def main():
    # Process model (q = noise strength, t = timestep)
    q = 0.02
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])
    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement model
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])
    R = np.array([[5, 2], 
                  [2, 5]])
    
    # Event parameter
    z_var = 500
    Z = np.array([[z_var,     0],
                  [    0, z_var]])
    
    # Encryption
    y_effective_range = 100
    sen_gen, filter_gen = keyed_num_gen.KeyStreamPairFactory.make_pair(2, -y_effective_range, y_effective_range)

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[4, 0, 0, 0], 
                         [0, 4, 0, 0], 
                         [0, 0, 4, 0], 
                         [0, 0, 0, 4]])
    
    # Ground truth init
    gt_init_state = np.array([0.5, 1, -0.5, 1])

    # Filters
    unprivileged_filter = UPFilterNoUpdate(F, Q, H, R, init_state, init_cov)
    privileged_filter = PFilter(F, Q, H, R, init_state, init_cov, Z, filter_gen)

    # Sensor
    sensor = SensorRandEvent(H, R, Z, sen_gen)

    # Ground truth (use same model filter)
    ground_truth = GroundTruth(F, Q, gt_init_state)

    # Data
    gts = []
    ys = []
    up_preds = []
    p_preds = []
    up_updates = []
    p_updates = []

    # Plotting
    plot_funcs.init_matplotlib_params(False)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot_funcs.plot_event_curve(ax2, lambda x: np.e ** (-0.5 * x*(z_var**-1)*x), -y_effective_range, y_effective_range)
    ax2.set_ylim((-0.1, 1.1))

    for _ in range(SIM_TIMESTEPS):
        gt = ground_truth.update()
        y = sensor.measure(gt)

        up_pred = unprivileged_filter.predict()
        p_pred = privileged_filter.predict()

        up_update = unprivileged_filter.update(y)
        p_update = privileged_filter.update(y)

        # Save all data
        gts.append(gt)
        ys.append(y)
        up_preds.append(up_pred)
        p_preds.append(p_pred)
        up_updates.append(up_update)
        p_updates.append(p_update)
    
    # Prints
    print(sep='\n', *(y for y in ys))

    # Ground truth and measurement plots
    plot_funcs.plot_all_gts(ax, gts, color='lightgrey')
    plot_funcs.plot_all_measurements(ax, ys, color='lightgrey', marker='x')
    plot_funcs.plot_all_gt_measurement_lines(ax, gts, ys, 5, color='lightgrey', linestyle='--')

    # Unprivileged estimation plots
    plot_funcs.plot_all_states(ax, [s[0] for s in up_updates], color='red')
    plot_funcs.plot_all_state_covs(ax, [s[1] for s in up_updates], [s[0] for s in up_updates], 5, fill=False, linestyle='-', edgecolor='red')

    # Privileged estimation plots
    plot_funcs.plot_all_states(ax, [s[0] for s in p_updates], color='green')
    plot_funcs.plot_all_state_covs(ax, [s[1] for s in p_updates], [s[0] for s in p_updates], 5, fill=False, linestyle='-', edgecolor='green')

    # Show figure
    plt.show()


if __name__ == '__main__':
    main()