"""

"""

import numpy as np
import matplotlib.pyplot as plt
import plotting as plot_funcs
import key_stream as keyed_num_gen
import estimation as est
import privilege_covariances as priv_cov

SIM_TIMESTEPS = 100
NUM_PRIVILEGE_CLASSES = 3

def main():
    # State dimension
    n = 4

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
    sen_gen, filter_gen = keyed_num_gen.KeyStreamPairFactory.make_pair()

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[4, 0, 0, 0], 
                         [0, 4, 0, 0], 
                         [0, 0, 4, 0], 
                         [0, 0, 0, 4]])
    
    # Ground truth init
    gt_init_state = np.array([0.5, 1, -0.5, 1])

    # Filters
    unpriv_filter = est.UnprivFilter(n, F, Q, H, R, init_state, init_cov)
    priv_filter = est.PrivFilter(n, F, Q, H, R, init_state, init_cov, None, None)
    all_key_priv_filter = est.MultKeyPrivFilter(n, F, Q, H, R, init_state, init_cov, None, None)

    # Sensor
    sensor = est.SensorWithPrivileges(n, H, R, None, None)

    # Ground truth (use same model filter)
    ground_truth = est.GroundTruth(F, Q, gt_init_state)

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

        up_pred = unpriv_filter.predict()
        p_pred = priv_filter.predict()

        up_update = unpriv_filter.update(y)
        p_update = priv_filter.update(y)

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