"""

"""

import numpy as np
import matplotlib.pyplot as plt
import plotting as plot_funcs
import key_stream as keyed_num_gen
import estimation as est
import privilege_covariances as priv_cov

SIM_TIMESTEPS = 100

def main():
    # State dimension
    n = 4

    # Measurement dimension
    m = 2

    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])
    
    # TEMP
    F2 = lambda w: np.array([[1, np.sin(w*t)/w, 0, -(1-np.cos(w*t))/w],
                         [0, np.cos(w*t), 0, -np.sin(w*t)], 
                         [0, -(1-np.cos(w*t))/w, 1, np.sin(w*t)/w], 
                         [0, np.sin(w*t), 0, np.cos(w*t)]])
    F = F2(0.05)
    F3 = np.array([[1.0058, 0.0077, -0.0002, -0.0148],
                   [0.7808, 1.0058, -0.2105, -0.0016],
                   [-0.0060, -0.0000, 1.0077, 0.0150],
                   [-0.7962, -0.0060, 1.0294, 1.0077]])
    F = F3

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])
    
    # TEMP
    q = np.array([0.003, 1.0000, -0.005, -2.150])
    Q = np.outer(q,q)

    # Measurement model
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])
    R = np.array([[5, 2], 
                  [2, 5]])
    
    # TEMP
    R2 = 0.001 * np.eye(2)
    R = R2

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[30, 0, 0, 0], 
                         [0, 30, 0, 0], 
                         [0, 0, 30, 0], 
                         [0, 0, 0, 30]])
    
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 1])

    # TEMP
    gt_init_state = np.array([0.5, 1.1, -0.2, 1])

    # Number of privilege classes
    num_priv_classes = 3

    # Additional covariances
    priv_covars = [np.array([[20, 0],
                             [0, 20]]),
                   np.array([[14, 0],
                             [0, 14]]),
                   np.array([[17, 0],
                             [0, 17]])]
    
    # TEMP
    priv_covars = [1*x for x in priv_covars]

    covars_to_remove = priv_cov.priv_covars_to_covars_to_remove(priv_covars)

    # Encryption
    sensor_generators = []
    filter_generators = []
    generator_sets = []
    for _ in range(num_priv_classes):
        generator_sets.append(keyed_num_gen.KeyStreamPairFactory.make_shared_key_streams(3))
    sensor_generators, filter_generators, filter_generators_copy = list(zip(*generator_sets))

    # Filters
    unpriv_filter = est.UnprivFilter(n, m, F, Q, H, R, init_state, init_cov, covars_to_remove)
    all_key_priv_filter = est.MultKeyPrivFilter(n, m, F, Q, H, R, init_state, init_cov, np.zeros((2,2)), covars_to_remove, filter_generators_copy)
    priv_filters = []
    for i in range(num_priv_classes):
        priv_filters.append(est.PrivFilter(n, m, F, Q, H, R, init_state, init_cov, priv_covars[i], covars_to_remove[i], filter_generators[i]))

    # Sensor
    sensor = est.SensorWithPrivileges(n, m, H, R, covars_to_remove, sensor_generators)

    # Ground truth (use same model filter)
    ground_truth = est.GroundTruth(F, Q, gt_init_state)

    # Data Storage
    gts = []
    ys = []
    unpriv_pred_list = []
    all_key_priv_pred_list = []
    priv_pred_lists = [[] for _ in range(num_priv_classes)]
    unpriv_upd_list = []
    all_key_priv_upd_list = []
    priv_upd_lists = [[] for _ in range(num_priv_classes)]

    # Plotting
    plot_funcs.init_matplotlib_params(False)
    fig = plt.figure()

    ax = fig.add_subplot(131)
    ax.set_title(r"Single Simulation Run")
    ax.set_xlabel(r"Location $x$")
    ax.set_ylabel(r"Location $y$")

    ax2 = fig.add_subplot(132)
    ax2.set_title(r"Estimator Traces (Single Run - TEMP)")
    ax2.set_xlabel(r"Simulation Time")
    ax2.set_ylabel(r"Covariance Trace")

    ax3 = fig.add_subplot(133)
    ax3.set_title(r"Estimator Errors (Single Run - TEMP)")
    ax3.set_xlabel(r"Simulation Time")
    ax3.set_ylabel(r"Estimation Error")

    # Start sim
    for _ in range(SIM_TIMESTEPS):
        gt = ground_truth.update()
        y = sensor.measure(gt)

        # Predict
        unpriv_pred = unpriv_filter.predict()
        all_key_priv_pred = all_key_priv_filter.predict()
        priv_preds = []
        for i in range(num_priv_classes):
            priv_preds.append(priv_filters[i].predict())

        # Update
        upriv_upd = unpriv_filter.update(y)
        all_key_priv_upd = all_key_priv_filter.update(y)
        priv_upds = []
        for i in range(num_priv_classes):
            priv_upds.append(priv_filters[i].update(y))

        # Save all data
        gts.append(gt)
        ys.append(y)
        unpriv_pred_list.append(unpriv_pred)
        all_key_priv_pred_list.append(all_key_priv_pred)
        for i in range(num_priv_classes):
            priv_pred_lists[i].append(priv_preds[i])
        unpriv_upd_list.append(upriv_upd)
        all_key_priv_upd_list.append(all_key_priv_upd)
        for i in range(num_priv_classes):
            priv_upd_lists[i].append(priv_upds[i])

    # Ground truth and measurement plots
    gt_c = 'lightgrey'
    gt_legend, = plot_funcs.plot_all_gts(ax, gts, color=gt_c)
    m_legend = plot_funcs.plot_all_measurements(ax, ys, color=gt_c, marker='x')
    plot_funcs.plot_all_gt_measurement_lines(ax, gts, ys, 5, color=gt_c, linestyle='--')

    # Unprivileged estimation plots
    unpriv_c = 'darkred'
    unpriv_legend, = plot_funcs.plot_all_states(ax, [s[0] for s in unpriv_upd_list], linestyle='--', color=unpriv_c)
    plot_funcs.plot_all_state_covs(ax, [s[1] for s in unpriv_upd_list], [s[0] for s in unpriv_upd_list], 10, fill=False, linestyle='--', edgecolor=unpriv_c)
    plot_funcs.plot_all_traces(ax2, [s[1] for s in unpriv_upd_list], linestyle='--', color=unpriv_c)
    plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in unpriv_upd_list], gts, linestyle='--', color=unpriv_c)

    # All key privileged estimation plots
    all_key_priv_c = 'darkgreen'
    all_key_priv_legend, = plot_funcs.plot_all_states(ax, [s[0] for s in all_key_priv_upd_list], linestyle='--', color=all_key_priv_c)
    plot_funcs.plot_all_state_covs(ax, [s[1] for s in all_key_priv_upd_list], [s[0] for s in all_key_priv_upd_list], 10, fill=False, linestyle='--', edgecolor=all_key_priv_c)
    plot_funcs.plot_all_traces(ax2, [s[1] for s in all_key_priv_upd_list], linestyle='--', color=all_key_priv_c)
    plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in all_key_priv_upd_list], gts, linestyle='--', color=all_key_priv_c)

    # Privileged estimation plots
    priv_cs = []
    priv_legends = []
    for i in range(num_priv_classes):
        c = 'C'+str(i)
        priv_update_list = priv_upd_lists[i]
        priv_cs.append(c)

        priv_legends.append(plot_funcs.plot_all_states(ax, [s[0] for s in priv_update_list], color=c)[0])
        plot_funcs.plot_all_state_covs(ax, [s[1] for s in priv_update_list], [s[0] for s in priv_update_list], 10, fill=False, linestyle='-', edgecolor=c)
        plot_funcs.plot_all_traces(ax2, [s[1] for s in priv_update_list], color=c)
        plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in priv_update_list], gts, color=c)
    
    # Shared legend
    fig.legend(handles=[gt_legend, m_legend, unpriv_legend, all_key_priv_legend]+priv_legends, 
               labels=["Ground Truth", "Measurements", "No Key Estimator", "All Key Estimator"]+["Privileged Estimator "+str(i+1) for i in range(num_priv_classes)],
               loc="upper center",
               ncol=2)

    # Show figure
    plt.show()


if __name__ == '__main__':
    main()