"""

"""

import numpy as np
import matplotlib.pyplot as plt
import plotting as plot_funcs
import key_stream as keyed_num_gen
import estimation as est
import privilege_covariances as priv_cov

SIM_TIMESTEPS = 1000
NUM_SIMS_TO_AVG = 1000

def main():

#==============================================================================#

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

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement models
    H1 = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])
    H2 = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])

    R = np.array([[5, 2], 
                  [2, 5]])

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]])
    
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 1])

#==============================================================================#

    # Single privilege class parameters
    priv_covar = np.array([[35, 0],
                          [0, 35]])
    covar_to_remove = priv_covar

    # Multiple privilege classes parameters
    num_priv_classes = 3
    priv_covars = [np.array([[20, 0],
                             [0, 20]]),
                   np.array([[14, 0],
                             [0, 14]]),
                   np.array([[17, 0],
                             [0, 17]])]
    covars_to_remove = priv_cov.priv_covars_to_covars_to_remove(priv_covars)

    # Encryption for single privilege class
    single_sensor_generator, single_filter_generator = keyed_num_gen.KeyStreamPairFactory.make_shared_key_streams(2)

    # Encryption for multiple privilege classes
    sensor_generators = []
    filter_generators = []
    generator_sets = []
    for _ in range(num_priv_classes):
        generator_sets.append(keyed_num_gen.KeyStreamPairFactory.make_shared_key_streams(3))
    sensor_generators, filter_generators, filter_generators_copy = list(zip(*generator_sets))

#==============================================================================#

    # Bounded filters
    unpriv_filter_bounded = est.UnprivFilter(n, m, F, Q, H1, R, init_state, init_cov, covar_to_remove)
    priv_filter_bounded = est.PrivFilter(n, m, F, Q, H1, R, init_state, init_cov, priv_covar, covar_to_remove, single_filter_generator)

    # Unbounded filters
    unpriv_filter_unbounded = est.UnprivFilter(n, m, F, Q, H2, R, init_state, init_cov, covar_to_remove)
    priv_filter_unbounded = est.PrivFilter(n, m, F, Q, H2, R, init_state, init_cov, priv_covar, covar_to_remove, single_filter_generator)
    
    # Multiple privilege filters
    unpriv_filter_mult = est.UnprivFilter(n, m, F, Q, H1, R, init_state, init_cov, covars_to_remove)
    all_key_priv_filter_mult = est.MultKeyPrivFilter(n, m, F, Q, H1, R, init_state, init_cov, np.zeros((2,2)), covars_to_remove, filter_generators_copy)
    priv_filters = []
    for i in range(num_priv_classes):
        priv_filters.append(est.PrivFilter(n, m, F, Q, H1, R, init_state, init_cov, priv_covars[i], covars_to_remove[i], filter_generators[i]))

    # Sensors
    sensor_bounded = est.SensorWithPrivileges(n, m, H1, R, [covar_to_remove], [single_sensor_generator])
    sensor_unbounded = est.SensorWithPrivileges(n, m, H2, R, [covar_to_remove], [single_sensor_generator])
    sensor_mult = est.SensorWithPrivileges(n, m, H2, R, covars_to_remove, sensor_generators)

    # Ground truth
    ground_truth = est.GroundTruth(F, Q, gt_init_state)

#==============================================================================#

    # Data Storage init
    all_sim_gts = []
    gts = []
    all_sim_ys = []
    ys = []

    all_sim_unpriv_pred_lists = []
    unpriv_pred_list = []
    all_sim_priv_pred_lists = []
    priv_pred_list = []
    
    all_key_priv_pred_list = []
    priv_pred_lists = [[] for _ in range(num_priv_classes)]
    
    all_sim_unpriv_upd_lists = []
    unpriv_upd_list = []
    all_sim_priv_upd_lists = []
    priv_upd_list = []
    
    all_key_priv_upd_list = []
    priv_upd_lists = [[] for _ in range(num_priv_classes)]

    # Plotting init
    plot_funcs.init_matplotlib_params(False, True)
    plot_width = 3.5
    plot_height = 2

    # Bounded figure
    fig_bounded = plt.figure()
    fig_bounded.set_size_inches(w=plot_width, h=plot_height)

    ax_tr_bounded = fig_bounded.add_subplot(121)
    ax_tr_bounded.set_title(r"Estimator Error Covariance Traces")
    ax_tr_bounded.set_xlabel(r"Simulation Time")
    ax_tr_bounded.set_ylabel(r"Trace")

    ax_rmse_bounded = fig_bounded.add_subplot(122)
    ax_rmse_bounded.set_title(r"Estimator Errors")
    ax_rmse_bounded.set_xlabel(r"Simulation Time")
    ax_rmse_bounded.set_ylabel(r"RMSE")

    # # Unbounded figure
    # fig_unbounded = plt.figure()
    
    # ax_tr_unbounded = fig_unbounded.add_subplot(121)
    # ax_tr_unbounded.set_title(r"Estimator Traces (Single Run - TEMP)")
    # ax_tr_unbounded.set_xlabel(r"Simulation Time")
    # ax_tr_unbounded.set_ylabel(r"Covariance Trace")

    # ax_rmse_unbounded = fig_unbounded.add_subplot(122)
    # ax_rmse_unbounded.set_title(r"Estimator Errors (Single Run - TEMP)")
    # ax_rmse_unbounded.set_xlabel(r"Simulation Time")
    # ax_rmse_unbounded.set_ylabel(r"Estimation Error")

    # # Multiple privilege levels figure
    # fig_mult = plt.figure()
    
    # ax_tr_mult = fig_mult.add_subplot(121)
    # ax_tr_mult.set_title(r"Estimator Traces (Single Run - TEMP)")
    # ax_tr_mult.set_xlabel(r"Simulation Time")
    # ax_tr_mult.set_ylabel(r"Covariance Trace")

    # ax_rmse_mult = fig_mult.add_subplot(122)
    # ax_rmse_mult.set_title(r"Estimator Errors (Single Run - TEMP)")
    # ax_rmse_mult.set_xlabel(r"Simulation Time")
    # ax_rmse_mult.set_ylabel(r"Estimation Error")

#==============================================================================#

    # Start sim
    for i in range(NUM_SIMS_TO_AVG):
        gts = []
        ys = []
        unpriv_pred_list = []
        priv_pred_list = []
        unpriv_upd_list = []
        priv_upd_list = []
        for _ in range(SIM_TIMESTEPS):
            gt = ground_truth.update()
            y = sensor_bounded.measure(gt)

            # Predict
            unpriv_pred = unpriv_filter_bounded.predict()
            priv_pred = priv_filter_bounded.predict()
            
            # Update
            upriv_upd = unpriv_filter_bounded.update(y)
            priv_upd = priv_filter_bounded.update(y)
            
            # Save all data
            gts.append(gt)
            ys.append(y)
            unpriv_pred_list.append(unpriv_pred)
            priv_pred_list.append(priv_pred)
            
            unpriv_upd_list.append(upriv_upd)
            priv_upd_list.append(priv_upd)
        
        all_sim_gts.append(gts)
        all_sim_ys.append(ys)
        all_sim_unpriv_pred_lists.append(unpriv_pred_list)
        all_sim_priv_pred_lists.append(priv_pred_list)
        all_sim_unpriv_upd_lists.append(unpriv_upd_list)
        all_sim_priv_upd_lists.append(priv_upd_list)
        
    
    # # Start sim
    # for _ in range(SIM_TIMESTEPS):
    #     gt = ground_truth.update()
    #     y = sensor.measure(gt)

    #     # Predict
    #     unpriv_pred = unpriv_filter.predict()
    #     all_key_priv_pred = all_key_priv_filter.predict()
    #     priv_preds = []
    #     for i in range(num_priv_classes):
    #         priv_preds.append(priv_filters[i].predict())

    #     # Update
    #     upriv_upd = unpriv_filter.update(y)
    #     all_key_priv_upd = all_key_priv_filter.update(y)
    #     priv_upds = []
    #     for i in range(num_priv_classes):
    #         priv_upds.append(priv_filters[i].update(y))

    #     # Save all data
    #     gts.append(gt)
    #     ys.append(y)
    #     unpriv_pred_list.append(unpriv_pred)
    #     all_key_priv_pred_list.append(all_key_priv_pred)
    #     for i in range(num_priv_classes):
    #         priv_pred_lists[i].append(priv_preds[i])
    #     unpriv_upd_list.append(upriv_upd)
    #     all_key_priv_upd_list.append(all_key_priv_upd)
    #     for i in range(num_priv_classes):
    #         priv_upd_lists[i].append(priv_upds[i])

    # Bounded plot
    unpriv_legend, = plot_funcs.plot_all_traces(ax_tr_bounded, [s[1] for s in unpriv_upd_list], linestyle='-', color='darkred')
    priv_legend, = plot_funcs.plot_all_traces(ax_tr_bounded, [s[1] for s in priv_upd_list], linestyle='-', color='darkgreen')

    plot_funcs.plot_root_sqr_error(ax_rmse_bounded, [s[0] for s in unpriv_upd_list], gts, linestyle='-', color='darkred')
    plot_funcs.plot_root_sqr_error(ax_rmse_bounded, [s[0] for s in priv_upd_list], gts, linestyle='-', color='darkgreen')

    fig_bounded.legend(handles=[unpriv_legend, priv_legend], 
               labels=["Unprivileged Estimator", "Privileged Estimator"],
               loc="upper center",
               ncol=2)


    # # Unprivileged estimation plots
    # unpriv_c = 'darkred'
    # unpriv_legend, = plot_funcs.plot_all_states(ax, [s[0] for s in unpriv_upd_list], linestyle='--', color=unpriv_c)
    # plot_funcs.plot_all_state_covs(ax, [s[1] for s in unpriv_upd_list], [s[0] for s in unpriv_upd_list], 10, fill=False, linestyle='--', edgecolor=unpriv_c)
    # plot_funcs.plot_all_traces(ax2, [s[1] for s in unpriv_upd_list], linestyle='--', color=unpriv_c)
    # plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in unpriv_upd_list], gts, linestyle='--', color=unpriv_c)

    # # All key privileged estimation plots
    # all_key_priv_c = 'darkgreen'
    # all_key_priv_legend, = plot_funcs.plot_all_states(ax, [s[0] for s in all_key_priv_upd_list], linestyle='--', color=all_key_priv_c)
    # plot_funcs.plot_all_state_covs(ax, [s[1] for s in all_key_priv_upd_list], [s[0] for s in all_key_priv_upd_list], 10, fill=False, linestyle='--', edgecolor=all_key_priv_c)
    # plot_funcs.plot_all_traces(ax2, [s[1] for s in all_key_priv_upd_list], linestyle='--', color=all_key_priv_c)
    # plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in all_key_priv_upd_list], gts, linestyle='--', color=all_key_priv_c)

    # # Privileged estimation plots
    # priv_cs = []
    # priv_legends = []
    # for i in range(num_priv_classes):
    #     c = 'C'+str(i)
    #     priv_update_list = priv_upd_lists[i]
    #     priv_cs.append(c)

    #     priv_legends.append(plot_funcs.plot_all_states(ax, [s[0] for s in priv_update_list], color=c)[0])
    #     plot_funcs.plot_all_state_covs(ax, [s[1] for s in priv_update_list], [s[0] for s in priv_update_list], 10, fill=False, linestyle='-', edgecolor=c)
    #     plot_funcs.plot_all_traces(ax2, [s[1] for s in priv_update_list], color=c)
    #     plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in priv_update_list], gts, color=c)
    
    # # Shared legend
    # fig.legend(handles=[gt_legend, m_legend, unpriv_legend, all_key_priv_legend]+priv_legends, 
    #            labels=["Ground Truth", "Measurements", "No Key Estimator", "All Key Estimator"]+["Privileged Estimator "+str(i+1) for i in range(num_priv_classes)],
    #            loc="upper center",
    #            ncol=2)

    # Show figure
    plt.show()


if __name__ == '__main__':
    main()