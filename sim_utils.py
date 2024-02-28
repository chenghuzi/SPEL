from analysis import preprocess


def read_sim_data(sim_dir, fn_prefix='rnn_agent_records'):
    fn_balanced = sim_dir / f'{fn_prefix}-balanced.pth'
    fn_prepotent = sim_dir / f'{fn_prefix}-prepotent.pth'

    if 'normal_rnn' in fn_prefix:
        use_e = False
    else:
        use_e = True

    dt = sim_dir.name.split('-')[-1]
    (trial_type_balanced,
     dt_balanced,
     dt2phases_balanced,
     nids_balanced,
     df_all_balanced,
     hs_all_concat_balanced,
     hs_dict_all_balanced) = preprocess(
        fn_balanced, dt, use_e=use_e)
    (trial_type_prepotent,
     dt_prepotent,
     dt2phases_prepotent,
     nids_prepotent,
     df_all_prepotent,
     hs_all_concat_prepotent,
     hs_dict_all_prepotent) = preprocess(
        fn_prepotent, dt, use_e=use_e)

    assert set(nids_balanced) == set(nids_prepotent)
    nids = nids_balanced
    assert dt_balanced == dt_prepotent
    dt = int(dt_balanced)
    return (
        (trial_type_balanced, dt_balanced, dt2phases_balanced, nids_balanced,
         df_all_balanced, hs_all_concat_balanced, hs_dict_all_balanced),
        (trial_type_prepotent, dt_prepotent, dt2phases_prepotent, nids_prepotent,
         df_all_prepotent, hs_all_concat_prepotent, hs_dict_all_prepotent),
        dt,
        nids,
    )
