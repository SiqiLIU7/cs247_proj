    filename_train = 'data/' + dataset + '/underexpose_train_click-{}.csv'
    filename_test = 'data/' + dataset + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'
    phase = 3
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    for p in range(phase):
        print("load data for phase{}".format(p))
        df_train = pd.read_csv(
            filename_train.format(p), sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'timestamp'], dtype={ 'u_nodes': np.int32, 'v_nodes': np.int32, 'timestamp': np.float64})
        df_train['ratings'] = 1.
        df_train['ratings'].astype(dtypes['ratings'])
        df_test = pd.read_csv(
            filename_test.format(p, p), sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'timestamp'], dtype={ 'u_nodes': np.int32, 'v_nodes': np.int32, 'timestamp': np.float64})
        df_test['ratings'] = 1.
        df_test['ratings'].astype(dtypes['ratings'])

        data_train = data_train.append(df_train)
        data_test = data_test.append(df_test)

    data_all = data_train.append(data_test)

    # data_all = data_all.groupby(['u_nodes', 'v_nodes'])['ratings'].count().reset_index()
    #
    # plt.hist(data_all['ratings'].values.tolist())
    # plt.show()

    time_list = data_all['timestamp'].values.tolist()
    max_time = np.max(time_list)
    min_time = np.min(time_list)
    data_all['timestamp'] = 10 * (data_all['timestamp'] - min_time)/(max_time - min_time)

    data_all['ratings'] = data_all['ratings'] * data_all['timestamp']
    data_all = data_all.groupby(['u_nodes', 'v_nodes'])['ratings'].sum().reset_index()
    data_all['ratings'] = np.ceil(data_all['ratings'] / 5)
    data_all = data_all.sort_values(by=['u_nodes']).reset_index(drop=True)

    del data_train
    del data_test

    data_array = data_all.as_matrix().tolist()
    data_array = np.array(data_array)
    train_test_split = int(math.ceil(data_array.shape[0] * 0.6))
    data_array_train = data_array[:train_test_split]
    data_array_test = data_array[train_test_split:]

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])