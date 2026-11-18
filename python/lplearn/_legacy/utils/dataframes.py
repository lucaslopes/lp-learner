'''
This module contains functions that return dataframes for comparison or plotting purposes.
'''


import numpy as np
import pandas as pd


def compare_results(
        result_learner: dict,
        result_solver: dict,
    ) -> pd.DataFrame:

    '''
    :param result_learner: dict
    :param result_solver: dict
    :return: pd.DataFrame
    '''

    data = list()

    for idx, epoch in enumerate(result_learner['epoch']):

        dict = {
            'Epoch': epoch,
            'Norm Cost': np.linalg.norm(result_learner['solution'][idx] - result_solver['solution']),
            'Norm Y': np.linalg.norm(result_learner['y'][idx] - result_solver['y']),
            'Norm Lambda 1': np.linalg.norm(result_learner['lambda_1'][idx] - result_solver['lambda_1']),
            'Norm Lambda 2': np.linalg.norm(result_learner['lambda_2'][idx] - result_solver['lambda_2']),
            'Norm Lambda 3': np.linalg.norm(result_learner['lambda_3'][idx] - result_solver['lambda_3']),
            'Norm Mu 1': np.linalg.norm(result_learner['mu_1'][idx] - result_solver['mu_1']),
            'Norm Mu 2': np.linalg.norm(result_learner['mu_2'][idx] - result_solver['mu_2'])}

        if 'mu_3' in result_learner:
            dict['Norm Mu 3'] = np.linalg.norm(result_learner['mu_3'][idx] - result_solver['mu_3'])

        data.append(dict)

    df = pd.DataFrame(data).melt(
        id_vars='Epoch',
        var_name='Norm Type',
        value_name='Norm Value')

    return df
