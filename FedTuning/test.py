"""
    FedTuning
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import numpy as np

# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

from ServerClient.FLServer import FLServer
from ServerClient.FLClientManager import FLClientManager
from ClientSelection.RandomSelection import RandomSelection
from Helper.FileLogger import FileLogger
from FedTuning.FedTuningTunerTest import FedTuningTunerTest

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='FedTuning: Automatic Tuning of Federated Learning Hyper-Parameters from System Perspective',
        add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Add back help
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    required.add_argument("--enable_fedtuning", help="whether enable fedtuning or not", type=str2bool, required=True)
    required.add_argument("--alpha", help="computation time preference, "
                                          "ignored if --enable_fedtuning == False",
                          type=float, default=0)
    required.add_argument("--beta", help="transmission time preference, "
                                         "ignored if --enable_fedtuning == False",
                          type=float, default=0)
    required.add_argument("--gamma", help="computation load preference, "
                                          "ignored if --enable_fedtuning == False",
                          type=float, default=0)
    required.add_argument("--delta", help="transmission load preference, "
                                          "ignored if --enable_fedtuning == False",
                          type=float, default=0)
    required.add_argument("--model", help="model for training", type=str, required=True)
    required.add_argument("--dataset", help="dataset for training", type=str, required=True)
    required.add_argument("--target_model_accuracy", help='target model accuracy, e.g., 0.8.', type=float, required=True)
    required.add_argument("--n_participant", help='number of participants (M)', type=int, required=True)
    required.add_argument("--n_training_pass", help="number of training passes (E), support fraction, e.g., 1.2",
                          type=float, required=True)

    optional.add_argument("--n_consecutive_better",
                          help='stop training if model accuracy is __n_consecutive_better times better than '
                               'the __target_model_accuracy', type=int, default=5)
    optional.add_argument("--trace_id", help='appending __trace_id to the logged file', type=int, default=1)
    # parser._action_groups.append(optional)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # required arguments
    enable_FedTuning = args.enable_fedtuning
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    delta = args.delta
    if not enable_FedTuning:
        # ignore training preferences
        alpha = beta = gamma = delta = 0

    model_name = args.model
    dataset_name = args.dataset
    target_model_accuracy = args.target_model_accuracy
    M = args.n_participant
    E = args.n_training_pass
    # optional arguments
    n_consecutive_better = args.n_consecutive_better
    trace_id = args.trace_id

    E_str = f'{E:.2f}'.replace('.', '_')
    alpha_str = f'{alpha:.2f}'.replace('.', '_')
    beta_str = f'{beta:.2f}'.replace('.', '_')
    gamma_str = f'{gamma:.2f}'.replace('.', '_')
    delta_str = f'{delta:.2f}'.replace('.', '_')
    write_filename = f'{project_dir}/Result/fedtuning_{enable_FedTuning}__{dataset_name}__{model_name}__M_{M}__E_{E_str}__' \
                     f'alpha_{alpha_str}__beta_{beta_str}__gamma_{gamma_str}__delta_{delta_str}__{trace_id}.csv'
    file_logger = FileLogger(file_path=write_filename)

    print(f'FedTuning enabled: {enable_FedTuning}, alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}'
          f'\n\tmodel={model_name}, dataset={dataset_name}, target_model_accuracy={target_model_accuracy}, M={M}, E={E}'
          f'\n\tn_consecutive_better={n_consecutive_better}, trace_id={trace_id}')
    print(f'Saving results to {write_filename}')

    FL_client_manager = FLClientManager(model_name=model_name, dataset_name=dataset_name)
    FL_server = FLServer(model_name=model_name, dataset_name=dataset_name, client_manager=FL_client_manager)

    # Set M, E ranges
    M_min, M_max = 1, FL_server.get_total_number_of_clients()
    E_min, E_max = 0.1, np.Inf

    fedTuningTuner = None
    if enable_FedTuning:
        # we set minimum of E to 1 in FedTuning
        E_min = 1
        fedTuningTuner = FedTuningTunerTest(alpha=alpha, beta=beta, gamma=gamma, delta=delta, initial_M=M, initial_E=E,
                                            M_min=M_min, M_max=M_max, E_min=E_min, E_max=E_max)

    i_round = 0  # index of training rounds
    n_cur_consecutive_better = 0  # number of times the model is higher than the target accuracy

    while n_cur_consecutive_better < n_consecutive_better:

        # increase the round number
        i_round += 1

        # Participant selection method: random
        random_selection = RandomSelection(client_manager=FL_client_manager, n_select_client=M)

        # Select clients to participate
        selected_client_ids = FL_server.select_clients(client_selection_method=random_selection)

        # Assign server model to the selected clients
        FL_server.copy_model_to_clients(client_ids=selected_client_ids)

        # Training selected clients for one round
        FL_server.train_clients_for_one_round(client_ids=selected_client_ids, training_pass=E)

        # Aggregate model weights from clients
        FL_server.aggregate_model_from_clients(client_ids=selected_client_ids)

        # Get cost of the selected clients
        cost_arr = FL_server.get_cost_of_selected_clients(client_ids=selected_client_ids)

        # computation time, transmission time, computation load, and transmission load on this training round
        round_compT = max(cost_arr)
        round_transT = 1.0
        round_compL = sum(cost_arr)
        round_transL = len(cost_arr)

        # Evaluate the server model performance using both validation set and testing set
        accuracy = FL_server.evaluate_model_performance(include_valid=True, include_test=True)

        # record number of consecutive times that model accuracy higher than a target
        if accuracy >= target_model_accuracy:
            n_cur_consecutive_better += 1
        else:
            n_cur_consecutive_better = 0

        eta_and_zeta_arr = [0] * 8
        if enable_FedTuning:
            eta_and_zeta_arr = fedTuningTuner.get_eta_and_zeta()
        eta_and_zeta_str = ','.join(format(x, ".2f") for x in eta_and_zeta_arr)

        print(f'{datetime.datetime.now()} --- round {i_round}, model accuracy: {accuracy:.2f}, '
              f'eta and zeta: {eta_and_zeta_str}, M: {M}, E: {E}, '
              f'compT: {round_compT}, transT: {round_transT}, compL: {round_compL}, transL: {round_transL}')
        cost_str = ','.join(format(x, ".2f") for x in cost_arr)
        file_logger.write(message=f'{i_round},{accuracy:.2f},{eta_and_zeta_str},{M},{E},{cost_str}\n')

        # FedTuning decisions
        if enable_FedTuning:
            M, E = fedTuningTuner.update(model_accuracy=accuracy,
                                         compT=round_compT,
                                         transT=round_transT,
                                         compL=round_compL,
                                         transL=round_transL)

    print(f'Results are saved to {file_logger.get_file_path()}')
    file_logger.close()

    print('Done!')
