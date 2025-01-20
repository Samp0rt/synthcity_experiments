import argparse
import os 

# stdlib
import random
import warnings
warnings.filterwarnings("ignore")

# synthcity absolute
from synthcity.plugins.core.dataloader import GenericDataLoader

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import GenericDataLoader

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.metrics.scores import ScoreEvaluator


def data_exist(train_folder, test_folder, df_name) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{df_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{df_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{df_name}"')
    
    if not test_exist:
        print(f'No test data for dataset "{df_name}"')

    return train_exist and test_exist


def save_benchmark_csv(output_folder, df_name, plugin_name, benchmark_res):
    save_folder = f'{output_folder}/benchmarks/{df_name}/{plugin_name}/'
    os.makedirs(os.path.dirname(save_folder), exist_ok=True)
    benchmark_res.to_csv(save_folder + 'res_df.csv')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Runner")
    parser.add_argument("--data_folder", default=f"./data", help="Data Folder")
    parser.add_argument("--output_folder", default=f"./results", help="Data Folder")
    parser.add_argument("--repeats", default=5, type=int, help="Number of experiments")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder

    train_folder = f'{data_folder}/train'
    test_folder = f'{data_folder}/test'

    if not os.path.exists(f'{data_folder}/data_info.csv'):
        raise Exception(f'Missing Targets df in folder "{data_folder}"')

    plugins_list = [
                'ctgan',
                'tvae',
                'rtvae', 
                'ddpm', 
                'ganblr',
                'ganblr++', 
                'pategan', 
                'adsgan', 
                'bayesian_network'
                ]
    
    if 'paft' in plugins_list:
        if not os.path.exists('feature_order_permutation.txt'):
            raise Exception('Need to calculate feature order for PAFT benchmarks')
        
        with open('./feature_order_permutation.txt', 'r') as f:
            sorted_order_dict = eval(f.read())

    # benchmark metrics
    metrics =  {
            'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
            'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
            'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
            'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
            'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
    }

    
    # Sort dfs according to row number
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')
    df_names = data_info.index

    for df_name in df_names:
        print(f'Start "{df_name}"')

        if not data_exist(train_folder, test_folder, df_name):
            print(f'{df_name}, Benchmarks skipped')
            continue
        
        df_test = pd.read_csv(f'{test_folder}/{df_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type   = data_info.loc[df_name, 'task_type']

        for plugin_name in plugins_list:
            # reorder for paft plugin
            if plugin_name == 'paft':
                order = sorted_order_dict[df_name]

                if order == '':
                    # No df in calculated func dependencies
                    order = df_test.columns.tolist() # a fixed random order for all samples
                    # random order
                    random.shuffle(order)
                    order = ','.join(order)

                test_loader  = GenericDataLoader(df_test[order.split(',')], target_column=target_name)            
            else:
                test_loader  = GenericDataLoader(df_test, target_column=target_name)

            print(f'Start plugin "{plugin_name}"')
            
            scores = ScoreEvaluator()

            # Start of experiments
            generated_data_folder = output_folder + f'/generated_data/{df_name}/{plugin_name}/'
            if len(os.listdir(generated_data_folder)) < args.repeats:
                print(df_name, plugin_name, 
                      f': not enough generated data, need {args.repeats} but got {len(os.listdir(generated_data_folder))}, skipping experiments...\n')
                continue

            for repeat in range(args.repeats):
                X_syn_path = generated_data_folder + f'X_syn_{repeat}.csv'
                X_syn_df = pd.read_csv(X_syn_path)
                X_syn = GenericDataLoader(X_syn_df, target_column=target_name)

                print(plugin_name, repeat, 'loaded data OK')

                experiment_scores = Metrics.evaluate(
                            X_gt=test_loader,
                            X_syn=X_syn,
                            metrics=metrics,
                            task_type=task_type,
                            random_state=239*repeat,
                            )
        
                mean_score = experiment_scores["mean"].to_dict()
                errors = experiment_scores["errors"].to_dict()
                duration = experiment_scores["durations"].to_dict()
                direction = experiment_scores["direction"].to_dict()

                for key in mean_score:
                    scores.add(
                        key,
                        mean_score[key],
                        errors[key],
                        duration[key],
                        direction[key],
                    )

                print(repeat, 'experiment OK')

            save_benchmark_csv(output_folder, df_name, plugin_name, scores.to_dataframe())
            print(df_name, plugin_name, ', Benchmarks results saved\n')

            
        