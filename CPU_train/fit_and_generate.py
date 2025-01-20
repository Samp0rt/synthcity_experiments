import argparse
import os 
from ast import literal_eval

# stdlib
import warnings

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import GenericDataLoader


# synthcity absolute
from synthcity.plugins import Plugins
from GANBLR_plugin import GANBLR_plugin, GANBLRPP_plugin


def data_exist(train_folder, test_folder, df_name) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{df_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{df_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{df_name}"')
    
    if not test_exist:
        print(f'No test data for dataset "{df_name}"')

    return train_exist and test_exist

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Baseline Runner")
    parser.add_argument("--data_folder", default=f"./data", help="Data Folder")
    parser.add_argument("--output_folder", default=f"./results", help="Data Folder")
    parser.add_argument("--repeats", default=5, type=int, help="Number of experiments")
    parser.add_argument("--no_rows_filter", action='store_true', help="Include dfs with rows >= 50k")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder

    train_folder = f'{data_folder}/train'
    test_folder = f'{data_folder}/test'

    if not os.path.exists(f'{data_folder}/data_info.csv'):
        raise Exception(f'Missing Targets df in folder "{data_folder}"')

    plugins_list = [
                'ganblr',
                'ganblr++',
                'ddpm',
                'bayesian_network'
               ]
    
    init_kwargs = {
        'ganblr' : {},
        'ganblr++' : {'random_state' : 239},
        'ddpm' : {},
        'bayesian_network' : {}
    }

    fit_kwargs = {
        'ganblr' : {'epochs' : 150, 'verbose' : 10},
        'ganblr++' : {'epochs' : 150, 'verbose' : 10},
        'ddpm' : {},
        'bayesian_network' : {}
    }

    generators = Plugins()
        
    if 'ganblr' in plugins_list:
        generators.add('ganblr', GANBLR_plugin)
    
    if 'ganblr++' in plugins_list:
        generators.add('ganblr++', GANBLRPP_plugin)


    print('Plugins Added')

    # Sort dfs according to row number
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')
    
    if not args.no_rows_filter:
        data_info = data_info[data_info['row_number'] <= 50_000]

    df_names = data_info.index

    for df_name in df_names:
        print(f'Start "{df_name}"')

        if not data_exist(train_folder, test_folder, df_name):
            print(f'"{df_name}", No train or test')
            continue
        
        df_train = pd.read_csv(f'{train_folder}/{df_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type   = data_info.loc[df_name, 'task_type']
        df_test_len = data_info.loc[df_name, 'df_test_len']

        # get numerical columns for init 
        if 'ganblr++' in plugins_list:
            init_kwargs['ganblr++']['numerical_columns'] = literal_eval(data_info.loc[df_name, 'numeric_cols_indxs'])
        
        for plugin_name in plugins_list:
            train_loader = GenericDataLoader(df_train, target_column=target_name)

            print(f'Start training plugin "{plugin_name}"')
            gen = generators.get(plugin_name,
                                strict=False,
                                compress_dataset=False,
                                **init_kwargs[plugin_name])
            
            gen.fit(train_loader, **fit_kwargs[plugin_name])
            print(plugin_name, 'fitted OK')

            # Start of experiments
            generated_data_folder = args.output_folder + f'/generated_data/{df_name}/{plugin_name}/'
            os.makedirs(os.path.dirname(generated_data_folder), exist_ok=True)

            for repeat in range(args.repeats):
                X_syn = gen.generate(df_test_len)

                repeat_save_path = generated_data_folder + f'/X_syn_{repeat}.csv'
                
                X_syn.dataframe().to_csv(repeat_save_path, index=False)

                print(plugin_name, repeat, 'generated and saved OK')

            
        print(df_name, ', Done\n')