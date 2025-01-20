import argparse
import os
# stdlib
import random
import warnings
warnings.filterwarnings("ignore")

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import GenericDataLoader

# synthcity absolute
from synthcity.plugins import Plugins
from GREAT_plugin import PaftPlugin

def data_exist(train_folder, test_folder) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{df_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{df_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{df_name}"')
    
    if not test_exist:
        print(f'No test data for dataset "{df_name}"')

    return train_exist and test_exist

if __name__ == '__main__':
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
                'ctgan',
                'tvae',
                'rtvae',
                'pategan',
                'adsgan',
                ]
    
    init_kwargs = {
        'ctgan' : {'n_iter' : 150},
        'tvae' : {'n_iter' : 150},
        'rtvae' : {'n_iter' : 150},
        'pategan' : {'generator_n_iter' : 150},
        'adsgan' : {'n_iter' : 150},
    }

    fit_kwargs = {
        'great' : {},
        'paft' : {},
        'ctgan' : {},
        'tvae' : {},
        'rtvae' : {},
        'pategan' : {},
        'adsgan' : {},
    }


    generators = Plugins()

    if 'paft' in plugins_list:
        if not os.path.exists('feature_order_permutation.txt'):
            raise Exception('Need to calculate feature order for PAFT benchmarks')
        
        with open('./feature_order_permutation.txt', 'r') as f:
            sorted_order_dict = eval(f.read())

        generators.add('paft', PaftPlugin)
        

    print('Plugins Added')

    # Sort dfs according to row number
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')

    if not args.no_rows_filter:
        data_info = data_info[data_info['row_number'] <= 50_000]

    df_names = data_info.index

    for df_name in df_names:
        print(f'Start "{df_name}"')

        if not data_exist(train_folder, test_folder):
            print(f'"{df_name}": No train or test, skipped')
            continue
        
        df_train = pd.read_csv(f'{train_folder}/{df_name}.csv')
        df_test = pd.read_csv(f'{test_folder}/{df_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type   = data_info.loc[df_name, 'task_type']

        for plugin_name in plugins_list:
            # reorder for paft plugin
            if plugin_name == 'paft':
                order = sorted_order_dict[df_name]

                if order == '':
                    # No df in calculated func dependencies
                    order = df_train.columns.tolist() # a fixed random order for all samples
                    # random order
                    random.shuffle(order)
                    order = ','.join(order)

                train_loader = GenericDataLoader(df_train[order.split(',')], target_column=target_name)
                test_loader  = GenericDataLoader(df_test[order.split(',')], target_column=target_name)            
            else:
                train_loader = GenericDataLoader(df_train, target_column=target_name)
                test_loader  = GenericDataLoader(df_test, target_column=target_name)

            print(f'Start training plugin "{plugin_name}"')
            gen = generators.get(plugin_name,
                                compress_dataset=False,
                                strict=False,
                                **init_kwargs[plugin_name])
            
            gen.fit(train_loader, **fit_kwargs[plugin_name])
            print(plugin_name, 'fitted OK')

            # Start of experiments
            generated_data_folder = args.output_folder + f'/generated_data/{df_name}/{plugin_name}/'
            os.makedirs(os.path.dirname(generated_data_folder), exist_ok=True)

            for repeat in range(args.repeats):
                X_syn = gen.generate(len(df_test))

                repeat_save_path = generated_data_folder + f'/X_syn_{repeat}.csv'
                X_syn.dataframe().to_csv(repeat_save_path, index=False)

                print(plugin_name, repeat, 'generated and saved OK')

        print(df_name, ', Done\n')