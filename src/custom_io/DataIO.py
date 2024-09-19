import torch
import json
from pathlib import Path


class DataHandler:
    def __init__(self, rel_output_dir='output', rel_input_dir='input'):
        self.class_dir = Path(__file__).parent.parent.parent
        self.output_dir = self.class_dir / Path(rel_output_dir)
        self.input_dir = self.class_dir / Path(rel_input_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, data, loo, obs):
        output_filename = f'output_data_{loo}_{obs}.json'
        output_file = self.output_dir / output_filename
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def read_all_jsons(self):
        mean_dist_c_list, mean_dist_n_list, avg_var_list = [], [], []
        means_list, mins_list, maxs_list = [], [], []
        observed_list = []
        ratio_par_list, ratio_trans_list, ratio_rot_list, ratio_tot_list = [], [], [], []
        loo_list, obs_list = [], []

        json_files = sorted(self.input_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)

                mean_dist_c = torch.tensor(loaded_data['float_tensors_200']['tensor_1'])
                mean_dist_n = torch.tensor(loaded_data['float_tensors_200']['tensor_2'])
                avg_var = torch.tensor(loaded_data['float_tensors_200']['tensor_3'])

                mean_dist_c_list.append(mean_dist_c)
                mean_dist_n_list.append(mean_dist_n)
                avg_var_list.append(avg_var)

                means = torch.tensor(loaded_data['float_tensors_20']['tensor_1'])
                mins = torch.tensor(loaded_data['float_tensors_20']['tensor_2'])
                maxs = torch.tensor(loaded_data['float_tensors_20']['tensor_3'])

                means_list.append(means)
                mins_list.append(mins)
                maxs_list.append(maxs)

                observed = torch.tensor(loaded_data['bool_tensor_200']['tensor'], dtype=torch.bool)
                observed_list.append(observed)

                ratio_par = loaded_data['floats']['float_1']
                ratio_trans = loaded_data['floats']['float_2']
                ratio_rot = loaded_data['floats']['float_3']
                ratio_tot = loaded_data['floats']['float_4']

                ratio_par_list.append(ratio_par)
                ratio_trans_list.append(ratio_trans)
                ratio_rot_list.append(ratio_rot)
                ratio_tot_list.append(ratio_tot)

                loo = loaded_data['int_variables']['int_var_1']
                obs = loaded_data['int_variables']['int_var_2']

                loo_list.append(loo)
                obs_list.append(obs)

        stacked_data = {
            'accuracy': {
                'mean_dist_c': torch.stack(mean_dist_c_list),
                'mean_dist_n': torch.stack(mean_dist_n_list),
                'avg_var': torch.stack(avg_var_list)
            },
            'log density': {
                'means': torch.stack(means_list),
                'mins': torch.stack(mins_list),
                'maxs': torch.stack(maxs_list)
            },
            'observed': torch.stack(observed_list),
            'acceptance ratios': {
                'model parameters': torch.tensor(ratio_par_list),
                'translation parameters': torch.tensor(ratio_trans_list),
                'rotation parameters': torch.tensor(ratio_rot_list),
                'combined acceptance ratio': torch.tensor(ratio_tot_list)
            },
            'identifiers': {
                'loo': torch.tensor(loo_list),
                'obs': torch.tensor(obs_list)
            }
        }

        return stacked_data

    def read_single_json(self, filename):
        input_file = self.input_dir / filename
        with open(input_file, 'r') as f:
            loaded_data = json.load(f)
            return loaded_data
