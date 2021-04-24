from configparser import ConfigParser
from distutils.util import strtobool

# # # #
# CONFIG KEYS
utils_section = "UTILS"
# data config:
data_section = "DATA"
csv_section = "CSV"
# model config:
params_section = "PARAMS"
optimizer_section = "OPTIMIZER"
# # # #


def read_config(fpath):
    config = ConfigParser()
    with open(fpath, 'r') as f:
        config.read_file(f)
    return config._sections


def parse_model_config(config_path):
    config = read_config(config_path)
    config[params_section]['conv_layers_num'] = int(config[params_section]['conv_layers_num'])
    config[params_section]['dense_layers_num'] = int(config[params_section]['dense_layers_num'])
    config[params_section]['model_dim'] = int(config[params_section]['model_dim'])
    config[params_section]['hidden_size'] = int(config[params_section]['hidden_size'])
    config[params_section]['dropout'] = float(config[params_section]['dropout'])
    config[params_section]['batchnorm'] = strtobool(config[params_section]['batchnorm'])
    config[optimizer_section]['lr'] = float(config[optimizer_section]['lr'])
    config[optimizer_section]['scheduler'] = float(config[optimizer_section]['scheduler'])
    config[optimizer_section]['n_epochs'] = int(config[optimizer_section]['n_epochs'])
    config[optimizer_section]['batch_size'] = int(config[optimizer_section]['batch_size'])

    return config


def parse_representation_config(config_path):
    config = read_config(config_path)
    config[utils_section]["neighbours"] = strtobool(config[utils_section]["neighbours"])
    config[utils_section]["total_num_hs"] = strtobool(config[utils_section]["total_num_hs"])
    config[utils_section]["formal_charge"] = strtobool(config[utils_section]["formal_charge"])
    config[utils_section]["is_in_ring"] = strtobool(config[utils_section]["is_in_ring"])
    config[utils_section]["is_aromatic"] = strtobool(config[utils_section]["is_aromatic"])

    if "get_positions" in config[utils_section]:
        config[utils_section]["get_positions"] = strtobool(config[utils_section]["get_positions"])

    return config


def parse_data_config(config_path):
    config = read_config(config_path)

    config[utils_section]["cv"] = strtobool(config[utils_section]["cv"])
    config[csv_section]["smiles_index"] = int(config[csv_section]["smiles_index"])
    config[csv_section]["y_index"] = int(config[csv_section]["y_index"])
    config[csv_section]["skip_line"] = strtobool(config[csv_section]["skip_line"])
    config[utils_section]["calculate_parity"] = strtobool(config[utils_section]["calculate_parity"])
    config[utils_section]["calculate_rocauc"] = strtobool(config[utils_section]["calculate_rocauc"])

    if config[csv_section]["delimiter"] == '\\t' or config[csv_section]["delimiter"] == 'tab':
        config[csv_section]["delimiter"] = '\t'

    return config
