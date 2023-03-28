import os
import sys
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from evaluate import test_model
from savingutils import save_configs, save_history, LoggerWrapper
from config import parse_model_config, parse_representation_config, parse_data_config
from config import utils_section, data_section, params_section, optimizer_section

from chemprop.args import ModelArgs
from chemprop.model import MoleculeModel
from chemprop.features import set_representation
from main_dmpnn_utils import load_data_chemprop, run_epoch, predict

# run: python main.py simple_nn.cfg esol.cfg simple_repr.cfg /home/abc/results

n_args = 1 + 4  # namefile, architecture_config, data_config, representation_config, main_saving_directory
NUM_WORKERS = 2

if __name__ == '__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: {sys.argv[0]} architecture.cfg data.cfg representation.cfg main_saving_directory")
        quit(1)

    # set global saving subdir for this experiment and create it
    # name of the experiment subdir is derived from the names of the configs
    # config name should be: {unique_key}_{whatever}.{ext} ex. 2_best_model.cfg
    basename = lambda x: os.path.basename(x).split('.')[0].split('_')[0]
    dname = "_".join([basename(x) for x in [sys.argv[1], sys.argv[2], sys.argv[3]]])
    saving_dir = os.path.join(sys.argv[4], dname)
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {[basename(x) for x in [sys.argv[1], sys.argv[2], sys.argv[3]]]}')

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configs
    model_config = parse_model_config(sys.argv[1])
    data_config = parse_data_config(sys.argv[2])
    representation_config = parse_representation_config(sys.argv[3])
    save_configs(sys.argv[1], sys.argv[2], sys.argv[3], saving_dir)

    ################
    # # # DATA # # #
    ################
    set_representation(**representation_config[utils_section])
    
    # define train and validation data splits
    if data_config[utils_section]['cv']:
        trains = []
        vals = []
        for val_fold in data_config[data_section].values():
            trains.append([fold for fold in data_config[data_section].values() if fold != val_fold])
            vals.append([val_fold, ])
        splits = list(zip(trains, vals))
    else:
        trains = [[data_config[data_section][key],] for key in data_config[data_section] if 'train' in key.lower()]
        vals = [[data_config[data_section][key],] for key in data_config[data_section] if 'valid' in key.lower()]

        splits = list(zip(trains, vals))

    # load test
    test_smiles, test_labels, test_loader = load_data_chemprop([data_config[utils_section]["test"], ],
                                                               data_config, model_config,
                                                               shuffle=False, num_workers=NUM_WORKERS)
    
    #####################
    # # # MAIN LOOP # # #
    #####################
    for fold_idx, (train_paths, validation_path) in enumerate(splits):
        logger_wrapper.logger.info(f'Running fold {fold_idx+1}/{len(splits)}')
        # subdirectory for this fold, file names
        fold_subdirectory = os.path.join(saving_dir, f"fold{fold_idx+1}")
        try:
            os.makedirs(fold_subdirectory)
        except FileExistsError:
            pass
        timestamp = time.strftime('%Y-%m-%d-%H-%M')
        best_model_path = os.path.join(fold_subdirectory, f"{timestamp}-best_model_weights.pt")

        # loading train and validation datasets
        train_dataset, train_smiles, train_loader = load_data_chemprop(train_paths,
                                                                       data_config, model_config,
                                                                       shuffle=True, num_workers=NUM_WORKERS)
        
        valid_dataset, valid_smiles, valid_loader = load_data_chemprop(validation_path,
                                                                       data_config, model_config,
                                                                       shuffle=False, num_workers=NUM_WORKERS)

        # defining model, optimizer, scheduler, and the loss function
        if "mse" == data_config[utils_section]['cost_function'].lower().strip():
            loss_function = F.mse_loss
            loss_function_valid = mean_squared_error  # workaround cause we have np.arrays not torch.Tensors // FIXME?
            loss_function_model_args = 'mse'
        else:
            raise NotImplementedError("Unknown loss function; only MSE is currently implemented")
        
        modelArgs = ModelArgs(**model_config[params_section], device=device, loss_function=loss_function_model_args)
        model = MoleculeModel(modelArgs).to(device)
        logger_wrapper.logger.info(model)
        logger_wrapper.logger.info(f"Number of conv layers: {model.encoder.encoder[0].depth}")
        
        if "adam" == model_config[optimizer_section]['optimizer'].lower().strip():
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config[optimizer_section]['lr'])
        else:
            raise NotImplementedError("Unknown optimizer; only Adam is currently implemented")

        scheduler = None  # for easy checkup later
        if model_config[optimizer_section]['scheduler'] > 0:
            assert 0 < model_config[optimizer_section]['scheduler'] < 1, "scheduler value must be -1 (no scheduler) or between 0 and 1"
            step_size = int(model_config[optimizer_section]['scheduler'] * model_config[optimizer_section]["n_epochs"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)  # divide lr by ten after every step_size epochs

        # actual training
        train_loss = []
        valid_loss = []
        min_valid_loss = sys.maxsize
        for epoch in range(model_config[optimizer_section]["n_epochs"]):
            # train
            cumulative_epoch_train_loss = run_epoch(model, loss_function, optimizer, train_loader, device)
            # validate
            true_ys, pred_ys = predict(model, valid_loader, device)
            # scheduler
            if scheduler is not None:
                scheduler.step()

            # remember stuff
            epoch_valid_loss = loss_function_valid(pred_ys, true_ys)
            train_loss.append(cumulative_epoch_train_loss / len(train_dataset))
            valid_loss.append(epoch_valid_loss)

            logger_wrapper.logger.info(f'Epoch: {epoch}, train loss: {train_loss[-1]}, valid loss: {epoch_valid_loss}')

            if epoch_valid_loss < min_valid_loss:
                logger_wrapper.logger.info("Saving model")
                torch.save(model.state_dict(), best_model_path)
                min_valid_loss = epoch_valid_loss

        save_history(train_loss, valid_loss, fold_subdirectory)

        # testing on the test set
        # load the best version of the model, then repack data and run the test function
        model.load_state_dict(torch.load(best_model_path))
        model.eval()  # set dropout and batch normalization layers to evaluation mode before running inference
        
        # train gets new loader without shuffling so the order of smiles is OK # FIXME this is not ideal
        train_dataset, train_smiles, train_loader = load_data_chemprop(train_paths,
                                                                       data_config, model_config,
                                                                       shuffle=False, num_workers=NUM_WORKERS)
        data = ((train_loader, train_smiles), (valid_loader, valid_smiles), (test_loader, test_smiles))
        test_model(model, data, device, fold_subdirectory,
                   calculate_parity=data_config[utils_section]["calculate_parity"],
                   calculate_rocauc=data_config[utils_section]["calculate_rocauc"],
                   predict_func=predict)
