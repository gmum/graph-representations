# based on conv_qsar_fast by Connor Coley
import os
import time
import json
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error
from .graphconv import predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import traceback

logger = logging.getLogger('')

def linreg(x, y):
    '''Computes a linear regression through the origin using OLS'''
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x, y, rcond=-1.)
    return a


def rocauc_plot(true, pred, set_label, saving_directory):
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    try:
        roc_x, roc_y, _ = roc_curve(true, pred)
        rocauc_score = roc_auc_score(true, pred)
        logger.info(f"AUC = {rocauc_score}")

        plt.figure()
        plt.plot(roc_x, roc_y, color='darkorange',
                 lw=2, label='ROC curve (area = %0.3f)' % rocauc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for {}'.format(set_label))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saving_directory, f"{timestamp}_{set_label}_rocauc.png"), bbox_inches='tight')
        plt.clf()

        return rocauc_score
    except Exception as e:
        logger.error(traceback.print_exc())
        logger.error(e)
        return 99999


def parity_plot(true, pred, set_label, saving_directory):
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    try:
        # calculate extermal values
        min_y = np.min((true, pred))
        max_y = np.max((true, pred))

        # casting to float, because the default is float32 which is then not json serialisable
        mse = float(mean_squared_error(true, pred))
        mae = float(mean_absolute_error(true, pred))
        a = linreg(true, pred)  # predicted v observed
        ap = linreg(pred, true)  # observed v predicted

        # printing
        logger.info(f"{set_label}: mse = {mse}, mae = {mae}")
        # Create parity plot
        plt.scatter(true, pred, alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Parity plot for {} '.format(set_label) +
                  '\nMSE = {}, MAE = {}'.format(np.round(mse, 3), np.round(mae, 3)) +
                  '\na = {}'.format(np.round(a, 3)) +
                  '\na` = {}'.format(np.round(ap, 3)))
        plt.grid(True)
        plt.plot(true, true * a, 'r--')
        plt.axis([min_y, max_y, min_y, max_y])
        plt.savefig(os.path.join(saving_directory, f"{timestamp}_{set_label}_parity.png"), bbox_inches='tight')
        plt.clf()

        return mse, mae
    except Exception as e:
        logger.error(traceback.print_exc())
        logger.error(e)
        return 99999, 99999


def test_model(model, data, device, fpath, calculate_parity: bool = True, calculate_rocauc: bool = True,
               predict_func=predict):
    """This function evaluates model performance using test data.

    inputs:
        model - the trained torch model
        data - (train, validation, test)
        fpath - directory for saving the results
        calculate_parity - should parity plot be calculated and saved
        calculate_rocauc - should rocauc plot be calculated and saved
        batch_size - batch_size to use while testing
        """

    # Create folder to dump testing info to
    try:
        os.makedirs(fpath)
    except:  # file exists
        pass

    # Unpack data
    (train, smiles_train), (val, smiles_val), (test, smiles_test) = data

    # make predictions
    set_part_names = ['train', 'val', 'test']
    loaders = (train, val, test)
    true_ys = []
    pred_ys = []

    # MAKE PREDICTIONS
    for data_loder in loaders:
        this_true_ys, this_pred_ys = predict_func(model, data_loder, device)
        
        true_ys.append(this_true_ys)
        pred_ys.append(this_pred_ys)

    #  Plots
    mse_mae = {}
    rocauc = {}

    for set_part, ys, preds in zip(set_part_names, true_ys, pred_ys):
        if len(ys) > 0:
            if calculate_parity:
                mse_mae[set_part] = parity_plot(ys, preds, set_part, saving_directory=fpath)
            if calculate_rocauc:
                rocauc[set_part] = rocauc_plot(ys, preds, set_part, saving_directory=fpath)

    # SAVING EVERYTHING
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    # save mse_mae
    if calculate_parity:
        with open(os.path.join(fpath, f"{timestamp}-mse-mae.json"), "w") as f:
            json.dump(mse_mae, f)

    # save rocauc
    if calculate_rocauc:
        with open(os.path.join(fpath, f"{timestamp}-rocauc.json"), "w") as f:
            json.dump(rocauc, f)

    # Save predictions for each set
    all_smiles = (smiles_train, smiles_val, smiles_test)
 
    for set_part, smiles, true_y, pred_y in zip(set_part_names, all_smiles, true_ys, pred_ys):
        with open(os.path.join(fpath, f"{timestamp}-{set_part}.predictions"), 'w') as fid:
            fid.write('test entry\tsmiles\tactual\tpredicted\tactual - predicted\n')
            for i in range(len(smiles)):
                fid.write(f"{i}\t{smiles[i]}\t{true_y[i]}\t{pred_y[i]}\t{true_y[i] - pred_y[i]}\n")
