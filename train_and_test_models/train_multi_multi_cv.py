import os
import random
import sys
# sys.path.append("/home/seongjinpark/research/git_repo/proficiency_judgment")
sys.path.append("/home/paperspace/proficiency_judgment")

import numpy as np
import data_prep.prof_data_prep as pdp
import torch

from models.parameters.rnn_params import params
# from models.parameters.mtl_params import params
from models.input_models import *
from models.train_and_test_models import *

print(params)

cuda = True
# Check CUDA
if not torch.cuda.is_available():
    cuda = False

rnn = False
device = torch.device("cuda" if cuda else "cpu")

seed = params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_path = "./"

model = params.model

# path to directory where best models are saved
model_save_path = "output/models/"
# make sure the full save path exists; if not, create it
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_save_path))
# set dir to plot the loss/accuracy curves for training
model_plot_path = "output/plots/"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_plot_path))

if __name__ == "__main__":

    ### 1. BUILD DATASET

    # dir and parameter settings
    data_path ="data/"
    acoustic_path = "data/audio/IS09_featureset"
    rhythm_file = "rhythm_v6.csv"
    feats = "AudioPhon"
    result_file = "models/results.csv"
    rnn = False

    data = pdp.DataPrep(data_path=data_path, acoustic_path=acoustic_path, rhythm_file=rhythm_file, rnn=rnn)
    data.generate_cv_data()

    """
    raw_feat, raw_feat_len, phono_feat, phono_feat_len, phonetic_feat, ys_acc, ys_flu, ys_comp
    """

    print("type of data", type(data))
    train_data = data.get_train()
    test_data = data.get_test()

    print("DATASET CREATED")
    # exit()

    all_test_losses = []

    # 2. CREATE NN
    ratings = ["acc", "flu", "comp"]
    lrs = [1e-04]
    for rating in ratings:
        for lr in lrs:

            predictions_from_cv = []
            gold_label_from_cv = []

            model_type = "AudioPhonMTLNN"

            for i in range(0, 5):
                cv_idx = i + 1
                print("Starting CV{0}".format(cv_idx))

                train_ds = train_data[i]
                test_ds = test_data[i]

                model_save_file = os.path.join(model_save_path,
                                               "CV{0}_{1}_{2}_lr{3}_epoch{4}.pth".format(cv_idx, rating, model_type, lr, params.num_epochs))
                model_name = "{0}_{1}_lr{2}".format(rating, model_type, lr)
                train_state = make_train_state(lr, model_save_file)

                model = MultiInput_multi_cv(device=device, feats=feats, params=params)
                test_model = MultiInput_multi_cv(device=device, feats=feats, params=params)

                optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=params.weight_decay)

                loss_func = torch.nn.MSELoss()

                model = model.to(device)
                print(model)

                # 3. TRAIN AND TEST MODEL

                if not os.path.exists(model_save_file):
                    train_and_predict_embrace_multi_cv(
                        classifier=model,
                        train_state=train_state,
                        feats=feats,
                        rating=rating,
                        cv_idx=cv_idx,
                        train_ds=train_ds,
                        val_ds=test_ds,
                        batch_size=params.batch_size,
                        num_epochs=params.num_epochs,
                        loss_func=loss_func,
                        optimizer=optimizer,
                        device=device,
                        rnn=rnn,
                        model_type=model_type,
                        lr=lr
                    )

                    ### SAVE TRAINING RESULT
                    loss_title = "Training and Dev loss for model CV{0}_{1}".format(
                        cv_idx, model_name
                    )
                    acc_title = "Avg R2 scores for model CV{0}_{1}".format(cv_idx, model_name)

                    # set save names
                    loss_save = "output/plots/CV{0}_{1}_loss.png".format(cv_idx, model_name)
                    acc_save = "output/plots/CV{0}_{1}_r2.png".format(cv_idx, model_name)

                    # plot the loss from model
                    plot_train_dev_curve(
                        train_state["train_loss"],
                        train_state["val_loss"],
                        x_label="Epoch",
                        y_label="Loss",
                        title=loss_title,
                        save_name=loss_save,
                        set_axis_boundaries=False,
                    )
                    # plot the accuracy from model
                    plot_train_dev_curve(
                        train_state["train_r2"],
                        train_state["val_r2"],
                        x_label="Epoch",
                        y_label="R2",
                        title=acc_title,
                        save_name=acc_save,
                        losses=False,
                        set_axis_boundaries=False,
                    )

                    all_test_losses.append(train_state["early_stopping_best_val"])

                ### TEST CV
                test_model.load_state_dict(torch.load(model_save_file))

                test_model.to(device)

                test_model_multi_multi_cv(
                    classifier=test_model,
                    test_ds=test_ds,
                    cv_idx=cv_idx,
                    batch_size=params.batch_size,
                    rating=rating,
                    feats=feats,
                    result_file=result_file,
                    model_type=model_type,
                    lr=lr,
                    device=device,
                    gold_labels=gold_label_from_cv,
                    pred_labels=predictions_from_cv
                )

            all_cv_loss = mean_squared_error(gold_label_from_cv, predictions_from_cv)
            all_cv_r2 = r2_score(gold_label_from_cv, predictions_from_cv)

            with open(result_file, "a") as out:
                # model, feature, rating, lr, r2, mse
                result = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(model_type, feats, rating, lr, params.num_epochs, all_cv_r2, all_cv_loss)
                out.write(result)
