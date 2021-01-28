import os
import random
import sys
import numpy as np
import data_prep.prof_data_prep as pp
import torch
import pickle

from models.parameters.mtl_params import params
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

# sys.path.append("/home/seongjinpark/research/git_repo/automatic_speech_eval/torch_asa")
print(os.getcwd())

data_path = "data/"

model = params.model

# path to directory where best models are saved
model_save_path = "models/output/models/"
# make sure the full save path exists; if not, create it
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_save_path))
# set dir to plot the loss/accuracy curves for training
model_plot_path = "../models/output/plots/"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_plot_path))

if __name__ == "__main__":
    # 1. BUILD DATASET
    data = pp.ASJPrep(data_path=data_path, rnn=rnn)

    print("type of data", type(data))
    train_data = data.get_train()
    test_data = data.get_test()

    print("DATASET CREATED")
    # exit()

    all_test_losses = []
    all_cv_golds = []
    all_cv_preds = []

    # 2. CREATE NN
    for lr in params.lrs:
        for wd in params.weight_decay:
            for cv in range(0, 5):

                train_cv = train_data[cv]
                test_cv = test_data[cv]

                model_type = ("MI-MTL-ATTN-LSTM")

                model = MultiAcousticModelEarlyMTL(params=params)

                optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=wd)

                loss_func = torch.nn.MSELoss(reduction="mean")

                model = model.to(device)
                print(model)

                # create a a save path and file for the model
                model_save_file = "{0}_lr{1}.pth".format(
                    model_type, lr
                )

                model_file = os.path.join(model_save_path, model_save_file)

                train_state = make_train_state(lr, model_file)

                golds, preds = train_and_predict_mtl(
                    classifier=model,
                    train_state=train_state,
                    train_ds=train_data,
                    val_ds=test_data,
                    batch_size=params.batch_size,
                    num_epochs=params.num_epochs,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    device=device
                )

                all_cv_golds.extend(golds)
                all_cv_preds.extend(preds)

                loss_title = "Training and Dev loss for model {0} with lr {1} at cv {2}".format(
                    model_type, lr, cv
                )
                acc_title = "Avg R2 scores for model {0} with lr {1} at cv {2}".format(model_type, lr, cv)

                # set save names
                loss_save = "./output/plots/{0}_lr{1}_cv{2}_loss.png".format(model_type, lr, cv)
                acc_save = "./output/plots/{0}_lr{1}_cv{2}_r2.png".format(model_type, lr, cv)

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

                # plot_train_dev_curve(train_state['train_r2'], train_state['val_r2'], x_label="Epoch",
                #                         y_label="Accuracy", title=acc_title, save_name=acc_save, losses=False,
                #                         set_axis_boundaries=False)

                # add best evaluation losses and accuracy from training to set
                all_test_losses.append(train_state["early_stopping_best_val"])
                # all_test_accs.append(train_state['best_val_acc'])

    with open(os.path.join(data_path, "model_result/accented.csv"), "w") as outfile:
        header = "prediction\tresponse\n"
        outfile.write(header)
        for i in range(len(all_cv_golds)):
            result = "%s\t%s\n" % (all_cv_preds[i], all_cv_golds[i])
            outfile.write(result)

    # for i, item in enumerate(all_test_losses):
    #     print("Losses for model with lr={0}: {1}".format(params.lrs[i], item))
        # print("Accuracy for model with lr={0}: {1}".format(params.lrs[i], all_test_accs[i]))

