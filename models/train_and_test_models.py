# implement training and testing for models
import os, sys
import numpy as np
import random

# import numpy as np

# import parameters for model
from torch.utils.data import DataLoader, RandomSampler

from models.attn_models import *
from models.parameters.mtl_params import *
from models.plot_training import *

from statistics import mean
from sklearn.metrics import r2_score, mean_squared_error


def calc_r_squared(valy, y_preds):
    mean_y = mean(valy)
    ss_res = mean_squared_error(valy, y_preds)
    ss_tot = np.var(valy, ddof=1)
    # ss_res = []
    # ss_tot = []
    # for i, item in enumerate(valy):
    #     ss_res.append((item - y_preds[i]) ** 2)
    #     ss_tot.append((item - mean_y) ** 2)
    # r_value = 1 - (sum(ss_res) / (sum(ss_tot) + 0.0000001))
    r_value = 1 - (ss_res / (ss_tot + 1e-6))
    return float(r_value)


# adapted from https://github.com/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_6/classifying-surnames/Chapter-6-Surname-Classification-with-RNNs.ipynb
def make_train_state(learning_rate, model_save_file):
    # makes a train state to save information on model during training/testing
    return {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e8,
        "learning_rate": learning_rate,
        "epoch_index": 0,
        "tasks": [],
        "train_loss": [],
        "train_r2": [],
        "val_loss": [],
        "val_r2": [],
        "best_val_loss": [],
        "best_val_r2": [],
        "best_loss": 100,
        "test_loss": -1,
        "test_r2": -1,
        "model_filename": model_save_file,
    }


def update_train_state(model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

        # use best validation accuracy for early stopping
        train_state["early_stopping_best_val"] = train_state["val_loss"][-1]
        # train_state["early_stopping_best_val"] = train_state["val_r2"][-1]
        # train_state['best_val_r2'] = train_state['val_r2'][-1]

    # Save model if performance improved
    elif train_state["epoch_index"] >= 1:
        loss_t = train_state["val_loss"][-1]
        # loss_t = train_state["val_r2"][-1]

        # If loss worsened relative to BEST
        if loss_t >= train_state["early_stopping_best_val"]:
        # if loss_t <= train_state["early_stopping_best_val"]:
            # Update step
            train_state["early_stopping_step"] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state["early_stopping_best_val"]:
            # if loss_t > train_state["early_stopping_best_val"]:
                train_state["early_stopping_best_val"] = loss_t
                torch.save(model.state_dict(), train_state["model_filename"])
                train_state['best_val_loss'] = loss_t
                # train_state['best_val_r2'] = loss_t

            # Reset early stopping step
            train_state["early_stopping_step"] = 0

        # Stop early ?
        train_state["stop_early"] = (
                train_state["early_stopping_step"] >= params.early_stopping_criteria
        )

    return train_state

def train_and_predict_transformer(
        classifier,
        train_state,
        train_ds,
        val_ds,
        # batch_size,
        # num_workers,
        num_epochs,
        loss_func,
        optimizer,
        device,
        scheduler=None,
        sampler=None,
        binary=False,
        split_point=0.0,
):
    # print(classifier)
    for epoch_index in range(num_epochs):
        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0
        running_acc = 0.0

        # set classifier(s) to training mode
        classifier.train()

        # batches = DataLoader(
        #     train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, sampler=sampler
        # )

        # assign holders
        ys_holder = []
        preds_holder = []

        # for batch in batches:
        for (batch_index, batch) in enumerate(train_ds):
            # reset the gradients
            optimizer.zero_grad()

            # Load batch
            input_ids = batch['token'].type(torch.LongTensor)
            # print(input_ids.size())
            # print(batch['length'])
            # attention_mask = batch['length'].to(device)
            labels = torch.as_tensor(batch['label']).to(device)
            # print(labels.size())
            # outputs = classifier(input_ids, attention_mask=attention_mask, labels=labels)
            # outputs = classifier(input_ids, labels=labels)
            outputs = classifier.predict('meld_emotion', input_ids.to(device))
            preds = outputs.argmax(dim=1)

            # print(preds)
            # print(labels)

            ys_holder.extend(labels.tolist())
            preds_holder.extend(preds.tolist())

            loss = loss_func(outputs, labels)
            loss_t = loss.item()

            # calculating running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # backprop. loss
            loss.backward()
            # optimizer for the gradient step
            optimizer.step()

            # compute the accuracy
            acc_t = torch.eq(preds, labels).sum().item() / len(labels)

            running_acc += (acc_t - running_acc) / (batch_index + 1)

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["train_avg_f1"].append(avg_f1[2])
        # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
        print("Training weighted F-score: " + str(avg_f1))

        # Iterate over dev dataset
        running_loss = 0.0
        running_acc = 0.0

        # set classifier(s) to evaluation mode
        classifier.eval()

        # batches = DataLoader(
        #     val_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, sampler=sampler
        # )

        # assign holders
        ys_holder = []
        preds_holder = []

        # for batch in batches:
        for (batch_index, batch) in enumerate(val_ds):
            # Load batch
            input_ids = batch['token'].type(torch.LongTensor)
            # print(batch['length'])
            # attention_mask = batch['length'].to(device)
            labels = torch.as_tensor(batch['label']).to(device)

            # outputs = classifier(input_ids, attention_mask=attention_mask, labels=labels)
            # outputs = classifier(input_ids, labels=labels)
            outputs = classifier.predict('meld_emotion', input_ids.to(device))
            preds = outputs.argmax(dim=1)

            # print(preds)
            # print(labels)

            ys_holder.extend(labels.tolist())
            preds_holder.extend(preds.tolist())

            loss = loss_func(outputs, labels)
            loss_t = loss.item()

            # calculating running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = torch.eq(preds, labels).sum().item() / len(labels)

            running_acc += (acc_t - running_acc) / (batch_index + 1)

        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["val_avg_f1"].append(avg_f1[2])
        print("Weighted F=score: " + str(avg_f1))

        # get confusion matrix
        if epoch_index % 5 == 0:
            print(confusion_matrix(ys_holder, preds_holder))
            print("Classification report: ")
            print(classification_report(ys_holder, preds_holder, digits=4))

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

def train_and_predict_attn(
        classifier,
        train_state,
        train_ds,
        val_ds,
        batch_size,
        num_epochs,
        loss_func,
        optimizer,
        device="cpu",
        scheduler=None,
        sampler=None,
        binary=False,
        split_point=0.0,
):
    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0
        running_acc = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
        )

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # get the gold labels
            y_gold = batch['label'].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )
            # y_gold = batch.targets()

            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            batch_acoustic = batch['audio'].to(device)
            # print(batch_acoustic.size())
            batch_acoustic = batch_acoustic.transpose(1, 2)
            # batch_acoustic = nn.utils.rnn.pad_sequence(batch_acoustic)
            batch_acoustic_lengths = batch['length']

            y_pred, _ = classifier(batch_acoustic, batch_acoustic_lengths)

            # print("pred_size: ", y_pred.size())

            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([item[0] for item in y_pred.tolist()])
                else:
                    # if type(y_gold[0]) == list or torch.is_tensor(y_gold[0]):
                    # y_gold = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
                    y_pred_class = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
                    # y_pred = y_pred
                    # print(y_gold)
                    # print(y_pred)
                    # print(type(y_gold))
                    # print(type(y_pred))
            else:
                y_pred = torch.round(y_pred)
                # y_pred = y_pred.to(device)
            if binary:
                preds_holder.extend(y_pred)
            else:
                preds_holder.extend(y_pred_class)
                y_pred_class = y_pred_class.to(device)
            ys_holder.extend(y_gold.tolist())

            # print(y_pred_class)
            # print(y_gold)
            # print("preds_holder: ", preds_holder)
            # print("ys_holder: ", ys_holder)

            # y_pred = y_pred.squeeze(1).to(device)
            # y_pred_class = y_pred_class.to(device)
            # print(y_pred.size(), y_gold.size())
            # print(y_pred)
            # print(y_gold)
            loss = loss_func(y_pred, y_gold)
            loss_t = loss.item()  # loss for the item

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # compute the accuracy
            acc_t = torch.eq(y_pred_class, y_gold).sum().item() / len(y_gold)

            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy measures for every minibatch
            # print("loss: {0}, running_loss: {1}, acc: {0}, running_acc: {1}".format(loss_t, running_loss,

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        # print(ys_holder)
        # print(preds_holder)
        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["train_avg_f1"].append(avg_f1[2])
        # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
        print("Training weighted F-score: " + str(avg_f1))

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # reset loss and accuracy to zero
        running_loss = 0.0
        running_acc = 0.0

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the dataloader
        for batch_index, batch in enumerate(val_batches):
            # compute the output
            batch_acoustic = batch['audio'].to(device)
            batch_acoustic = batch_acoustic.transpose(1, 2)
            # batch_acoustic = nn.utils.rnn.pad_sequence(batch_acoustic)
            batch_acoustic_lengths = batch['length']

            y_gold = batch['label'].to(device)

            y_pred, _ = classifier(batch_acoustic, batch_acoustic_lengths)

            # print("pred_size: ", y_pred.size())

            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([item[0] for item in y_pred.tolist()])
                else:
                    # if type(y_gold[0]) == list or torch.is_tensor(y_gold[0]):
                    #     y_gold = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
                    y_pred_class = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
                    # y_pred = y_pred
                    # print(y_gold)
                    # print(y_pred)
                    # print(type(y_gold))
                    # print(type(y_pred))
            # else:
            #     y_pred = torch.round(y_pred)

            if binary:
                preds_holder.extend(y_pred)
            else:
                preds_holder.extend(y_pred_class)
                y_pred_class = y_pred_class.to(device)
            ys_holder.extend(y_gold.tolist())

            # print(y_pred)
            # print(y_gold)

            # y_pred = y_pred.squeeze(1).to(device)
            # y_pred_class = y_pred_class.to(device)

            loss = loss_func(y_pred, y_gold)
            loss_t = loss.item()  # loss for the item

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            y_pred = y_pred.to(device)
            # compute the accuracy
            acc_t = torch.eq(y_pred_class, y_gold).sum().item() / len(y_gold)
            # acc_t = torch.eq(torch.round(y_pred), y_gold).sum().item() / len(y_gold)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy for each minibatch
            # print("val_loss: {0}, running_val_loss: {1}, val_acc: {0}, running_val_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["val_avg_f1"].append(avg_f1[2])
        print("Weighted F=score: " + str(avg_f1))

        # get confusion matrix
        if epoch_index % 5 == 0:
            print(confusion_matrix(ys_holder, preds_holder))
            print("Classification report: ")
            print(classification_report(ys_holder, preds_holder, digits=4))

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break


def train_and_predict_mtl(
        classifier,
        train_state,
        train_ds,
        val_ds,
        batch_size,
        num_epochs,
        loss_func,
        optimizer,
        device="cpu",
        scheduler=None,
        sampler=None,
        binary=False,
        split_point=0.0,
):
    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0
        running_acc = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
        )

        # set holders to use for error analysis
        ys_acc_holder = []
        preds_acc_holder = []
        ys_flu_holder = []
        preds_flu_holder = []
        ys_comp_holder = []
        preds_comp_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # get the gold labels
            y_gold_acc = batch['acc'].to(device)
            y_gold_flu = batch['flu'].to(device)
            y_gold_comp = batch['comp'].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )

            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            batch_audio = batch['audio'].to(device)
            batch_length = batch['length']

            batch_acoustic = batch['acoustic'].to(device)
            batch_acoustic_length = batch['acoustic_length']

            y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                audio_input=batch_audio,
                audio_length=batch_length,
                acoustic_input=batch_acoustic,
                acoustic_length=batch_acoustic_length)

            # print("pred_size: ", y_pred.size())
            # For first prediction
            preds_acc_holder.extend(y_pred_acc)
            y_pred_acc = y_pred_acc.to(device)
            ys_acc_holder.extend(y_gold_acc.tolist())

            # For second prediction
            preds_flu_holder.extend(y_pred_flu)
            y_pred_flu = y_pred_flu.to(device)
            ys_flu_holder.extend(y_gold_flu.tolist())

            # For third prediction
            preds_comp_holder.extend(y_pred_comp)
            y_pred_comp = y_pred_comp.to(device)
            ys_comp_holder.extend(y_gold_comp.tolist())

            # Calculate Loss
            loss1 = loss_func(y_pred_acc, y_gold_acc)
            loss2 = loss_func(y_pred_flu, y_gold_flu)
            loss3 = loss_func(y_pred_comp, y_gold_comp)
            loss = loss1 + loss2 + loss3
            loss_t1 = loss1.item()  # loss for the item
            loss_t2 = loss2.item()  # loss for the item
            loss_t3 = loss3.item()  # loss for the item

            loss_t = (loss_t1 + loss_t2 + loss_t3) / 3

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # compute the accuracy
            # r_squared = calc_r_squared(y_pred_acc, y_gold_acc)
            # running_acc += (r_squared - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy measures for every minibatch
            # print("loss: {0}, running_loss: {1}, acc: {0}, running_acc: {1}".format(loss_t, running_loss,

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)
        # train_state["train_acc"].append(running_acc)

        # print(ys_holder)
        # print(preds_holder)

        r_squared = calc_r_squared(ys_acc_holder, preds_acc_holder)

        train_state["train_acc_r2"].append(r_squared)
        # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
        print("Training r^2: " + str(r_squared))

        # Iterate over validation set--put it in a dataloader

        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # reset loss and accuracy to zero
        running_loss = 0.0
        running_acc = 0.0

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_acc_holder = []
        preds_acc_holder = []
        ys_flu_holder = []
        preds_flu_holder = []
        ys_comp_holder = []
        preds_comp_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # get the gold labels
            y_gold_acc = batch['acc'].to(device)
            y_gold_flu = batch['flu'].to(device)
            y_gold_comp = batch['comp'].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )

            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            batch_audio = batch['audio'].to(device)
            batch_length = batch['length']

            batch_acoustic = batch['acoustic'].to(device)
            batch_acoustic_length = batch['acoustic_length']

            y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                audio_input=batch_audio,
                audio_length=batch_length,
                acoustic_input=batch_acoustic,
                acoustic_length=batch_acoustic_length)

            # print("pred_size: ", y_pred.size())
            # For first prediction
            preds_acc_holder.extend(y_pred_acc)
            y_pred_acc = y_pred_acc.to(device)
            ys_acc_holder.extend(y_gold_acc.tolist())

            # For second prediction
            preds_flu_holder.extend(y_pred_flu)
            y_pred_flu = y_pred_flu.to(device)
            ys_flu_holder.extend(y_gold_flu.tolist())

            # For third prediction
            preds_comp_holder.extend(y_pred_comp)
            y_pred_comp = y_pred_comp.to(device)
            ys_comp_holder.extend(y_gold_comp.tolist())

            # Calculate Loss
            loss1 = loss_func(y_pred_acc, y_gold_acc)
            loss2 = loss_func(y_pred_flu, y_gold_flu)
            loss3 = loss_func(y_pred_comp, y_gold_comp)
            loss = loss1 + loss2 + loss3
            loss_t1 = loss1.item()  # loss for the item
            loss_t2 = loss2.item()  # loss for the item
            loss_t3 = loss3.item()  # loss for the item

            loss_t = (loss_t1 + loss_t2 + loss_t3) / 3

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute the accuracy
            # acc_t = torch.eq(y_pred_sarc_class, y_gold_sarc).sum().item() / len(y_gold_sarc)

            # running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy measures for every minibatch
            # print("loss: {0}, running_loss: {1}, acc: {0}, running_acc: {1}".format(loss_t, running_loss,

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        r_squared = calc_r_squared(ys_acc_holder, preds_acc_holder)
        train_state["val_acc_r2"].append(r_squared)
        print("val r^2: " + str(r_squared))

        # get confusion matrix

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

        return ys_acc_holder, preds_acc_holder

def train_and_predict_single_cv(
        classifier,
        train_state,
        feats,
        rating,
        cv_idx,
        train_ds,
        val_ds,
        batch_size,
        num_epochs,
        loss_func,
        optimizer,
        device,
        rnn,
        model_type,
        lr,
        scheduler=None,
        sampler=None,
):
    model_name = "{0}_{1}_lr{2}".format(rating, model_type, lr)

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # set classifier(s) to training mode
        classifier.train()

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
        )

        # Iterate over training dataset
        running_loss = 0.0

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # reset the gradients
            optimizer.zero_grad()

            if rating == "acc":
                y_gold = batch["acc"].clone().detach().to(torch.float).to(device)
            elif rating == "flu":
                y_gold = batch["flu"].clone().detach().to(torch.float).to(device)
            elif rating == "comp":
                y_gold = batch["comp"].clone().detach().to(torch.float).to(device)
            else:
                print("Wrong response data")
                exit()
            # print(y_gold.size(), y_gold)
            # y_gold = y_gold.squeeze(1).to(torch.float)

            # step 2. select input and compute output

            if feats == "audio":
                batch_feat = batch['audio'].clone().detach().to(device)
                batch_feat = batch_feat.transpose(1, 2)

                batch_length = batch['length'].clone().detach()

                y_pred = classifier(
                    input_features=batch_feat,
                    input_length=batch_length
                )

            elif feats == "acoustic":
                batch_feat = batch["acoustic"].clone().detach().to(device)
                # batch_length = batch['acoustic_length'].clone().detach().to(device)
                #
                # y_pred = classifier(
                #     input_features=batch_feat,
                #     input_length=batch_length
                # )

                y_pred = classifier(
                    input_features=batch_feat
                )
                # y_pred = y_pred.squeeze(1).to(device)

            elif feats == "rhythm":
                batch_feat = batch['rhythm'].clone().detach().to(device)
                y_pred = classifier(
                    input_features=batch_feat
                )
            else:
                print("Wrong input feature")
                sys.exit(1)

            y_pred = y_pred.squeeze(1).to(device)
            # add ys to holder for error analysis
            preds_holder.extend(y_pred.tolist())
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)/
            # print(f"y-pred shape is: {y_pred.shape}")
            loss_t = loss_func(y_pred, y_gold)
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # loss = torch.sqrt(loss + eps)
            # step 4. use loss to produce gradients
            loss_t.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

        # add loss and accuracy information to the train state
        # epoch_loss = mean_squared_error(preds_holder, ys_holder)
        # epoch_loss = loss_func(preds_holder, ys_holder)
        epoch_loss = running_loss
        # epoch_r2 = r2_score(ys_holder, preds_holder)
        epoch_r2 = calc_r_squared(ys_holder, preds_holder)
        # epoch_r2 = (1 - (1 - epoch_r2)) * ((len(batch_feat) - 1) / (len(batch_feat) - len(batch_feat[0]) - 1))
        train_state["train_loss"].append(epoch_loss)
        train_state["train_r2"].append(epoch_r2)

        print("Epoch {0}\tTrain R2: {1}\tTrain Loss: {2}".format(epoch_index, epoch_r2, epoch_loss))
        # print("Training r2: " + str(epoch_r2))

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # set classifier to evaluation mode
        classifier.eval()
        running_loss = 0.0

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the dataloader
        for batch_index, batch in enumerate(val_batches):
            if rating == "acc":
                y_gold = batch["acc"].clone().detach().to(torch.float).to(device)
            elif rating == "flu":
                y_gold = batch["flu"].clone().detach().to(torch.float).to(device)
            elif rating == "comp":
                y_gold = batch["comp"].clone().detach().to(torch.float).to(device)
            else:
                print("Wrong response data")
                exit()
                # print(y_gold.size(), y_gold)
                # y_gold = y_gold.squeeze(1).to(torch.float)

                # step 2. select input and compute output

            if feats == "audio":
                batch_feat = batch['audio'].clone().detach().to(device)
                batch_feat = batch_feat.transpose(1, 2)

                batch_length = batch['length'].clone().detach()

                y_pred = classifier(
                    input_features=batch_feat,
                    input_length=batch_length
                )

            elif feats == "acoustic":
                batch_feat = batch["acoustic"].clone().detach().to(device)
                batch_length = batch['acoustic_length'].clone().detach()

                y_pred = classifier(
                    input_features=batch_feat # ,
                    # input_length=batch_length
                )
                # y_pred = y_pred.squeeze(1).to(device)

            elif feats == "rhythm":
                batch_feat = batch['rhythm'].clone().detach().to(device)
                y_pred = classifier(
                    input_features=batch_feat
                )
            else:
                print("Wrong input feature")
                exit()

            # uncomment for prediction spot-checking during training
            # if epoch_index % 10 == 0:
            #     print(y_pred)
            #     print(y_gold)
            # if epoch_index == 35:
            #     sys.exit(1)
            # print("THE PREDICTIONS ARE: ")
            # print(y_pred)
            # print(y_gold)

            y_pred = y_pred.squeeze(1).to(device)
            # add ys to holder for error analysis
            preds_holder.extend(y_pred.tolist())
            ys_holder.extend(y_gold.tolist())

            loss_t = loss_func(y_pred, y_gold)
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)
            # print(f"y-pred shape is: {y_pred.shape}")

            # uncomment to see loss and accuracy for each minibatch
            # print("val_loss: {0}, running_val_loss: {1}, val_acc: {0}, running_val_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        # epoch_val_r2 = r2_score(ys_holder, preds_holder)
        epoch_val_r2 = calc_r_squared(ys_holder, preds_holder)
        # epoch_val_r2 = (1 - (1 - epoch_val_r2)) * ((len(batch_feat) - 1) / (len(batch_feat) - len(batch_feat[0]) - 1))
        # epoch_val_loss = mean_squared_error(preds_holder, ys_holder)
        epoch_val_loss = running_loss
        train_state["val_r2"].append(epoch_val_r2)
        print("Epoch {0}\tDev R2: {1}\tDev Loss: {2}".format(epoch_index, epoch_val_r2, epoch_val_loss))
        # print("dev r2: " + str(epoch_val_r2))

        # add loss and accuracy to train state
        train_state["val_loss"].append(epoch_val_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

def train_and_predict_embrace_single_cv(
        classifier,
        train_state,
        feats,
        rating,
        cv_idx,
        train_ds,
        val_ds,
        batch_size,
        num_epochs,
        loss_func,
        optimizer,
        device,
        rnn,
        model_type,
        lr,
        scheduler=None,
        sampler=None,
):
    model_name = "{0}_{1}_lr{2}".format(rating, model_type, lr)

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # set classifier(s) to training mode
        classifier.train()

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
        )

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # reset the gradients
            optimizer.zero_grad()

            if rating == "acc":
                y_gold = batch["acc"].clone().detach().to(torch.float).to(device)
            elif rating == "flu":
                y_gold = batch["flu"].clone().detach().to(torch.float).to(device)
            elif rating == "comp":
                y_gold = batch["comp"].clone().detach().to(torch.float).to(device)
            else:
                print("Wrong response data")
                exit()
            # print(y_gold.size(), y_gold)
            # y_gold = y_gold.squeeze(1).to(torch.float)

            # step 2. select input and compute output

            if feats == "AudioAcoustic":
                audio_batch_feat = batch['audio'].clone().detach().to(device)
                audio_batch_feat = audio_batch_feat.transpose(1, 2)

                audio_batch_length = batch['length'].clone().detach()

                acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
                # acoustic_batch_length = batch['acoustic_length'].clone().detach()

                y_pred = classifier(
                    feats=feats,
                    input_features=[audio_batch_feat, acoustic_batch_feat],
                    input_lengths=audio_batch_length
                )

            elif feats == "AudioPhon":
                audio_batch_feat = batch["acoustic"].clone().detach().to(device)
                audio_batch_feat = audio_batch_feat.transpose(1, 2)
                audio_batch_length = batch['acoustic_length'].clone().detach()

                rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

                y_pred = classifier(
                    feats=feats,
                    input_features=[audio_batch_feat, rhythm_batch_feat],
                    input_lengths=audio_batch_length
                )

            elif feats == "AcousticPhon":
                acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
                acoustic_batch_length = batch['acoustic_length'].clone().detach()

                rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

                y_pred = classifier(
                    feats=feats,
                    input_features=[acoustic_batch_feat, rhythm_batch_feat],
                    input_lengths=acoustic_batch_length
                )
            elif feats == "All":
                audio_batch_feat = batch['audio'].clone().detach().to(device)
                audio_batch_feat = audio_batch_feat.transpose(1, 2)

                audio_batch_length = batch['length'].clone().detach()

                acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
                # acoustic_batch_length = batch['acoustic_length'].clone().detach()

                rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

                y_pred = classifier(
                    feats=feats,
                    input_features=[audio_batch_feat, acoustic_batch_feat, rhythm_batch_feat],
                    input_lengths=audio_batch_length
                )

            else:
                print("Wrong input feature")
                exit()

            y_pred = y_pred.squeeze(1).to(device)
            # add ys to holder for error analysis
            preds_holder.extend(y_pred.tolist())
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)
            # print(f"y-pred shape is: {y_pred.shape}")
            loss = loss_func(y_pred, y_gold)



            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

        # add loss and accuracy information to the train state
        epoch_loss = mean_squared_error(ys_holder, preds_holder)
        # epoch_r2 = r2_score(ys_holder, preds_holder)
        epoch_r2 = calc_r_squared(ys_holder, preds_holder)
        train_state["train_loss"].append(epoch_loss)
        train_state["train_r2"].append(epoch_r2)

        print("Epoch {0}\tTrain R2: {1}\tTrain Loss: {2}".format(epoch_index, epoch_r2, epoch_loss))
        # print("Training r2: " + str(epoch_r2))

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the dataloader
        for batch_index, batch in enumerate(val_batches):
            if rating == "acc":
                y_gold = batch["acc"].clone().detach().to(torch.float).to(device)
            elif rating == "flu":
                y_gold = batch["flu"].clone().detach().to(torch.float).to(device)
            elif rating == "comp":
                y_gold = batch["comp"].clone().detach().to(torch.float).to(device)
            else:
                print("Wrong response data")
                exit()
                # print(y_gold.size(), y_gold)
                # y_gold = y_gold.squeeze(1).to(torch.float)

                # step 2. select input and compute output

            if feats == "AudioAcoustic":
                audio_batch_feat = batch['audio'].clone().detach().to(device)
                audio_batch_feat = audio_batch_feat.transpose(1, 2)

                audio_batch_length = batch['length'].clone().detach()

                acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
                # acoustic_batch_length = batch['acoustic_length'].clone().detach()

                y_pred = classifier(
                    feats=feats,
                    input_features=[audio_batch_feat, acoustic_batch_feat],
                    input_lengths=audio_batch_length
                )

            elif feats == "AudioPhon":
                audio_batch_feat = batch["acoustic"].clone().detach().to(device)
                audio_batch_feat = audio_batch_feat.transpose(1, 2)
                audio_batch_length = batch['acoustic_length'].clone().detach()

                rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

                y_pred = classifier(
                    feats=feats,
                    input_features=[audio_batch_feat, rhythm_batch_feat],
                    input_lengths=audio_batch_length
                )

            elif feats == "AcousticPhon":
                acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
                acoustic_batch_length = batch['acoustic_length'].clone().detach()

                rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

                y_pred = classifier(
                    feats=feats,
                    input_features=[acoustic_batch_feat, rhythm_batch_feat],
                    input_lengths=acoustic_batch_length
                )
            elif feats == "All":
                audio_batch_feat = batch['audio'].clone().detach().to(device)
                audio_batch_feat = audio_batch_feat.transpose(1, 2)

                audio_batch_length = batch['length'].clone().detach()

                acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
                # acoustic_batch_length = batch['acoustic_length'].clone().detach()

                rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

                y_pred = classifier(
                    feats=feats,
                    input_features=[audio_batch_feat, acoustic_batch_feat, rhythm_batch_feat],
                    input_lengths=audio_batch_length
                )

            else:
                print("Wrong input feature")
                exit()

            # uncomment for prediction spot-checking during training
            # if epoch_index % 10 == 0:
            #     print(y_pred)
            #     print(y_gold)
            # if epoch_index == 35:
            #     sys.exit(1)
            # print("THE PREDICTIONS ARE: ")
            # print(y_pred)
            # print(y_gold)

            # add ys to holder for error analysis
            preds_holder.extend(y_pred.tolist())
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)
            # print(f"y-pred shape is: {y_pred.shape}")

            # uncomment to see loss and accuracy for each minibatch
            # print("val_loss: {0}, running_val_loss: {1}, val_acc: {0}, running_val_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        # epoch_val_r2 = r2_score(ys_holder, preds_holder)
        epoch_val_r2 = r2_score(ys_holder, preds_holder)
        epoch_val_loss = mean_squared_error(ys_holder, preds_holder)
        train_state["val_r2"].append(epoch_val_r2)
        print("Epoch {0}\tDev R2: {1}\tDev Loss: {2}".format(epoch_index, epoch_val_r2, epoch_val_loss))
        # print("dev r2: " + str(epoch_val_r2))

        # add loss and accuracy to train state
        train_state["val_loss"].append(epoch_val_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break


def train_and_predict_single_multi_cv(
        classifier,
        train_state,
        feats,
        rating,
        cv_idx,
        train_ds,
        val_ds,
        batch_size,
        num_epochs,
        loss_func,
        optimizer,
        device,
        rnn,
        model_type,
        lr,
        scheduler=None,
        sampler=None,
):
    model_name = "{0}_{1}_lr{2}".format(rating, model_type, lr)

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # set classifier(s) to training mode
        classifier.train()

        running_loss = 0.0

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
        )

        # set holders to use for error analysis
        ys_acc_holder = []
        preds_acc_holder = []
        ys_flu_holder = []
        preds_flu_holder = []
        ys_comp_holder = []
        preds_comp_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # reset the gradients
            optimizer.zero_grad()

            y_gold_acc = batch["acc"].clone().detach().to(torch.float).to(device)
            y_gold_flu = batch["flu"].clone().detach().to(torch.float).to(device)
            y_gold_comp = batch["comp"].clone().detach().to(torch.float).to(device)

            # print(y_gold.size(), y_gold)
            # y_gold = y_gold.squeeze(1).to(torch.float)

            # step 2. select input and compute output

            if feats == "audio":
                batch_feat = batch['audio'].clone().detach().to(device)
                batch_feat = batch_feat.transpose(1, 2)

                batch_length = batch['length'].clone().detach()

                y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                    input_features=batch_feat,
                    input_length=batch_length
                )

            elif feats == "acoustic":
                batch_feat = batch["acoustic"].clone().detach().to(device)
                batch_length = batch['acoustic_length'].clone().detach()

                y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                    input_features=batch_feat
                )
                y_pred_acc = y_pred_acc.squeeze(1).to(device)
                y_pred_flu = y_pred_flu.squeeze(1).to(device)
                y_pred_comp = y_pred_comp.squeeze(1).to(device)

            elif feats == "rhythm":
                batch_feat = batch['rhythm'].clone().detach().to(device)
                y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                    input_features=batch_feat
                )
            else:
                print("Wrong input feature")
                exit()

            # For first prediction
            y_pred_acc = y_pred_acc.squeeze(1).to(device)
            preds_acc_holder.extend(y_pred_acc.tolist())
            ys_acc_holder.extend(y_gold_acc.tolist())

            # For second prediction
            y_pred_flu = y_pred_flu.squeeze(1).to(device)
            preds_flu_holder.extend(y_pred_flu.tolist())
            ys_flu_holder.extend(y_gold_flu.tolist())

            # For third prediction
            y_pred_comp = y_pred_comp.squeeze(1).to(device)
            preds_comp_holder.extend(y_pred_comp.tolist())
            ys_comp_holder.extend(y_gold_comp.tolist())

            # step 3. compute the loss
            # Calculate Loss
            loss1 = loss_func(y_pred_acc, y_gold_acc)
            loss2 = loss_func(y_pred_flu, y_gold_flu)
            loss3 = loss_func(y_pred_comp, y_gold_comp)
            loss_t = (loss1 + loss2 + loss3) / 3

            # step 4. use loss to produce gradients
            loss_t.backward()

            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 5. use optimizer to take gradient step
            optimizer.step()

        # add loss and accuracy information to the train state

        if rating == "acc":
            epoch_loss = mean_squared_error(ys_acc_holder, preds_acc_holder)
            epoch_r2 = calc_r_squared(ys_acc_holder, preds_acc_holder)
        elif rating == "flu":
            epoch_loss = mean_squared_error(ys_flu_holder, preds_flu_holder)
            epoch_r2 = calc_r_squared(ys_flu_holder, preds_flu_holder)
        elif rating == "comp":
            epoch_loss = mean_squared_error(ys_comp_holder, preds_comp_holder)
            epoch_r2 = calc_r_squared(ys_comp_holder, preds_comp_holder)
        else:
            print("Wrong rating!")
            sys.exit(1)

        # train_state["train_loss"].append(epoch_loss)
        train_state["train_loss"].append(running_loss)
        train_state["train_r2"].append(epoch_r2)

        print("Epoch {0}\tTrain R2: {1}\tTrain Loss: {2}".format(epoch_index, epoch_r2, epoch_loss))
        # print("Training r2: " + str(epoch_r2))

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # set classifier to evaluation mode
        classifier.eval()
        running_loss = 0.0

        # set holders to use for error analysis
        ys_acc_holder = []
        preds_acc_holder = []
        ys_flu_holder = []
        preds_flu_holder = []
        ys_comp_holder = []
        preds_comp_holder = []

        # for each batch in the dataloader
        for batch_index, batch in enumerate(val_batches):
            y_gold_acc = batch["acc"].clone().detach().to(torch.float).to(device)
            y_gold_flu = batch["flu"].clone().detach().to(torch.float).to(device)
            y_gold_comp = batch["comp"].clone().detach().to(torch.float).to(device)

            # print(y_gold.size(), y_gold)
            # y_gold = y_gold.squeeze(1).to(torch.float)

            # step 2. select input and compute output

            if feats == "audio":
                batch_feat = batch['audio'].clone().detach().to(device)
                batch_feat = batch_feat.transpose(1, 2)

                batch_length = batch['length'].clone().detach()

                y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                    input_features=batch_feat,
                    input_length=batch_length
                )

            elif feats == "acoustic":
                batch_feat = batch["acoustic"].clone().detach().to(device)
                batch_length = batch['acoustic_length'].clone().detach()

                y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                    input_features=batch_feat # ,
                    # input_length=batch_length
                )
                y_pred_acc = y_pred_acc.squeeze(1).to(device)
                y_pred_flu = y_pred_flu.squeeze(1).to(device)
                y_pred_comp = y_pred_comp.squeeze(1).to(device)

            elif feats == "rhythm":
                batch_feat = batch['rhythm'].clone().detach().to(device)
                y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                    input_features=batch_feat
                )
            else:
                print("Wrong input feature")
                exit()

            # For first prediction
            y_pred_acc = y_pred_acc.squeeze(1).to(device)
            preds_acc_holder.extend(y_pred_acc.tolist())
            ys_acc_holder.extend(y_gold_acc.tolist())

            # For second prediction
            y_pred_flu = y_pred_flu.squeeze(1).to(device)
            preds_flu_holder.extend(y_pred_flu.tolist())
            ys_flu_holder.extend(y_gold_flu.tolist())

            # For third prediction
            y_pred_comp = y_pred_comp.squeeze(1).to(device)
            preds_comp_holder.extend(y_pred_comp.tolist())
            ys_comp_holder.extend(y_gold_comp.tolist())

            loss1 = loss_func(y_pred_acc, y_gold_acc)
            loss2 = loss_func(y_pred_flu, y_gold_flu)
            loss3 = loss_func(y_pred_comp, y_gold_comp)
            loss_t = (loss1 + loss2 + loss3) / 3

            running_loss += (loss_t - running_loss) / (batch_index + 1)

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        if rating == "acc":
            epoch_val_loss = mean_squared_error(ys_acc_holder, preds_acc_holder)
            epoch_val_r2 = calc_r_squared(ys_acc_holder, preds_acc_holder)
        elif rating == "flu":
            epoch_val_loss = mean_squared_error(ys_flu_holder, preds_flu_holder)
            epoch_val_r2 = calc_r_squared(ys_flu_holder, preds_flu_holder)
        elif rating == "comp":
            epoch_val_loss = mean_squared_error(ys_comp_holder, preds_comp_holder)
            epoch_val_r2 = calc_r_squared(ys_comp_holder, preds_comp_holder)
        else:
            print("Wrong rating!")
            sys.exit(1)

        train_state["val_r2"].append(epoch_val_r2)
        print("Epoch {0}\tDev R2: {1}\tDev Loss: {2}".format(epoch_index, epoch_val_r2, epoch_val_loss))
        # add loss and accuracy to train state
        # train_state["val_loss"].append(epoch_val_loss)
        train_state["val_loss"].append(running_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break


def train_and_predict_transformer(
        roberta,
        classifier,
        train_state,
        train_ds,
        val_ds,
        rating,
        # batch_size,
        # num_workers,
        num_epochs,
        loss_func,
        optimizer,
        device,
        scheduler=None,
        sampler=None,
        binary=False,
        split_point=0.0,
):
    # print(classifier)
    for epoch_index in range(num_epochs):
        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # set classifier(s) to training mode
        roberta.train()
        classifier.train()

        # batches = DataLoader(
        #     train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, sampler=sampler
        # )

        # assign holders
        ys_holder = []
        preds_holder = []

        # for batch in batches:
        for (batch_index, batch) in enumerate(train_ds):
            # reset the gradients
            optimizer.zero_grad()

            # Load batch
            input_ids = batch['token'].type(torch.LongTensor)
            # print(input_ids.size())
            # print(batch['length'])
            # attention_mask = batch['length'].to(device)
            if rating == "acc":
                labels = torch.as_tensor(batch['acc']).to(device)
            elif rating == "flu":
                labels = torch.as_tensor(batch['flu']).to(device)
            elif rating == "comp":
                labels = torch.as_tensor(batch['comp']).to(device)
            # print(labels.size())
            # outputs = classifier(input_ids, attention_mask=attention_mask, labels=labels)
            # outputs = classifier(input_ids, labels=labels)

            # outputs = classifier.predict('apj', input_ids.to(device))
            outputs = roberta(input_ids)
            last_hidden_states = outputs.pooler_output
            y_preds = classifier(
                input_features=last_hidden_states
            )
            y_pred = y_pred.squeeze(1).to(device)

            # print(preds)
            # print(labels)

            ys_holder.extend(labels.tolist())
            preds_holder.extend(y_pred.tolist())

            loss = loss_func(y_pred, labels)

            # backprop. loss
            loss.backward()

            # optimizer for the gradient step
            optimizer.step()

        # add loss and accuracy information to the train state
        epoch_loss = mean_squared_error(ys_holder, preds_holder)
        # epoch_r2 = r2_score(ys_holder, preds_holder)
        epoch_r2 = calc_r_squared(ys_holder, preds_holder)
        train_state["train_loss"].append(epoch_loss)
        train_state["train_r2"].append(epoch_r2)

        print("Epoch {0}\tTrain R2: {1}\tTrain Loss: {2}".format(epoch_index, epoch_r2, epoch_loss))
        # print("Training r2: " + str(epoch_r2))

        # Iterate over dev dataset
        running_loss = 0.0
        running_acc = 0.0

        # set classifier(s) to evaluation mode
        roberta.eval()
        classifier.eval()

        # batches = DataLoader(
        #     val_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, sampler=sampler
        # )

        # assign holders
        ys_holder = []
        preds_holder = []

        # for batch in batches:
        for (batch_index, batch) in enumerate(val_ds):
            # Load batch
            input_ids = batch['token'].type(torch.LongTensor)
            # print(input_ids.size())
            # print(batch['length'])
            # attention_mask = batch['length'].to(device)
            if rating == "acc":
                labels = torch.as_tensor(batch['acc']).to(device)
            elif rating == "flu":
                labels = torch.as_tensor(batch['flu']).to(device)
            elif rating == "comp":
                labels = torch.as_tensor(batch['comp']).to(device)
            # print(labels.size())
            # outputs = classifier(input_ids, attention_mask=attention_mask, labels=labels)
            # outputs = classifier(input_ids, labels=labels)

            # outputs = classifier.predict('apj', input_ids.to(device))
            outputs = roberta(input_ids)
            last_hidden_states = outputs.last_hidden_states
            y_preds = classifier(
                input_features=last_hidden_states
            )
            y_pred = y_pred.squeeze(1).to(device)

            # print(preds)
            # print(labels)

            ys_holder.extend(labels.tolist())
            preds_holder.extend(y_pred.tolist())

        # epoch_val_r2 = r2_score(ys_holder, preds_holder)
        epoch_val_r2 = r2_score(ys_holder, preds_holder)
        epoch_val_loss = mean_squared_error(ys_holder, preds_holder)
        train_state["val_r2"].append(epoch_val_r2)
        print("Epoch {0}\tDev R2: {1}\tDev Loss: {2}".format(epoch_index, epoch_val_r2, epoch_val_loss))
        # print("dev r2: " + str(epoch_val_r2))

        # add loss and accuracy to train state
        train_state["val_loss"].append(epoch_val_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break


def test_model_cv(
        classifier,
        test_ds,
        cv_idx,
        batch_size,
        rating,
        feats,
        result_file,
        model_type,
        lr,
        loss_func,
        device="cpu",
        gold_labels=[],
        pred_labels=[]
):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    val_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # reset loss and accuracy to zero

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    ys_holder = []
    preds_holder = []

    running_loss = 0
    # for each batch in the dataloader
    classifier.eval()

    for batch_index, batch in enumerate(val_batches):
        if rating == "acc":
            y_gold = batch["acc"].clone().detach().to(torch.float).to(device)
        elif rating == "flu":
            y_gold = batch["flu"].clone().detach().to(torch.float).to(device)
        elif rating == "comp":
            y_gold = batch["comp"].clone().detach().to(torch.float).to(device)
        else:
            print("Wrong response data")
            exit()
            # print(y_gold.size(), y_gold)
            # y_gold = y_gold.squeeze(1).to(torch.float)

            # step 2. select input and compute output

        if feats == "audio":
            batch_feat = batch['audio'].clone().detach().to(device)
            batch_feat = batch_feat.transpose(1, 2)

            batch_length = batch['length'].clone().detach()

            y_pred = classifier(
                input_features=batch_feat,
                input_length=batch_length
            )

        elif feats == "acoustic":
            batch_feat = batch["acoustic"].clone().detach().to(device)
            # batch_length = batch['acoustic_length'].clone().detach().to(device)

            y_pred = classifier(
                input_features=batch_feat # ,
                # input_length=batch_length
            )

        elif feats == "rhythm":
            batch_feat = batch['rhythm'].clone().detach().to(device)
            y_pred = classifier(
                input_features=batch_feat
            )
        else:
            print("Wrong input feature")
            exit()

        y_pred = y_pred.squeeze(1).to(device)

        # add ys to holder for error analysis
        preds_holder.extend(y_pred.tolist())
        pred_labels.extend(y_pred.tolist())
        ys_holder.extend(y_gold.tolist())
        gold_labels.extend(y_gold.tolist())

        loss_t = loss_func(y_pred, y_gold)
        running_loss += (loss_t - running_loss) / (batch_index + 1)

    result_file_cv = result_file.replace(".csv", "_cv.csv")
    # cv_loss = mean_squared_error(ys_holder, preds_holder)
    cv_loss = running_loss
    # cv_r2 = r2_score(ys_holder, preds_holder)
    cv_r2 = calc_r_squared(ys_holder, preds_holder)
    # cv_r2 = (1 - (1 - cv_r2)) * ((len(batch_feat) - 1) / (len(batch_feat) - len(batch_feat[0]) - 1))
    cv_result = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(model_type, cv_idx, feats, rating, lr, cv_r2, cv_loss)

    with open(result_file_cv, "a") as cv_out:
        cv_out.write(cv_result)

def test_model_multi_single_cv(
        classifier,
        test_ds,
        cv_idx,
        batch_size,
        rating,
        feats,
        result_file,
        model_type,
        lr,
        device="cpu",
        gold_labels=[],
        pred_labels=[]
):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    val_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # reset loss and accuracy to zero

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    ys_holder = []
    preds_holder = []

    # for each batch in the dataloader
    classifier.eval()

    for batch_index, batch in enumerate(val_batches):
        if rating == "acc":
            y_gold = batch["acc"].clone().detach().to(torch.float).to(device)
        elif rating == "flu":
            y_gold = batch["flu"].clone().detach().to(torch.float).to(device)
        elif rating == "comp":
            y_gold = batch["comp"].clone().detach().to(torch.float).to(device)
        else:
            print("Wrong response data")
            exit()
            # print(y_gold.size(), y_gold)
            # y_gold = y_gold.squeeze(1).to(torch.float)

            # step 2. select input and compute output

        if feats == "AudioAcoustic":
            audio_batch_feat = batch['audio'].clone().detach().to(device)
            audio_batch_feat = audio_batch_feat.transpose(1, 2)

            audio_batch_length = batch['length'].clone().detach()

            acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
            # acoustic_batch_length = batch['acoustic_length'].clone().detach()

            y_pred = classifier(
                feats=feats,
                input_features=[audio_batch_feat, acoustic_batch_feat],
                input_lengths=audio_batch_length
            )

        elif feats == "AudioPhon":
            audio_batch_feat = batch["acoustic"].clone().detach().to(device)
            audio_batch_feat = audio_batch_feat.transpose(1, 2)
            audio_batch_length = batch['acoustic_length'].clone().detach()

            rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

            y_pred = classifier(
                feats=feats,
                input_features=[audio_batch_feat, rhythm_batch_feat],
                input_lengths=audio_batch_length
            )

        elif feats == "AcousticPhon":
            acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
            acoustic_batch_length = batch['acoustic_length'].clone().detach()

            rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

            y_pred = classifier(
                feats=feats,
                input_features=[acoustic_batch_feat, rhythm_batch_feat],
                input_lengths=acoustic_batch_length
            )
        elif feats == "All":
            audio_batch_feat = batch['audio'].clone().detach().to(device)
            audio_batch_feat = audio_batch_feat.transpose(1, 2)

            audio_batch_length = batch['length'].clone().detach()

            acoustic_batch_feat = batch["acoustic"].clone().detach().to(device)
            # acoustic_batch_length = batch['acoustic_length'].clone().detach()

            rhythm_batch_feat = batch['rhythm'].clone().detach().to(device)

            y_pred = classifier(
                feats=feats,
                input_features=[audio_batch_feat, acoustic_batch_feat, rhythm_batch_feat],
                input_lengths=audio_batch_length
            )

        else:
            print("Wrong input feature")
            exit()

        y_pred = y_pred.squeeze(1).to(device)

        # add ys to holder for error analysis
        preds_holder.extend(y_pred.tolist())
        pred_labels.extend(y_pred.tolist())
        ys_holder.extend(y_gold.tolist())
        gold_labels.extend(y_gold.tolist())

    result_file_cv = result_file.replace(".csv", "_cv.csv")
    cv_loss = mean_squared_error(ys_holder, preds_holder)
    # cv_r2 = r2_score(ys_holder, preds_holder)
    cv_r2 = calc_r_squared(ys_holder, preds_holder)
    cv_result = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(model_type, cv_idx, feats, rating, lr, cv_r2, cv_loss)

    with open(result_file_cv, "a") as cv_out:
        cv_out.write(cv_result)


def test_model_single_multi_cv(
        classifier,
        test_ds,
        cv_idx,
        batch_size,
        rating,
        feats,
        result_file,
        model_type,
        lr,
        device="cpu",
        gold_labels=[],
        pred_labels=[]
):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    val_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # reset loss and accuracy to zero

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    ys_acc_holder = []
    preds_acc_holder = []
    ys_flu_holder = []
    preds_flu_holder = []
    ys_comp_holder = []
    preds_comp_holder = []

    # for each batch in the dataloader
    for batch_index, batch in enumerate(val_batches):
        y_gold_acc = batch["acc"].clone().detach().to(torch.float).to(device)
        y_gold_flu = batch["flu"].clone().detach().to(torch.float).to(device)
        y_gold_comp = batch["comp"].clone().detach().to(torch.float).to(device)

        # print(y_gold.size(), y_gold)
        # y_gold = y_gold.squeeze(1).to(torch.float)

        # step 2. select input and compute output

        if feats == "audio":
            batch_feat = batch['audio'].clone().detach().to(device)
            batch_feat = batch_feat.transpose(1, 2)

            batch_length = batch['length'].clone().detach()

            y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                input_features=batch_feat,
                input_length=batch_length
            )

        elif feats == "acoustic":
            batch_feat = batch["acoustic"].clone().detach().to(device)
            batch_length = batch['acoustic_length'].clone().detach()

            y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                input_features=batch_feat # ,
                # input_length=batch_length
            )
            y_pred_acc = y_pred_acc.squeeze(1).to(device)
            y_pred_flu = y_pred_flu.squeeze(1).to(device)
            y_pred_comp = y_pred_comp.squeeze(1).to(device)

        elif feats == "rhythm":
            batch_feat = batch['rhythm'].clone().detach().to(device)
            y_pred_acc, y_pred_flu, y_pred_comp = classifier(
                input_features=batch_feat
            )
        else:
            print("Wrong input feature")
            exit()

        # For first prediction
        y_pred_acc = y_pred_acc.squeeze(1).to(device)
        preds_acc_holder.extend(y_pred_acc.tolist())
        ys_acc_holder.extend(y_gold_acc.tolist())

        # For second prediction
        y_pred_flu = y_pred_flu.squeeze(1).to(device)
        preds_flu_holder.extend(y_pred_flu.tolist())
        ys_flu_holder.extend(y_gold_flu.tolist())

        # For third prediction
        y_pred_comp = y_pred_comp.squeeze(1).to(device)
        preds_comp_holder.extend(y_pred_comp.tolist())
        ys_comp_holder.extend(y_gold_comp.tolist())

        if rating == "acc":
            pred_labels.extend(y_pred_acc.tolist())
            gold_labels.extend(y_gold_acc.tolist())
        elif rating == "flu":
            pred_labels.extend(y_pred_flu.tolist())
            gold_labels.extend(y_gold_flu.tolist())
        elif rating == "comp":
            pred_labels.extend(y_pred_comp.tolist())
            gold_labels.extend(y_gold_comp.tolist())

    # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
    if rating == "acc":
        cv_loss = mean_squared_error(ys_acc_holder, preds_acc_holder)
        cv_r2 = calc_r_squared(ys_acc_holder, preds_acc_holder)
    elif rating == "flu":
        cv_loss = mean_squared_error(ys_flu_holder, preds_flu_holder)
        cv_r2 = calc_r_squared(ys_flu_holder, preds_flu_holder)
    elif rating == "comp":
        cv_loss = mean_squared_error(ys_comp_holder, preds_comp_holder)
        cv_r2 = calc_r_squared(ys_comp_holder, preds_comp_holder)
    else:
        print("Wrong rating!")
        sys.exit(1)

    result_file_cv = result_file.replace(".csv", "_cv.csv")
    # cv_loss = mean_squared_error(ys_holder, preds_holder)
    # cv_r2 = r2_score(ys_holder, preds_holder)
    # cv_r2 = calc_r_squared(ys_holder, preds_holder)
    cv_result = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(model_type, cv_idx, feats, rating, lr, cv_r2, cv_loss)

    with open(result_file_cv, "a") as cv_out:
        cv_out.write(cv_result)