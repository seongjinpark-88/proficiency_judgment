# implement training and testing for models
import os, sys
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
    ss_res = []
    ss_tot = []
    for i, item in enumerate(valy):
        ss_res.append((item - y_preds[i]) ** 2)
        ss_tot.append((item - mean_y) ** 2)
    r_value = (1 - sum(ss_res) / (sum(ss_tot) + 0.0000001))
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
                # train_state['best_val_acc'] = train_state['val_acc'][-1]

            # Reset early stopping step
            train_state["early_stopping_step"] = 0

        # Stop early ?
        train_state["stop_early"] = (
                train_state["early_stopping_step"] >= params.early_stopping_criteria
        )

    return train_state


def train_and_predict(
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
        avgd_acoustic=True,
        use_speaker=True,
        use_gender=False,
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
            # print(batch)
            # print(batch[0])
            # print(batch[1])
            # print(batch[2])
            # print(batch[3])
            # print(batch[4])
            # print(len(batch))
            # sys.exit()
            # get the gold labels
            y_gold = batch[4].to(device)
            # y_gold = batch[7].to(device)  # 4 is emotion, 5 is sentiment

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
            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None

            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                )

            if binary:
                y_pred = y_pred.float()
                y_gold = y_gold.float()

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
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)
            # print(f"y-pred shape is: {y_pred.shape}")
            loss = loss_func(y_pred, y_gold)
            loss_t = loss.item()  # loss for the item

            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    # if type(y_gold[0]) == list or torch.is_tensor(y_gold[0]):
                    #     y_gold = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
                    # print(y_gold)
                    # print(y_pred)
                    # print(type(y_gold))
                    # print(type(y_pred))
            else:
                y_pred = torch.round(y_pred)

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # compute the accuracy
            acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)

            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy measures for every minibatch
            # print("loss: {0}, running_loss: {1}, acc: {0}, running_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

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
            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None

            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                )

            # get the gold labels
            # y_gold = batch[7].to(device)
            y_gold = batch[4].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )
            # y_gold = batch.targets()

            if binary:
                y_pred = y_pred.float()
                y_gold = y_gold.float()

            # add ys to holder for error analysis
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            loss = loss_func(y_pred, y_gold)
            running_loss += (loss.item() - running_loss) / (batch_index + 1)

            # compute the loss
            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
            else:
                y_pred = torch.round(y_pred)

            # compute the accuracy
            acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)
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


def train_and_predict_w2v(
        classifier,
        train_state,
        train_ds,
        val_ds,
        batch_size,
        num_epochs,
        loss_func,
        optimizer,
        rnn=False,
        device="cpu",
        scheduler=None,
        sampler=None,
        binary=False,
        split_point=0.0,
):
    # print(type(train_ds['audio']))
    # print(train_ds['audio'].size())
    # train_ds['audio'] = nn.utils.rnn.pad_sequence(train_ds['audio'])
    # val_ds['audio'] = nn.utils.rnn.pad_sequence(val_ds['audio'])
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
            # batch_acoustic = batch_acoustic.transpose(1, 2)
            # batch_acoustic = nn.utils.rnn.pad_sequence(batch_acoustic)

            if rnn:
                batch_acoustic_lengths = batch['length']
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    acoustic_len_input=batch_acoustic_lengths,
                )
            else:
                batch_acoustic = batch_acoustic.unsqueeze(1)
                y_pred = classifier(
                    acoustic_input=batch_acoustic
                )

            if binary:
                y_pred = y_pred.float()
                y_gold = y_gold.float()

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
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)
            # print(f"y-pred shape is: {y_pred.shape}")
            loss = loss_func(y_pred, y_gold)
            loss_t = loss.item()  # loss for the item

            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    # if type(y_gold[0]) == list or torch.is_tensor(y_gold[0]):
                    #     y_gold = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
                    # print(y_gold)
                    # print(y_pred)
                    # print(type(y_gold))
                    # print(type(y_pred))
            else:
                y_pred = torch.round(y_pred)

            y_pred = y_pred.to(device)
            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # compute the accuracy
            acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)

            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy measures for every minibatch
            # print("loss: {0}, running_loss: {1}, acc: {0}, running_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["train_avg_f1"].append(avg_f1[2])
        # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
        print("Training weighted F-score: " + str(avg_f1))

        torch.cuda.empty_cache()

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
            # batch_acoustic = batch_acoustic.transpose(1, 2)
            # batch_acoustic = nn.utils.rnn.pad_sequence(batch_acoustic)
            if rnn:
                batch_acoustic_lengths = batch['length']

                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    acoustic_len_input=batch_acoustic_lengths,
                )
            else:
                batch_acoustic = batch_acoustic.unsqueeze(1)
                y_pred = classifier(acoustic_input=batch_acoustic)

            # get the gold labels
            # y_gold = batch[7].to(device)
            y_gold = batch['label'].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )
            # y_gold = batch.targets()

            if binary:
                y_pred = y_pred.float()
                y_gold = y_gold.float()

            # add ys to holder for error analysis
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            loss = loss_func(y_pred, y_gold)
            running_loss += (loss.item() - running_loss) / (batch_index + 1)

            # compute the loss
            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
            else:
                y_pred = torch.round(y_pred)
            y_pred = y_pred.to(device)
            # compute the accuracy
            acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)
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


def train_and_predict_multi(
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
            batch_audio = batch['audio'].to(device)
            batch_length = batch['length']

            batch_acoustic = batch['acoustic'].to(device)
            batch_acoustic_length = batch['acoustic_length']
            # batch_acoustic = nn.utils.rnn.pad_sequence(batch_acoustic.squeeze(0)).to(device)
            # print(batch_acoustic.size())

            y_pred = classifier(
                audio_input=batch_audio,
                audio_length=batch_length,
                acoustic_input=batch_acoustic,
                acoustic_length=batch_acoustic_length)

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
            # step 2. compute the output
            batch_audio = batch['audio'].to(device)
            batch_length = batch['length']

            batch_acoustic = batch['acoustic'].to(device)
            batch_acoustic_length = batch['acoustic_length']
            # batch_acoustic = nn.utils.rnn.pad_sequence(batch_acoustic.squeeze(0)).to(device)
            # print(batch_acoustic.size())

            y_gold = batch['label'].to(device)

            y_pred = classifier(
                audio_input=batch_audio,
                audio_length=batch_length,
                acoustic_input=batch_acoustic,
                acoustic_length=batch_acoustic_length)

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


#
# def single_dataset_multitask_train_and_predict(
#     classifier,
#     train_state,
#     train_ds_list,
#     val_ds_list,
#     batch_size,
#     num_epochs,
#     loss_func,
#     optimizer,
#     device="cpu",
#     scheduler=None,
#     sampler=None,
#     avgd_acoustic=True,
#     use_speaker=True,
#     use_gender=False,
# ):
#     """
#     Train_ds_list and val_ds_list are lists of datasets!
#     Each item in the list represents a dataset
#     Length of the list is the number of datasets used
#     todo: this is going to be tested with 2 tasks first, then more
#         it will be brittle at first, so make it more flexible
#     """
#     multi_dataset = (True if len(train_ds_list) > 1 else False)
#     # if len(train_ds_list) > 1:
#     #     multi_dataset = True
#     # else:
#     #     multi_dataset = False
#     num_tasks = len(train_ds_list)
#
#     for epoch_index in range(num_epochs):
#
#         print("Now starting epoch {0}".format(epoch_index))
#
#         train_state["epoch_index"] = epoch_index
#
#         # Iterate over training dataset
#         running_loss = 0.0
#
#         # set classifier(s) to training mode
#         classifier.train()
#
#         task_1_batches = DataLoader(
#             train_ds_list[0], batch_size=batch_size, shuffle=True, sampler=sampler
#         )
#         task_2_batches = DataLoader(
#             train_ds_list[1], batch_size=batch_size, shuffle=True, sampler=sampler
#         )
#
#         # set holders to use for error analysis
#         ys_holder = []
#         ys_2_holder = []
#         preds_holder = []
#         preds_2_holder = []
#
#         # for each batch in the list of batches created by the dataloader
#         for batch_index, batch in enumerate(task_1_batches):
#             # step 1. zero the gradients
#             optimizer.zero_grad()
#
#             # step 2. compute the output
#             y_pred, _, _, y_gold = get_batch_predictions(batch, classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#             _, y_2_pred, _, y_2_gold = get_batch_predictions(task_2_batches[batch_index], classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#
#             # add ys to holder for error analysis
#             preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
#             preds_2_holder.extend([item.index(max(item)) for item in y_2_pred.tolist()])
#             ys_holder.extend(y_gold.tolist())
#             ys_2_holder.extend(y_2_gold.tolist())
#
#             # step 3. compute the loss
#             class_1_loss = loss_func(y_pred, y_gold)
#             class_2_loss = loss_func(y_2_pred, y_2_gold)
#
#             loss = (class_1_loss / 1.6) + class_2_loss
#
#             loss_t = loss.item()  # loss for the item
#
#             # calculate running loss
#             running_loss += (loss_t - running_loss) / (batch_index + 1)
#
#             # step 4. use loss to produce gradients
#             loss.backward()
#
#             # step 5. use optimizer to take gradient step
#             optimizer.step()
#
#         # add loss and accuracy information to the train state
#         train_state["train_loss"].append(running_loss)
#
#         avg_f1 = precision_recall_fscore_support(
#             ys_holder, preds_holder, average="weighted"
#         )
#         train_state["train_avg_f1"].append(avg_f1[2])
#         # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
#         print("Training weighted F=score for EMOTION: " + str(avg_f1))
#
#         # Iterate over validation set--put it in a dataloader
#         val_1_batches = DataLoader(val_ds_list[0], batch_size=batch_size, shuffle=False)
#         val_2_batches = DataLoader(val_ds_list[1], batch_size=batch_size, shuffle=False)
#
#         # reset loss and accuracy to zero
#         running_loss = 0.0
#
#         # set classifier to evaluation mode
#         classifier.eval()
#
#         # set holders to use for error analysis
#         ys_holder = []
#         ys_2_holder = []
#         preds_holder = []
#         preds_2_holder = []
#
#         # for each batch in the dataloader
#         # todo: what if there are different numbers of batches? (diff dataset sizes)
#         for batch_index, batch in enumerate(val_1_batches):
#             # compute the output
#             y_pred, _, _, y_gold = get_batch_predictions(batch, classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#             _, y_2_pred, _, y_2_gold = get_batch_predictions(val_2_batches[batch_index], classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#
#             # add ys to holder for error analysis
#             preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
#             preds_2_holder.extend([item.index(max(item)) for item in y_2_pred.tolist()])
#             ys_holder.extend(y_gold.tolist())
#             ys_2_holder.extend(y_2_gold.tolist())
#
#             class_1_loss = loss_func(y_pred, y_gold)
#             class_2_loss = loss_func(y_2_pred, y_2_gold)
#
#             loss = (class_1_loss / 1.6) + class_2_loss
#
#             # loss = loss_func(ys_pred, ys_gold)
#             running_loss += (loss.item() - running_loss) / (batch_index + 1)
#
#             # uncomment to see loss and accuracy for each minibatch
#             # print("val_loss: {0}, running_val_loss: {1}, val_acc: {0}, running_val_acc: {1}".format(loss_t, running_loss,
#             #                                                                       acc_t, running_acc))
#
#         # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
#         avg_f1 = precision_recall_fscore_support(
#             ys_holder, preds_holder, average="weighted"
#         )
#         train_state["val_avg_f1"].append(avg_f1[2])
#         print("Weighted F=score: " + str(avg_f1))
#
#         # get confusion matrix
#         if epoch_index % 5 == 0:
#             print(confusion_matrix(ys_holder, preds_holder))
#             print("Classification report for EMOTION: ")
#             print(classification_report(ys_holder, preds_holder, digits=4))
#             print("======================================================")
#             print(confusion_matrix(ys_2_holder, preds_2_holder))
#             print("Classification report for SENTIMENT")
#             print(classification_report(ys_2_holder, preds_2_holder, digits=4))
#
#         # add loss and accuracy to train state
#         train_state["val_loss"].append(running_loss)
#
#         # update the train state now that our epoch is complete
#         train_state = update_train_state(model=classifier, train_state=train_state)
#
#         # update scheduler if there is one
#         if scheduler is not None:
#             scheduler.step(train_state["val_loss"][-1])
#
#         # if it's time to stop, end the training process
#         if train_state["stop_early"]:
#             break


def get_batch_predictions(batch, classifier, gold_idx, use_speaker=False,
                          use_gender=True, avgd_acoustic=True, device="cpu"):
    """
    Get the predictions for a batch
    batch: the batch of data from dataloader
    model: the model
    gold_idx: the index of gold labels used within the data batch
    returns predictions and gold labels for the batch
    """
    # get gold labels for the batch
    gold = batch[gold_idx].to(device)

    batch_acoustic = batch[0].to(device)
    batch_text = batch[1].to(device)
    batch_lengths = batch[6].to(device)
    # get acoustic lengths if necessary
    if not avgd_acoustic:
        batch_acoustic_lengths = batch[7].to(device)
    else:
        batch_acoustic_lengths = None
    # get speakers if necessary
    if use_speaker:
        batch_speakers = batch[2].to(device)
    else:
        batch_speakers = None
    # get gender if necessary
    if use_gender:
        batch_genders = batch[3].to(device)
    else:
        batch_genders = None

    y_pred, y_2_pred, y_3_pred = classifier(
        acoustic_input=batch_acoustic,
        text_input=batch_text,
        speaker_input=batch_speakers,
        length_input=batch_lengths,
        acoustic_len_input=batch_acoustic_lengths,
        gender_input=batch_genders,
    )

    return y_pred, y_2_pred, y_3_pred, gold


def test_model(
        classifier,
        test_ds,
        batch_size,
        loss_func,
        device="cpu",
        avgd_acoustic=True,
        use_speaker=True,
        use_gender=False,
):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    val_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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
        batch_acoustic = batch[0].to(device)
        batch_text = batch[1].to(device)
        batch_lengths = batch[-2].to(device)
        batch_acoustic_lengths = batch[-1].to(device)
        if use_speaker:
            batch_speakers = batch[2].to(device)
        else:
            batch_speakers = None

        if use_gender:
            batch_genders = batch[3].to(device)
        else:
            batch_genders = None

        if avgd_acoustic:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                gender_input=batch_genders,
            )
        else:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                acoustic_len_input=batch_acoustic_lengths,
                gender_input=batch_genders,
            )

        # get the gold labels
        y_gold = batch[4].to(device)

        # add ys to holder for error analysis
        preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
        ys_holder.extend(y_gold.tolist())

        y_pred = y_pred.float()
        y_gold = y_gold.float()

        loss = loss_func(y_pred, y_gold)
        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        # compute the loss
        if len(list(y_pred.size())) > 1:
            y_pred = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
        else:
            y_pred = torch.round(y_pred)

        # compute the accuracy
        acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
    avg_f1 = precision_recall_fscore_support(
        ys_holder, preds_holder, average="weighted"
    )

    print("Weighted F=score: " + str(avg_f1))

    # get confusion matrix
    print(confusion_matrix(ys_holder, preds_holder))
    print("Classification report: ")
    print(classification_report(ys_holder, preds_holder, digits=4))


def multitask_train_and_predict(
        classifier,
        train_state,
        datasets_list,
        batch_size,
        num_epochs,
        optimizer,
        device="cpu",
        scheduler=None,
        sampler=None,
        avgd_acoustic=True,
        use_speaker=True,
        use_gender=False,
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    """
    num_tasks = len(datasets_list)
    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["train_avg_f1"][dset.task_num] = []
        train_state["val_avg_f1"][dset.task_num] = []

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches, tasks = get_all_batches(datasets_list, batch_size=batch_size, shuffle=True)

        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # find the task for this batch
            batch_task = tasks[batch_index]
            # print(f"BATCH TASK IS: {batch_task}")

            # step 1. zero the gradients
            # zero all optimizers
            # for dataset in datasets_list:
            #     dataset.optimizer.zero_grad()

            optimizer.zero_grad()

            y_gold = batch[4].to(device)
            #
            # print(y_gold)
            # print(y_gold.dtype)

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)

            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            # print(y_pred)

            batch_pred = y_pred[batch_task]
            # print(batch_pred)
            # print(batch_pred.dtype)
            # print(f"y predictions are:\n{y_pred}")
            # print(f"y labels are:\n{y_gold}")

            if datasets_list[batch_task].binary:
                batch_pred = batch_pred.float()
                y_gold = y_gold.float()

            # print(datasets_list[batch_task].loss_multiplier)
            # print(datasets_list[batch_task].loss_fx)
            # calculate loss
            loss = datasets_list[batch_task].loss_fx(batch_pred, y_gold) * datasets_list[batch_task].loss_multiplier

            loss_t = loss.item()
            # print(f"Loss for this batch is: {loss_t}")

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # print(f"Running loss is now: {running_loss}")

            # use loss to produce gradients
            loss.backward()

            # add ys to holder for error analysis
            preds_holder[batch_task].extend([item.index(max(item)) for item in batch_pred.tolist()])
            ys_holder[batch_task].extend(y_gold.tolist())

            # increment optimizer
            optimizer.step()
            # for dataset in datasets_list:
            #     dataset.optimizer.step()

        # print(f"All predictions are:\n{preds_holder}")
        # print(f"All labels are:\n{ys_holder}")

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(ys_holder[task], preds_holder[task], average="weighted")
            print(f"Training weighted f-score for task {task}: {task_avg_f1}")
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # Iterate over validation set--put it in a dataloader
        batches, tasks = get_all_batches(datasets_list, batch_size=batch_size, shuffle=True, partition="dev")

        # reset loss and accuracy to zero
        running_loss = 0.0

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # get the task for this batch
            batch_task = tasks[batch_index]

            y_gold = batch[4].to(device)

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)

            # compute the output
            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )

            batch_pred = y_pred[batch_task]

            # print(f"Batch predictions are:\n{batch_pred}")
            # print(f"Batch labels are:\n{y_gold}")

            if datasets_list[batch_task].binary:
                batch_pred = batch_pred.float()
                y_gold = y_gold.float()

            # calculate loss
            loss = datasets_list[batch_task].loss_fx(batch_pred, y_gold) * datasets_list[batch_task].loss_multiplier

            loss_t = loss.item()
            # print(f"Loss for this batch is: {loss_t}")

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # print(f"Running loss is: {running_loss}")

            # add ys to holder for error analysis
            preds_holder[batch_task].extend([item.index(max(item)) for item in batch_pred.tolist()])
            ys_holder[batch_task].extend(y_gold.tolist())

        # print(f"All evaluation predictions:\n{preds_holder}")
        # print(f"All evaluation labels:\n{ys_holder}")

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(ys_holder[task], preds_holder[task], average="weighted")
            print(f"Val weighted f-score for task {task}: {task_avg_f1}")
            train_state["val_avg_f1"][task].append(task_avg_f1[2])

        if epoch_index % 5 == 0:
            for task in preds_holder.keys():
                print(f"Classification report and confusion matrix for task {task}:")
                print(confusion_matrix(ys_holder[task], preds_holder[task]))
                print("======================================================")
                print(classification_report(ys_holder[task], preds_holder[task], digits=4))

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


def get_all_batches(dataset_list, batch_size, shuffle, partition="train"):
    """
    Create all batches and put them together as a single dataset
    """
    # set holder for batches
    all_batches = []
    all_loss_funcs = []

    # get number of tasks
    num_tasks = len(dataset_list)

    # batch the data for each task
    for i in range(num_tasks):
        if partition == "train":
            data = DataLoader(dataset_list[i].train, batch_size=batch_size, shuffle=shuffle)
        elif partition == "dev" or partition == "val":
            data = DataLoader(dataset_list[i].dev, batch_size=batch_size, shuffle=shuffle)
        elif partition == "test":
            data = DataLoader(dataset_list[i].test, batch_size=batch_size, shuffle=shuffle)
        else:
            sys.exit(f"Error: data partition {partition} not found")
        loss_func = dataset_list[i].loss_fx
        # put batches together
        all_batches.append(data)
        all_loss_funcs.append(loss_func)

    randomized_batches = []
    randomized_tasks = []

    # randomize batches
    task_num = 0
    for batches in all_batches:
        for i, batch in enumerate(batches):
            randomized_batches.append(batch)
            randomized_tasks.append(task_num)
        task_num += 1

    zipped = list(zip(randomized_batches, randomized_tasks))
    random.shuffle(zipped)
    randomized_batches, randomized_tasks = list(zip(*zipped))

    return randomized_batches, randomized_tasks


def get_all_batches_oversampling(dataset_list, batch_size, shuffle, partition="train"):
    """
    Create all batches and put them together as a single dataset
    """
    # set holder for batches
    all_batches = []
    all_loss_funcs = []

    # get number of tasks
    num_tasks = len(dataset_list)

    max_dset_len = 0
    if partition == "train":
        for i in range(num_tasks):
            if len(dataset_list[i].train) > max_dset_len:
                max_dset_len = len(dataset_list[i].train)

    print(f"Max dataset length is: {max_dset_len}")

    if partition == "train":
        # only train set should include this sampler!
        # cannot use shuffle with random sampler
        for i in range(num_tasks):
            data_sampler = RandomSampler(data_source=dataset_list[i].train, replacement=True,
                                         num_samples=max_dset_len)
            # print(f'length of samples is: {len(data_sampler)}')
            data = DataLoader(dataset_list[i].train, batch_size=batch_size, shuffle=False,
                              sampler=data_sampler)
            loss_func = dataset_list[i].loss_fx
            # put batches together
            all_batches.append(data)
            all_loss_funcs.append(loss_func)

        print(f"The total number of datasets should match this number: {len(all_batches)}")
        randomized_batches = []
        randomized_tasks = []

        # make batched tuples of (task 0, task 1, task 2)
        # all sets of batches should be same length
        for batch in all_batches[0]:
            randomized_batches.append([batch])
            randomized_tasks.append(0)
        # print(f"The total number of batches after the first dataset is {len(randomized_batches)}")
        # print(f"The total number of tasks after the first dataset is {len(randomized_tasks)}")
        for batches in all_batches[1:]:
            # print(f"The number of batches should be {len(batches)}")
            for i, batch in enumerate(batches):
                randomized_batches[i].append(batch)
                # randomized_batches.append((batch, all_batches[1][i], all_batches[2][i]))

    else:
        # batch the data for each task
        for i in range(num_tasks):
            if partition == "dev" or partition == "val":
                data = DataLoader(dataset_list[i].dev, batch_size=batch_size, shuffle=shuffle)
            elif partition == "test":
                data = DataLoader(dataset_list[i].test, batch_size=batch_size, shuffle=shuffle)
            else:
                sys.exit(f"Error: data partition {partition} not found")
            loss_func = dataset_list[i].loss_fx
            # put batches together
            all_batches.append(data)
            all_loss_funcs.append(loss_func)

        randomized_batches = []
        randomized_tasks = []

        # add all batches to list to be randomized
        task_num = 0
        for batches in all_batches:
            for i, batch in enumerate(batches):
                randomized_batches.append(batch)
                randomized_tasks.append(task_num)
            task_num += 1

    # print(f"length of batches before randomization: {len(randomized_batches)}")
    # print(f"len of batch 0: {len(randomized_batches[0])}")
    # print(f"len of batch 1: {len(randomized_batches[1])}")
    # for i, batch in enumerate(randomized_batches):
    #     if len(batch) != len(randomized_batches[0]):
    #         print(f"len of batch {i} is {len(batch)}")

    # randomize the batches
    zipped = list(zip(randomized_batches, randomized_tasks))
    random.shuffle(zipped)
    randomized_batches, randomized_tasks = list(zip(*zipped))

    # print(f"Length of batches after randomization: {len(randomized_batches)}")
    # print(f"len of batch 0: {len(randomized_batches[0])}")
    # print(f"len of batch 1: {len(randomized_batches[1])}")

    return randomized_batches, randomized_tasks


def predict_without_gold_labels(classifier,
                                test_ds,
                                batch_size,
                                device="cpu",
                                avgd_acoustic=True,
                                use_speaker=True,
                                use_gender=False, ):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    test_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    preds_holder = []

    # for each batch in the dataloader
    for batch_index, batch in enumerate(test_batches):
        # compute the output
        batch_acoustic = batch[0].to(device)
        batch_text = batch[1].to(device)
        batch_lengths = batch[-2].to(device)
        batch_acoustic_lengths = batch[-1].to(device)
        if use_speaker:
            batch_speakers = batch[2].to(device)
        else:
            batch_speakers = None

        if use_gender:
            batch_genders = batch[3].to(device)
        else:
            batch_genders = None

        if avgd_acoustic:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                gender_input=batch_genders,
            )
        else:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                acoustic_len_input=batch_acoustic_lengths,
                gender_input=batch_genders,
            )

        # add ys to holder for error analysis
        preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])

    return preds_holder


def multitask_train_and_predict_with_gradnorm(
        classifier,
        train_state,
        datasets_list,
        batch_size,
        num_epochs,
        optimizer1,
        device="cpu",
        avgd_acoustic=True,
        use_speaker=True,
        use_gender=False,
        optimizer2_learning_rate=0.001
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    Includes gradnorm from https://github.com/hosseinshn/GradNorm/blob/master/GradNormv10.ipynb
    """
    # set loss function
    loss_fx = nn.CrossEntropyLoss(reduction="mean")

    num_tasks = len(datasets_list)

    gradient_loss_fx = nn.L1Loss()

    # set holder for loss weights
    loss_weights = []

    # set holder for task loss 0s
    # todo: this is NOT how they do it in the gradnorm code
    #  but they only seem to have values for epoch 0...
    all_task_loss_0s = []

    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["train_avg_f1"][dset.task_num] = []
        train_state["val_avg_f1"][dset.task_num] = []
        # initialize weight for each dataset
        loss_weights.append(torch.tensor(torch.FloatTensor([1]), requires_grad=True))

    optimizer2 = torch.optim.Adam(loss_weights, lr=optimizer2_learning_rate)

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches, _ = get_all_batches_oversampling(datasets_list, batch_size=batch_size, shuffle=True)

        # print(f"printing length of all batches {len(batches)}")
        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # print(f"Starting batch {batch_index}")
            # set holder for all task losses and loss weights for the batch
            all_task_losses = []

            if epoch_index == 0:
                all_task_loss_0s = []

            # go through each task in turn from within the batch
            for task_idx, task_batch in enumerate(batch):
                # print(f"Starting task {task_idx}")
                # identify the task for this portion of the batch
                batch_task = task_idx
                # get gold labels from the task
                y_gold = task_batch[4].to(device)

                batch_acoustic = task_batch[0].to(device)
                batch_text = task_batch[1].to(device)
                if use_speaker:
                    batch_speakers = task_batch[2].to(device)
                else:
                    batch_speakers = None

                if use_gender:
                    batch_genders = task_batch[3].to(device)
                else:
                    batch_genders = None
                batch_lengths = task_batch[-2].to(device)
                batch_acoustic_lengths = task_batch[-1].to(device)

                if avgd_acoustic:
                    y_pred = classifier(
                        acoustic_input=batch_acoustic,
                        text_input=batch_text,
                        speaker_input=batch_speakers,
                        length_input=batch_lengths,
                        gender_input=batch_genders,
                        task_num=batch_task
                    )
                else:
                    y_pred = classifier(
                        acoustic_input=batch_acoustic,
                        text_input=batch_text,
                        speaker_input=batch_speakers,
                        length_input=batch_lengths,
                        acoustic_len_input=batch_acoustic_lengths,
                        gender_input=batch_genders,
                        task_num=batch_task
                    )

                batch_pred = y_pred[batch_task]

                # get the loss for that task in that batch
                task_loss = loss_weights[batch_task] * loss_fx(batch_pred, y_gold)
                # print(f"task_loss is {task_loss}")
                # sys.exit()
                all_task_losses.append(task_loss)

                # for first epoch, set loss per item
                if epoch_index == 0:
                    task_loss_0 = task_loss.item()
                    all_task_loss_0s.append(task_loss_0)

                # add ys to holder for error analysis
                preds_holder[batch_task].extend([item.index(max(item)) for item in batch_pred.tolist()])
                ys_holder[batch_task].extend(y_gold.tolist())

            # calculate total loss
            # print(f"All task losses are {all_task_losses}")
            loss = torch.div(sum(all_task_losses), len(all_task_losses))
            # print(loss)

            optimizer1.zero_grad()

            # use loss to produce gradients
            loss.backward(retain_graph=True)

            # get gradients of first layer of task-specific calculations
            param = list(classifier.parameters())
            final_shared_lyr_wt = param[40]
            all_normed_grads = []
            for task in range(num_tasks):
                # use the final shared layer weights to calculate gradient
                # here, this is param[40]
                task_grad = torch.autograd.grad(all_task_losses[task], final_shared_lyr_wt, create_graph=True)
                # print(task_grad)
                normed_grad = torch.norm(task_grad[0], 2)
                # print(normed_grad)
                all_normed_grads.append(normed_grad)

            # calculate average of normed gradients
            normed_grad_avg = torch.div(sum(all_normed_grads), len(all_normed_grads))

            # calculate relative losses
            all_task_loss_hats = []
            # print(f"The number of tasks is {num_tasks}")
            # print(f"All task losses are: {all_task_losses}")
            # print(f"All task loss 0s are: {all_task_loss_0s}")
            for task in range(num_tasks):
                task_loss_hat = torch.div(all_task_losses[task], all_task_loss_0s[task])
                all_task_loss_hats.append(task_loss_hat)
            loss_hat_avg = torch.div(sum(all_task_loss_hats), len(all_task_loss_hats))

            # calculate relative inverse training rate for tasks
            all_task_inv_rates = []
            for task in range(num_tasks):
                task_inv_rate = torch.div(all_task_loss_hats[task], loss_hat_avg)
                all_task_inv_rates.append(task_inv_rate)

            # calculate constant target for gradnorm paper equation 2
            alph = .16  # as selected in paper. could move to config + alter
            all_C_values = []
            for task in range(num_tasks):
                task_C = normed_grad_avg * all_task_inv_rates[task] ** alph
                task_C = task_C.detach()
                all_C_values.append(task_C)

            optimizer2.zero_grad()

            # calculate gradient loss using equation 2 in gradnorm paper
            all_task_gradient_losses = []
            for task in range(num_tasks):
                task_gradient_loss = gradient_loss_fx(all_normed_grads[task], all_C_values[task])
                all_task_gradient_losses.append(task_gradient_loss)
            gradient_loss = sum(all_task_gradient_losses)
            # propagate the loss
            gradient_loss.backward(retain_graph=True)

            # increment weights for loss
            optimizer2.step()

            # increment optimizer for model
            optimizer1.step()

            # renormalize the loss weights
            all_weights = sum(loss_weights)
            coef = num_tasks / all_weights
            loss_weights = [item * coef for item in loss_weights]

            # get loss calculations for train state
            # this is NOT gradnorm's calculation
            loss_t = loss.item()
            # print(loss_t)
            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(ys_holder[task], preds_holder[task], average="weighted")
            print(f"Training weighted f-score for task {task}: {task_avg_f1}")
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # Iterate over validation set--put it in a dataloader
        batches, tasks = get_all_batches(datasets_list, batch_size=batch_size, shuffle=True, partition="dev")

        # reset loss and accuracy to zero
        running_loss = 0.0

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # holder for losses for each task in dev set
        task_val_losses = []
        for _ in range(num_tasks):
            task_val_losses.append(0.0)

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # get the task for this batch
            batch_task = tasks[batch_index]

            y_gold = batch[4].to(device)

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)

            # compute the output
            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )

            batch_pred = y_pred[batch_task]

            if datasets_list[batch_task].binary:
                batch_pred = batch_pred.float()
                y_gold = y_gold.float()

            # calculate loss
            loss = loss_weights[batch_task] * loss_fx(batch_pred, y_gold)
            loss_t = loss.item()

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # add ys to holder for error analysis
            preds_holder[batch_task].extend([item.index(max(item)) for item in batch_pred.tolist()])
            ys_holder[batch_task].extend(y_gold.tolist())

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(ys_holder[task], preds_holder[task], average="weighted")
            print(f"Val weighted f-score for task {task}: {task_avg_f1}")
            train_state["val_avg_f1"][task].append(task_avg_f1[2])

        if epoch_index % 5 == 0:
            for task in preds_holder.keys():
                print(f"Classification report and confusion matrix for task {task}:")
                print(confusion_matrix(ys_holder[task], preds_holder[task]))
                print("======================================================")
                print(classification_report(ys_holder[task], preds_holder[task], digits=4))

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break


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
            eps = 1e-6
            loss = torch.sqrt(loss_func(y_pred, y_gold))
            # loss = torch.sqrt(loss + eps)
            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

        # add loss and accuracy information to the train state
        epoch_loss = mean_squared_error(preds_holder, ys_holder)
        # epoch_loss = loss_func(preds_holder, ys_holder)
        epoch_r2 = r2_score(ys_holder, preds_holder)
        # epoch_r2 = calc_r_squared(ys_holder, preds_holder)
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
        epoch_val_r2 = r2_score(ys_holder, preds_holder)
        # epoch_val_r2 = r2_score(ys_holder, preds_holder)
        epoch_val_loss = mean_squared_error(preds_holder, ys_holder)
        # epoch_val_loss = loss_func(preds_holder, ys_holder)
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
            loss = (loss1 + loss2 + loss3) / 3

            # step 4. use loss to produce gradients
            loss.backward()

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

        train_state["train_loss"].append(epoch_loss)
        train_state["train_r2"].append(epoch_r2)

        print("Epoch {0}\tTrain R2: {1}\tTrain Loss: {2}".format(epoch_index, epoch_r2, epoch_loss))
        # print("Training r2: " + str(epoch_r2))

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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
        train_state["val_loss"].append(epoch_val_loss)

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

    result_file_cv = result_file.replace(".csv", "_cv.csv")
    cv_loss = mean_squared_error(ys_holder, preds_holder)
    cv_r2 = r2_score(ys_holder, preds_holder)
    # cv_r2 = calc_r_squared(ys_holder, preds_holder)
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