# plot the training and validation curves

import matplotlib.pyplot as plt


def plot_train_dev_curve(
    train_losses,
    dev_losses,
    x_label="",
    y_label="",
    title="",
    save_name=None,
    show=False,
    losses=True,
    set_axis_boundaries=True,
):
    """
    plot the loss or accuracy curves over time for training and dev set
    """
    # get a list of the epochs
    epoch = [i for i, item in enumerate(train_losses)]

    # prepare figure
    fig, ax = plt.subplots()
    plt.grid(True)

    # add losses/epoch for train and dev set to plot
    ax.plot(epoch, train_losses, label="train")
    ax.plot(epoch, dev_losses, label="dev")

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # depending on type of input, set the y axis boundaries
    if set_axis_boundaries:
        if losses:
            ax.set_ylim([0.66, 0.72])
        else:
            ax.set_ylim([0.3, 1.0])

    # create title and legend
    ax.set_title(title, loc="center", wrap=True)
    ax.legend()

    # save the file
    if save_name is not None:
        plt.savefig(fname=save_name)
        plt.close()

    # show the plot
    if show:
        plt.show()
