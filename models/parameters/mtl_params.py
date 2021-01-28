from argparse import Namespace

params = Namespace(
    # consistency parameters
    seed=1007,  # 1007
    # trying text only model or not
    # overall model parameters
    model="MTL",
    num_epochs=1000,
    batch_size=32,  # 128,  # 32
    early_stopping_criteria=400,
    num_gru_layers=2,  # 1,   # 3,  # 1,  # 4, 2,
    bidirectional=False,
    # input dimension parameters
    audio_dim=39,  # 76,  # 79,  # 10 # audio vector length
    acoustic_dim=76,
    phono_dim=9,
    # acoustic NN
    avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
    add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
    acoustic_gru_hidden_dim=1024,
    # speaker embeddings
    # outputs
    output_dim=1,  # 7, 9 # length of output vector
    # FC layer parameters
    fc_hidden_dim=64,  # 20,
    dropout=0.2,  # 0.2
    # optimizer parameters
    lrs=[5e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
    embracement_size=256,
    bypass_docking=False,
    availabilities=None,
    selection_probabilities=None
)