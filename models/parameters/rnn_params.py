from argparse import Namespace

params = Namespace(
    # consistency parameters
    seed=88,  # 1007
    # trying text only model or not
    text_only=False,
    # overall model parameters
    model="w2v_rnn",
    num_epochs=1000,
    batch_size=32,  # 128,  # 32
    early_stopping_criteria=500,
    num_gru_layers=2,  # 1,   # 3,  # 1,  # 4, 2,
    bidirectional=False,
    # input dimension parameters
    # audio_dim=512,  # 76,  # 79,  # 10 # audio vector length
    audio_dim=39,
    phono_dim=384,
    # audio_dim=10,
    # text NN
    # text_output_dim=30,   # 100,   # 50, 300,
    text_gru_hidden_dim=500,  # 30,  # 50,  # 20
    # acoustic NN
    avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
    add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
    acoustic_gru_hidden_dim=1024,
    # speaker embeddings
    # outputs
    output_dim=1,  # 7, 9 # length of output vector
    output_2_dim=None,  # 3,    # length of second task output vec
    # FC layer parameters
    num_fc_layers=2,  # 1,  # 2,
    fc_hidden_dim=256,  # 20,
    dropout=0.4,  # 0.2
    # optimizer parameters
    lrs=[1e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.001,
    class_weight=[3.64, 18.47, 25.12, 3.12, 1.0, 6.04, 4.47]
)
