REMOTE_PORT = '2222'
REMOTE_HOST = "mininet@localhost"
REMOTE_SERVER_RUNNER_HOSTNAME = ["mininet@localhost"]
REMOTE_SERVER_RUNNER_PORT = ["2222"]

S_INFO = 6  # bandwidth_path_i, path_i_mean_RTT, path_i_retransmitted_packets + path_i_lost_packets
A_DIM = 5 # two actions -> path 1 or path 2



#model_name = 'minrtt' #'FALCON'
model_name = 'LSTM'
TRAINING = False #if true, store model after done, have high exploration
MODE = 'train' if TRAINING else 'test'
SEGMENT_LIMIT = 301
EPISODES_TO_RUN = 10

# hyperparameters
hidden_size = 256
learning_rate = 0.001 #0.005  #3e-4
apply_loss_steps = 25

GAMMA = 0.99
EPS_START = 0.05 #0.9
EPS_END = 0.05
EPS_DECAY = 1000 #higher = slower decay

#change detection
COOLDOWN_TIME = 60
CHANGE_PROB = 50.0

LSTM_TRAINED_MODEL = "runs/20230528_00_30_35_LSTM_train/9_model.tar"
