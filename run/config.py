# This is the config file which holds the configuration values for different architectures

# Configrations for different architectures explained below:


Res_Bidir_LSTM = {
	'name' : 'Res_Bidir_LSTM',
	'bidir' : True,
	'clip_val' : 10,
	'drop_prob' : 0,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0.001,
	'n_residual_layers' : 2,
	'n_highway_layers' : 2,
	'diag' : 'Architecure chosen is Residual Bidirectional LSTM with 2 layers',
	'save_file' : 'results_res_lstm1.txt'
}

Architecture = {
	'Res_Bidir_LSTM' : Res_Bidir_LSTM
}


# Choose what architecure you want here:
name_modle = 'Res_Bidir_LSTM'
arch = Architecture[name_modle]

# This will set the values according to that architecture
bidir = arch['bidir']
clip_val = arch['clip_val']
drop_prob = arch['drop_prob']
n_epochs_hold = arch['n_epochs_hold']
n_layers = arch['n_layers']
learning_rate = arch['learning_rate']
weight_decay = arch['weight_decay']
n_highway_layers = arch['n_highway_layers']
n_residual_layers = arch['n_residual_layers']

# These are for diagnostics
diag = arch['diag']
save_file = arch['save_file']

# This will stay common for all architectures:
n_classes = 2
n_input = 4
n_hidden = 32
batch_size = 64
n_epochs = 150

checkpoint = './Checkpoints_window/' + name_modle + '/best_model.pth'
#checkpoint = ''
#is_train = True
is_train = None

