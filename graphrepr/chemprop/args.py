import torch

class ModelArgs():
    def __init__(self, conv_layers_num, model_dim, dense_layers_num, hidden_size, dropout,
                 batchnorm=False, loss_function='mse', device=None):
        """
        Class that keeps arguments required to build chemprop.model.MoleculeModel
        but with graphconv.GraphConvNetwork API.
        
        batchnorm controls aggregation. It's unintuitive but this way
        GraphConvNetwork configuration files can be reused with chemprop.MoleculeModel.
        """
        # these we edit
        self.depth = conv_layers_num            # "Number of message passing steps."
        self.hidden_size = model_dim            # "Dimensionality of hidden layers in MPN."
        self.ffn_num_layers = dense_layers_num  # "Number of layers in FFN after MPN encoding."
        self.ffn_hidden_size = hidden_size      # "Hidden dim for higher-capacity FFN (defaults to hidden_size)."
        self.dropout = dropout                  # Dropout probability

        # "Aggregation scheme for atomic vectors into molecular vectors"
        self.aggregation = 'sum' if batchnorm else 'mean'   # Literal['mean', 'sum', 'norm'] = 'mean'
        self.aggregation_norm = 100  # "For norm aggregation, number by which to divide summed up atomic features"
        
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

        # these are constant
        self.atom_messages = False  # "Centers messages on atoms instead of on bonds." 
        self.undirected = False     # "Undirected edges (always sum the two relevant bond vectors)."
        
        self.bias = True                    # "Whether to add bias to linear layers."
        self.activation = 'ReLU'            # Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
        self.dataset_type = 'regression'    # Literal['regression', 'classification', 'multiclass', 'spectra']
        self.loss_function = loss_function  # Literal['mse', 'bounded_mse', 'binary_cross_entropy','cross_entropy', 'mcc', 'sid', 'wasserstein']
        
        self.num_tasks = 1            # we always predict a single property
        self.number_of_molecules = 1  # prediction is done for a single molecule (and not. ex. a pair)
        