from typing import List, Union, Tuple

from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .args import ModelArgs
from .features import BatchMolGraph
from .nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""
    def __init__(self, args: ModelArgs):
        """
        :param args: A :class:`~chemprop.args.ModelArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        
        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification:
            self.no_training_normalization = args.loss_function in ['cross_entropy', 'binary_cross_entropy']

        self.output_size = args.num_tasks
        
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: ModelArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.ModelArgs` object containing model arguments.
        """
        self.encoder = MPN(args)         
                
    def create_ffn(self, args: ModelArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.ModelArgs` object containing model arguments.
        """
        
        first_linear_dim = args.hidden_size * args.number_of_molecules
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        
    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]]) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """
        output = self.ffn(self.encoder(batch))

        # Don't apply sigmoid during training when using BCEWithLogitsLoss
        if self.classification and not (self.training and self.no_training_normalization):
            output = self.sigmoid(output)

        return output
