import os
import numpy as np
import sys

# create config files for all possible models

saving_dir = '.'

n_conv_layers = [1, 3, 5]
model_dims_hidden_sizes = [16, 64, 256]
dense_layers = [1, 3]
dr = [0.0, 0.2]
lr = [.01, .001, .0001, .00001, .000001]
batchnorm = [True, False]
batchsize = [8, 32, 128]
scheduler = [-1., 0.5, 0.8] # no scheduler, decrease after half 50% of the epochs, decrease after 80% of the epochs

model_template = """[PARAMS]
conv_layers_num: @conv_layers
model_dim: @model_dim
dense_layers_num: @dense_layers
hidden_size: @hidden_size
dropout: @dropout
batchnorm: @batchnorm

[OPTIMIZER]
optimizer: adam
batch_size: @batchsize
lr: @lr
scheduler: @scheduler
n_epochs: 750

"""

i = 0  # unique identifier for each configuration
for n_convs in n_conv_layers:
    for repr_dim in model_dims_hidden_sizes:
        for n_dense in dense_layers:
            for d in dr:
                for l in lr:
                    for b_n in batchnorm:
                        for b_s in batchsize:
                            for sch in scheduler:
                                i+=1
                                this_filename = f"{i}-model.cfg"

                                this_tmpl = model_template.replace('@conv_layers', str(n_convs)).replace('@model_dim', str(repr_dim))
                                this_tmpl = this_tmpl.replace('@dense_layers', str(n_dense)).replace('@hidden_size', str(repr_dim))
                                this_tmpl = this_tmpl.replace('@dropout', str(d)).replace('@lr', str(l))
                                this_tmpl = this_tmpl.replace('@batchnorm', str(b_n)).replace('@batchsize', str(b_s))
                                this_tmpl  = this_tmpl.replace('@scheduler', str(sch))

                                with open(os.path.join(saving_dir, this_filename), 'w') as f:
                                    f.write(this_tmpl)
