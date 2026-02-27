import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(module,ensemble_idx, critic_factor):
    if type(module).__name__ == 'VectorisedLinear':
        module.reset_parameters()
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))


class VectorisedLinear(nn.Module):
    
    '''Used for creating ensembles layers'''
    def __init__(self, in_features, out_features, critic_factor, ensemble_num):
        super().__init__()

        self.ensemble_num = ensemble_num
        self.critic_factor = critic_factor
        self.weight = nn.Parameter(torch.empty((ensemble_num, critic_factor, in_features, out_features)))
        self.bias = nn.Parameter(torch.empty((ensemble_num, critic_factor, 1, out_features)))
        self.reset_parameters()

    def reset_parameters(self):


        ##initialise parameters of network
        for ensemble_idx in range(self.ensemble_num):
            for critic_factor_idx in range(self.critic_factor):
                nn.init.kaiming_uniform_(self.weight[ensemble_idx][critic_factor_idx], a=math.sqrt(5))

        ##calculate number of inputs and ouptut for any network in the ensemble
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0][0])
        ##initialise bias
        bound = 1/math.sqrt(fan_in) if fan_in>0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        ## outside the inner most two dims broadcasting/batching is done
        ## where a one to one correspondence isn't possible

        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x@self.weight +self.bias



class BaseVectorisedNetwork(nn.Module):

    ''' base network that can be used for actor, critic or value networks '''

    def __init__(self, obs_dims, model_info, ensemble_num=1, critic_factor=1, **kwargs):

        super().__init__()
        
        self.ensemble_num = ensemble_num 
        self.critic_factor = critic_factor
        self.cat_dim = 3

        self.in_dims = obs_dims + kwargs['action_dims'] if kwargs.get('action_dims') else obs_dims

        self.hidden_activation = getattr(torch.nn,model_info['hidden_activation'],None)


    def reset_weights(self, ensemble_idx):
        'reset parameters of a specific ensemble'
        self.apply(lambda m: weights_init(module=m,
                                        ensemble_idx=ensemble_idx,
                                        critic_factor=self.critic_factor)
                                        )

    def construct_model(self, model_info, add_final=False):

        in_dims = self.in_dims
        layers_dim_list = model_info['layers']
        module_name_prefix = "layer_"

        model = torch.nn.Sequential()

        for idx, out_dims in enumerate(layers_dim_list):
            module_name = module_name_prefix + f"fc{idx+1}"
            if self.ensemble_num == 1 and self.critic_factor == 1:
                model.add_module(module_name, nn.Linear(in_dims,out_dims))
            else:
                model.add_module(module_name, VectorisedLinear(in_dims,out_dims,self.critic_factor,self.ensemble_num))
            model.add_module(f"{self.hidden_activation.__name__}{idx+1}",self.hidden_activation())
            in_dims = out_dims

        ##adding final layer to network
        if add_final:
            in_dims = self.add_final_layer(model,in_dims)
        return model

    def add_final_layer(self, model, in_dims):

        if self.ensemble_num == 1 and self.critic_factor == 1:
            model.add_module('final_layer',nn.Linear(in_dims, self.final_layer_dim))
        else:
            model.add_module('final_layer',VectorisedLinear(in_dims, self.final_layer_dim, self.critic_factor, self.ensemble_num))

        ##sometimes no final activation layer
        if getattr(self,'final_activation',None):
            ##need to figure out a way to make dim an optional argument!!!
            if self.final_activation.__name__ == 'Softmax':
                model.add_module(f"{self.final_activation.__name__}_final",self.final_activation(dim=self.cat_dim))
            else:
                model.add_module(f"{self.final_activation.__name__}_final",self.final_activation())

        return self.final_layer_dim

