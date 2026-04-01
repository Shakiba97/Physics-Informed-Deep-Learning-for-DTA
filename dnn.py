import torch

from collections import OrderedDict


class DNNCal(torch.nn.Module):
    """
    Input of the NN: q_u(t), q_d(t), p(t), v(t),
    and the pre-defined demand D(t),
    and parameters tau_0, tau_w, C_bar, Q_bar,
    and the link priority.
    The dimension of each output from MATLAB: n_edges;
    The dimension of each input in the NN: n_edges x 1.  # TODO: should this be changed to n_demand * n_edges * 1??
    Thus, the dimension of this input: n_edges x 10.
    And the input layer has 11 neurons.
    ------
    Output of the NN, t means each time instant: q_u(t+1), q_d(t+1), p(t+1), v(t+1),
    and the parameters we need from the calibration process: tau_0, tau_omega, Cbar, Qbar
    The dimension of each output in the NN: n_edges x 1.
    Thus, the dimension of this output: n_edges x 8.
    And the output layer has 8 neurons.
    """

    def __init__(self, layers):
        super(DNNCal, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation_1 = torch.nn.ReLU
        self.activation_2 = torch.nn.Sigmoid

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation_1()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_list.append(('activation_%d' % (self.depth - 1), self.activation_2()))
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class DNNHidden(torch.nn.Module):
    """
        Input of the NN: q_u(t), q_d(t), p(t), v(t), mu(t), delta(t) from DTA,
        and the pre-defined demand D(t),
        and parameters tau_0, tau_w, C_bar, Q_bar,
        and the link priority.
        The dimension of each output from MATLAB: n_edges;
        The dimension of each input in the NN: (n_edges) x 1.
        Thus, the dimension of this input: (n_edges) x 12.
        And the input layer has 13 neurons.
        ------
        Output of the NN, t means each time instant: q_u(t+1), q_d(t+1), p(t+1), v(t+1), mu(t+1), delta(t+1)
        The dimension of each output in the NN: (n_edges) x 1.
        Thus, the dimension of this output: (n_edges) x 6.
        And the output layer has 6 neurons.
    """

    def __init__(self, layers):
        super(DNNHidden, self).__init__()

        # hidden layer. try a in >> 16 >> 32 >> 16+1 >> out network
        self.linear_h_1 = torch.nn.Linear(layers[0], layers[1])
        self.linear_h_2 = torch.nn.Linear(layers[1], layers[2])
        self.linear_h_3_1 = torch.nn.Linear(layers[2], layers[3])
        self.linear_h_3_2 = torch.nn.Linear(layers[2], layers[4])
        self.linear_out_1 = torch.nn.Linear(layers[3], layers[5])  # 3: p, v, qd
        self.linear_out_2 = torch.nn.Linear(layers[4], layers[6])  # 1: beta
        self.linear_out_2_2 = torch.nn.Linear(layers[6], layers[7], bias=False)  # 1: beta to TT; the weight is gamma
        self.weight = torch.nn.Parameter(torch.Tensor(1, 1))
        #torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # defining layers as attributes
        self.layer_1 = None
        self.layer_1_act = None
        self.layer_2 = None
        self.layer_2_act = None
        self.layer_3_1 = None
        self.layer_3_1_act = None
        self.layer_3_2 = None
        self.layer_3_2_act = None
        self.layer_out_1 = None
        self.layer_out_2 = None
        self.layer_out_2 = None
        self.layer_out_2_act = None
        self.layer_out_2_2 = None


    def forward(self, x):
        # you can try different activation functions
        self.layer_1 = self.linear_h_1(x)
        self.layer_1_act = torch.sigmoid(self.layer_1)
        self.layer_2 = self.linear_h_2(self.layer_1_act)
        self.layer_2_act = torch.sigmoid(self.layer_2)
        self.layer_3_1 = self.linear_h_3_1(self.layer_2_act)
        self.layer_3_1_act = torch.sigmoid(self.layer_3_1)
        self.layer_3_2 = self.linear_h_3_2(self.layer_2_act)
        self.layer_3_2_act = torch.sigmoid(self.layer_3_2)
        self.layer_out_1 = self.linear_out_1(self.layer_3_1_act)
        self.layer_out_1_act=torch.sigmoid(self.layer_out_1)
        self.layer_out_2 = self.linear_out_2(self.layer_3_2_act)
        self.layer_out_2_act=torch.sigmoid(self.layer_out_2)
        self.layer_out_2_2 = self.weight*self.layer_out_2_act
        # ^ don't use activation function to maintain linearity: gamma * beta
        flows_pred = self.layer_out_1_act
        beta_pred = self.layer_out_2_act
        traveltime_pred = self.layer_out_2_2
        #gamma = torch.mean(self.linear_out_2_2.weight)
        # gamma = torch.transpose(self.linear_out_2_2.weight,0,1)
        # gamma= torch.mean(self.weight,1,True)
        gamma = self.weight

        return flows_pred, beta_pred, traveltime_pred, gamma
