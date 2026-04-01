import numpy as np
import torch
import scipy.io
import matlab.engine

import network

from logger import TrainLogger
from dnn import DNNCal, DNNHidden


def write_param(f, param, links):
    param = param.reshape(links.shape[0], 1)
    for i in range(links.shape[0]):
        f.write("{0:d}.{1:d} {2:2.1f}\n".format(int(links[i, 0]),
                                                int(links[i, 1]),
                                                float(torch.round(param[i]))))

    f.write("/;\n")
    f.write("\n")


def denormalize_max_and_min(data, ma, mi):
    data_process = data.clone()
    num_batches = data.shape[0]
    if len(data.shape) < 2 or data.shape[1] == 1:
        for i in range(num_batches):
            data_process[i] = data_process[i] * (ma[i] - mi[i]) + mi[i]
    else:
        for i in range(num_batches):
            data_process[i, :] = data_process[i, :] * (ma[i] - mi[i]) + mi[i]

    return data_process


def normalize_max_and_min(data, ma=0.0, mi=0.0):
    data_process = data.clone()
    num_batches = data.shape[0]
    if len(data.shape) < 2 or data.shape[1] == 1:
        for i in range(num_batches):
            if ma[i] - mi[i] > 0:
                data_process[i] = (data[i] - mi[i]) / (ma[i] - mi[i])
    else:
        for i in range(num_batches):
            if ma[i] - mi[i] > 0:
                data_process[i, :] = (data[i, :] - mi[i]) / (ma[i] - mi[i])

    return data_process


def produce_initial_guess(n, lower_bound=0.0, upper_bound=1.0):
    return (upper_bound - lower_bound) * torch.rand(n) + lower_bound


class PhysicsInformedNN:
    def __init__(self, device):
        """
        This is a physics-informed neural network.
        The goal is to complete a structure of training two NNs.
        NNCal is trained to calibrate the parameters in the physical DTA.
        NNHid is trained to estimate the hidden variables of the physical DTA.
        ------
        NNCal:
        We use p(t+1), v(t+1), q_u(t+1), q_d(t+1) to control the performance.
        These variables are read in read_obs().
        -----
        :param device: whether to use GPU to train NNs
        """
        self.device = device
        self.edges = network.get_edges()
        self.nodes = network.get_nodes()
        self.n_edges = len(self.edges)
        self.total_time = total_time
        self.i_time = 0

        self.tau_0_cal = None
        self.tau_0_cal_max = None
        self.tau_0_cal_min = None
        self.tau_w_cal = None
        self.tau_w_cal_max = None
        self.tau_w_cal_min = None
        self.C_bar_cal = None
        self.C_bar_cal_max = None
        self.C_bar_cal_min = None
        # self.Q_bar_cal = None
        # self.Q_bar_cal_max = None
        # self.Q_bar_cal_min = None
        self.p_max = None
        self.p_min = None
        self.v_max = None
        self.v_min = None
        self.qu_max = None
        self.qu_min = None
        self.qd_max = None
        self.qd_min = None
        self.tt_max = None
        self.tt_min = None
        self.gamma_max = None
        self.gamma_min = None
        self.delta_max = None
        self.delta_min = None
        self.mu_max = None
        self.mu_min = None
        self.demand = None
        self.adj = None
        self.priority = None

        self.retrieve_params()  ### shaki: it saves new calibrated parameters in GAMS??
        self.read_range()

        p_ob, v_ob, qd_ob, tt_ob = read_obs()  # note the obs have all info at all time steps

        # initialize observations
        self.p_ob = torch.tensor(p_ob).float().to(self.device)
        self.v_ob = torch.tensor(v_ob).float().to(self.device)
        # self.qu_ob = torch.tensor(qu_ob).float().to(self.device)
        self.qd_ob = torch.tensor(qd_ob).float().to(self.device)
        self.tt_ob = torch.tensor(tt_ob).float().to(self.device)

        # bb[:, 0:total_time - t0] = bb[:, 0:total_time - t0] + tau_0_cal.unsqueeze(1).round().cpu().numpy()
        # bb[0:6, total_time - t0:total_time] = np.repeat(t0, t0)

        # initialize DTA variables
        self.p_phy = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.v_phy = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.qu_phy = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.qd_phy = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.mu_phy = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.delta_phy = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)

        self.p_phy_norm = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.v_phy_norm = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.qu_phy_norm = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.qd_phy_norm = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.mu_phy_norm = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)
        self.delta_phy_norm = torch.tensor(np.zeros_like(p_ob)).float().to(self.device)

        # normalize data
        self.p_ob_norm = normalize_max_and_min(self.p_ob, self.p_max, self.p_min)
        self.v_ob_norm = normalize_max_and_min(self.v_ob, self.v_max, self.v_min)
        #self.qu_ob_norm = normalize_max_and_min(self.qu_ob, self.qu_max, self.qu_min)
        self.qd_ob_norm = normalize_max_and_min(self.qd_ob, self.qd_max, self.qd_min)
        # TODO: change later tt max and min

        self.tt_ob_norm = normalize_max_and_min(self.tt_ob, self.tt_max, self.tt_min)

        self.tau_0_cal_norm = normalize_max_and_min(self.tau_0_cal, self.tau_0_cal_max, self.tau_0_cal_min)
        self.tau_w_cal_norm = normalize_max_and_min(self.tau_w_cal, self.tau_w_cal_max, self.tau_w_cal_min)
        self.C_bar_cal_norm = normalize_max_and_min(self.C_bar_cal, self.C_bar_cal_max, self.C_bar_cal_min)
        #self.Q_bar_cal_norm = normalize_max_and_min(self.Q_bar_cal, self.Q_bar_cal_max, self.Q_bar_cal_min)

        # deep neural networks
        self.nn_cal = DNNCal(layers_nn_cal).to(self.device)
        self.nn_hid = DNNHidden(layers_nn_hid).to(self.device)

        # optimizers: using the same settings
        self.optimizer_cal = torch.optim.Adam(self.nn_cal.parameters(), lr=1e-3)
        self.optimizer_hid = torch.optim.Adam(self.nn_hid.parameters(), lr=5*1e-4)
        self.iter = 0

    def update_params(self, D, tau_0_cal, tau_w_cal, C_bar_cal):
        i = 0
        for edge_id, edge in self.edges.items():
            edge.set_freeflowTravelTime(torch.round(tau_0_cal[i]))
            edge.set_shockwaveTravelTime(torch.round(tau_w_cal[i]))
            edge.update_flowCap(torch.round(C_bar_cal[i]))
            #edge.update_queueCap(torch.round(Q_bar_cal[i]))
            i += 1
        network.refresh(gpu)
        self.retrieve_params(D)

    # TODO: demand should be changed for general networks
    def retrieve_params(self, D=0):  ### shaki: self is an object from class physicsinformednn
        tau_0 = torch.zeros(self.n_edges)
        tau_w = torch.zeros(self.n_edges)
        C_bar = torch.zeros(self.n_edges)
        #Q_bar = torch.zeros(self.n_edges)
        priority = torch.zeros(self.n_edges)
        demand = torch.zeros(self.n_edges, self.total_time)
        i = 0
        for edge_id, edge in self.edges.items():
            tau_0[i] = edge.freeflowTravelTime
            tau_w[i] = edge.shockwaveTravelTime
            C_bar[i] = edge.flowCap
            #Q_bar[i] = edge.queueCap
            priority[i] = edge.priority
            i += 1
        i = 0
        for node_id, node in self.nodes.items():
            # TODO: correct the demand here (node.demand is still based on previous network)
            #demand[i, :] = node.demand
            i += 1

        adj = network.adj

        # initialize
        self.tau_0_cal = tau_0.float().clone().to(self.device)
        self.C_bar_cal = C_bar.float().clone().to(self.device)
        #self.Q_bar_cal = Q_bar.float().clone().to(self.device)
        self.tau_w_cal = tau_w.float().clone().to(self.device)
        self.demand = demand.float().clone().to(self.device)
        self.adj = adj
        self.priority = priority.float().clone().to(self.device)

        # save the parameters in gams
        self.save_gams_params(D, self.tau_0_cal,
                              self.tau_w_cal,
                              self.C_bar_cal)

    def read_range(self, f="./data/given.mat"):
        bound = scipy.io.loadmat(f)
        bound['p_range'] = bound['p_range'].astype('int32')
        bound['v_range'] = bound['v_range'].astype('int32')
        bound['qu_range'] = bound['qu_range'].astype('int32')
        bound['qd_range'] = bound['qd_range'].astype('int32')
        bound['delta_range'] = bound['delta_range'].astype('int32')
        bound['mu_range'] = bound['mu_range'].astype('int32')
        bound['Cbar_range'] = bound['Cbar_range'].astype('int32')
        bound['Qbar_range'] = bound['Qbar_range'].astype('int32')


        self.p_max = torch.tensor(bound['p_range'][:, 1]).float().to(self.device)
        self.p_min = torch.tensor(bound['p_range'][:, 0]).float().to(self.device)
        self.v_max = torch.tensor(bound['v_range'][:, 1]).float().to(self.device)
        self.v_min = torch.tensor(bound['v_range'][:, 0]).float().to(self.device)
        self.qu_max = torch.tensor(bound['qu_range'][:, 1]).float().to(self.device)
        self.qu_min = torch.tensor(bound['qu_range'][:, 0]).float().to(self.device)
        self.qd_max = torch.tensor(bound['qd_range'][:, 1]).float().to(self.device)/3
        self.qd_min = torch.tensor(bound['qd_range'][:, 0]).float().to(self.device)
        self.tt_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]).float().to(self.device)*40
        self.tt_min = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]).float().to(self.device)*0
        self.beta_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]).float().to(self.device)*40
        self.beta_min = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]).float().to(self.device)*0
        self.delta_max = torch.tensor(bound['delta_range'][:, 1]).float().to(self.device)
        self.delta_min = torch.tensor(bound['delta_range'][:, 0]).float().to(self.device)
        self.mu_max = torch.tensor(bound['mu_range'][:, 1]).float().to(self.device)
        self.mu_min = torch.tensor(bound['mu_range'][:, 0]).float().to(self.device)
        self.tau_0_cal_max = torch.tensor(bound['tau0_range'][:, 1]).float().to(self.device)
        self.tau_0_cal_min = torch.tensor(bound['tau0_range'][:, 0]).float().to(self.device)
        self.tau_w_cal_max = torch.tensor(bound['tauw_range'][:, 1]).float().to(self.device)
        self.tau_w_cal_min = torch.tensor(bound['tauw_range'][:, 0]).float().to(self.device)
        self.C_bar_cal_max = torch.tensor(bound['Cbar_range'][:, 1]).float().to(self.device)
        self.C_bar_cal_min = torch.tensor(bound['Cbar_range'][:, 0]).float().to(self.device)
        #self.Q_bar_cal_max = torch.tensor(bound['Qbar_range'][:, 1]).float().to(self.device)
        #self.Q_bar_cal_min = torch.tensor(bound['Qbar_range'][:, 0]).float().to(self.device)

    def zip_inputs_hid(self, time_step,
                       qd, p, v, tt_or_beta):
        """
        Reshape the n_edges array into
        n_edges x 1 for each feature.
        We have 12 features thus output is in the dimension of
        n_edges x 12.
        The input order is consistent with the order in description in DNNHid
        q_u(t), q_d(t), p(t), v(t), mu(t), delta(t) from DTA,
        and parameters tau_0, tau_w, C_bar, Q_bar,
        and the pre-defined demand D(t),
        and the link priority.
        :return: concatenated input
        """
        #qu = qu_phy.reshape((self.n_edges, 1))
        qd = torch.transpose(qd,0,1)
        p = torch.transpose(p,0,1)
        v = torch.transpose(v,0,1)
        tt_or_beta = torch.transpose(tt_or_beta,0,1)

        # demand = self.demand[:, time_step - 1].reshape((self.n_edges, 1))
        # priority = self.priority.reshape((self.n_edges, 1))

        inputs = torch.cat([qd, p, v, tt_or_beta], dim=1)
        return inputs

    def zip_inputs_cal(self, time_step,
                       qd_nn, p_nn, v_nn,
                       tau_0, tau_w, C_bar):
        """
        Reshape the n_edges array into
        n_edges x total_time for each feature.
        We have 10 features thus output is in the dimension of
        n_edges x 10
        The input order is consistent with the order in description in DNNCal
        q_u(t), q_d(t), p(t), v(t),
        and parameters tau_0, tau_w, C_bar, Q_bar,
        and the pre-defined demand D(t),
        and the link priority.
        :return: concatenated input
        """
        #qu = qu_nn.reshape((self.n_edges, 1))
        qd = qd_nn.reshape((self.n_edges, 1))
        p = p_nn.reshape((self.n_edges, 1))
        v = v_nn.reshape((self.n_edges, 1))
        tau_0 = tau_0.reshape((self.n_edges, 1)).to(self.device)
        tau_w = tau_w.reshape((self.n_edges, 1)).to(self.device)
        C_bar = C_bar.reshape((self.n_edges, 1)).to(self.device)
        #Q_bar = Q_bar.reshape((self.n_edges, 1)).to(self.device)
        demand = self.demand.reshape((self.n_edges, total_time)).to(self.device)
        priority = self.priority.reshape((self.n_edges, 1))

        inputs = torch.cat([qd, p, v,
                            tau_0, tau_w, C_bar,
                            priority], dim=1)
        return inputs

    def decompose_outputs_cal(self, zipped_output):
        """
        Reshape the n_edges x 8 array into
        n_edges x 1 for each feature.
        We have 10 features thus output is in the dimension of
        n_edges x 8.
        The output order is consistent with the order in description in DNNCal
        q_u(t+1), q_d(t+1), p(t+1), v(t+1),
        and the parameters we need from the calibration process: tau_0, tau_omega, Cbar, Qbar.
        :return:  q_u, q_d, p, v, tau_0, tau_omega, Cbar, Qbar
        """
        #qu = zipped_output[:, :total_time]
        qd = zipped_output[:, :total_time*n_demand]
        p = zipped_output[:, total_time*n_demand:2*total_time*n_demand]
        v = zipped_output[:, 2*total_time*n_demand:3*total_time*n_demand]
        # tau_0 = zipped_output[:, 3*total_time*n_demand]
        tau_w = zipped_output[:, 3*total_time*n_demand+0]
        C_bar = zipped_output[:, 3*total_time*n_demand+1]
        #Q_bar=(tau_0+tau_w)*C_bar
        #Q_bar = zipped_output[:, 3*total_time*n_demand+3]
        return qd, p, v, tau_w, C_bar

    def decompose_outputs_hid(self, flow, beta, tt, gamma):
        """
        Reshape the n_edges x 6 array into
        n_edges x 1 for each feature.
        We have 10 features thus output is in the dimension of
        n_edges x 6.
        The output order is consistent with the order in description in DNNCal
        q_u(t+1), q_d(t+1), p(t+1), v(t+1), mu(t+1), delta(t+1)
        :return:  q_u, q_d, p, v, mu, delta
        """
        qd = flow[:, 0:(n_links)]
        p = flow[:, (n_links):2 * (n_links)]
        v = flow[:, 2 * (n_links):3 * (n_links)]
        tt = tt
        beta = beta
        gamma = gamma
        return qd, p, v, beta, tt, gamma

    def save_gams_params(self, Demand, tau_0_cal, tau_w_cal, C_bar_cal):
        with open("gamsCapa_config_and_param.gms", "w") as w:
            w.write("set nodes /{}*{}/;\n".format(int(list(sorted(self.nodes.keys()))[0]),
                                                  int(list(sorted(self.nodes.keys()))[-1])))
            w.write("alias(nodes, n, i, j, k, l, kk);\n")
            w.write("set desti(nodes) /{}/;\n".format(destination_node))
            w.write("set dummyori(nodes) /{}/;\n".format(origin_node_dummy))
            w.write("set dummydesti(nodes) /{}/;\n".format(destination_node_dummy))
            w.write("\n")
            w.write("parameter hlength /1/;\n")
            w.write("set h /1*{}/;\n".format(total_time))
            w.write("alias(h, hh, s, r);\n")
            w.write("set links (i, j) /\n")
            for link in self.adj:
                w.write("{}.{}\n".format(int(link[0]), int(link[1])))
            w.write("/;\n")
            w.write("\n")

            w.write("parameter tao0(i,j)/\n")
            write_param(w, tau_0_cal, self.adj)
            w.write("parameter nh(i,j);\n")
            w.write("nh(i,j) = tao0(i,j);\n")
            w.write("\n")
            w.write("parameter nomegah(i,j)/\n")
            write_param(w, tau_w_cal, self.adj)
            w.write("parameter Cbar(i,j)/\n")
            write_param(w, C_bar_cal, self.adj)
            w.write("""parameter Qbar(i,j);
            Qbar(i,j)$(not dummydesti(j) and not dummyori(i) and links(i,j)) = Cbar(i,j)*(nh(i,j)+nOmegah(i,j))*hlength;
            Qbar(i,j)$(dummyori(i) and links(i,j)) = 5000;
            Qbar(i,j)$(dummydesti(j) and links(i,j)) = 5000;\n""")
            #write_param(w, C_bar_cal*(tau_0_cal+tau_w_cal), self.adj)

            w.write("parameter nhead(i) /\n")
            for node_id, node in self.nodes.items():
                w.write("{} {}\n".format(int(node_id), node.nhead.item()))
            w.write("/;\n")
            w.write("\n")
            w.write("parameter nbeforehead(i)/\n")
            for node_id, node in self.nodes.items():
                w.write("{} {}\n".format(int(node_id), node.nbeforehead.item()))
            w.write("/;\n")
            w.write("\n")
            w.write("parameter pi0(i)/\n")
            for node_id, node in self.nodes.items():
                w.write("{} {}\n".format(int(node_id), node.pi0.item()))
            w.write("/;\n")
            w.write("\n")
            # TODO: make this part general later on
            w.write("""
parameter D0(i) /
1 0
2 0
3 0
4 0
5 0
6 0
7 0
/;
parameter Dbar(i) /
1 0
2 0
3 0
4 0
5 0
6 0
7 {}
/;
*100

parameter lambdaup(i) /
1 1000
2 1000
3 1000
4 1000
5 1000
6 1000
7 1000
/;

parameter alpha /0.5/;
parameter gamma /1/;
parameter Delta /5/;
parameter R_para(i) /
1 20
2 20
3 20
4 20
5 20
6 20
7 35
/;
*parameter epsilon /1e-3/;
parameter epsilon2 /1e-3/;
*parameter epsilon2 /0/;
*parameter epsilon4 /1/;


$ontext
parameter d(i,r)/
5.h1 0
5.h2 2
5.h3 4
5.h4 6
5.h5 7
5.h6 7.5
5.h7 7
5.h8 6
5.h9 4
5.h10 2
/;
$offtext


\n""".format(Demand))

    def loss_func_cal(self):
        # assign observed flow
        #ratio_qu_obs = self.qu_ob_norm.clone()
        ratio_qd_obs = self.qd_ob_norm.clone()
        ratio_p_obs = self.p_ob_norm.clone()
        ratio_v_obs = self.v_ob_norm.clone()

        # obtain the initial/updated parameters
        if self.iter == 0:
            # ratio_tau_0_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
            # ratio_tau_w_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
            # ratio_C_bar_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
            # #ratio_Q_bar_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
            #
            # tau_0_cal = denormalize_max_and_min(ratio_tau_0_cal, self.tau_0_cal_max, self.tau_0_cal_min)
            # tau_w_cal = denormalize_max_and_min(ratio_tau_w_cal, self.tau_w_cal_max, self.tau_w_cal_min)
            # C_bar_cal = denormalize_max_and_min(ratio_C_bar_cal, self.C_bar_cal_max, self.C_bar_cal_min)
            # #Q_bar_cal = denormalize_max_and_min(ratio_Q_bar_cal, self.Q_bar_cal_max, self.Q_bar_cal_min)

            tau_0_cal = torch.tensor([3.0, 4.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0])
            tau_w_cal = torch.tensor([6.0, 8.0, 2.0, 2.0, 4.0, 2.0, 0.0, 0.0])
            C_bar_cal = torch.tensor([25.0, 35.0, 10.0, 10.0, 5.0, 10.0, 1000.0, 0.0])

        else:
            params = scipy.io.loadmat("./output/params.mat")
            tau_0_cal = torch.tensor([3.0, 4.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0])
            tau_w_cal = params["tau_w_cal"]
            C_bar_cal = params["C_bar_cal"]
            #Q_bar_cal = params["Q_bar_cal"]

            tau_0_cal = torch.tensor(tau_0_cal).float().to(self.device).reshape(self.n_edges)
            tau_w_cal = torch.tensor(tau_w_cal).float().to(self.device).reshape(self.n_edges)
            C_bar_cal = torch.tensor(C_bar_cal).float().to(self.device).reshape(self.n_edges)
            # Q_bar_cal = torch.tensor(Q_bar_cal).float().to(self.device).reshape(self.n_edges)

            # predict flow using DTA model

            # TODO: general
        p = np.zeros((8, n_demand * total_time))
        v = np.zeros((8, n_demand * total_time))
        qu = np.zeros((8, n_demand * total_time))
        qd = np.zeros((8, n_demand * total_time))
        mu = np.zeros((8, n_demand * total_time))
        delta = np.zeros((8, n_demand * total_time))
        beta = np.zeros((8, n_demand * total_time))
        bb = np.zeros((8, total_time))
        # tau_0_phy = np.array(tau_0_cal.clone()).round()
        # C_bar_phy = np.array(C_bar_cal.clone()).round()
        for n in range(1, n_demand + 1):
            self.update_params(D[n - 1], tau_0_cal, tau_w_cal, C_bar_cal)
            eng = matlab.engine.start_matlab()
            eng.run(nargout=0)
            f = "./data/dta.mat"
            physics = scipy.io.loadmat(f)

            physics['p_save'] = np.vstack((physics['p_save'], np.zeros((2, total_time))))
            physics['v_save'] = np.vstack((physics['v_save'], np.zeros((2, total_time))))
            physics['qu_save'] = np.vstack((physics['qu_save'], np.zeros((2, total_time))))
            physics['qd_save'] = np.vstack((physics['qd_save'], np.zeros((2, total_time))))
            physics['mu_save'] = np.vstack((physics['mu_save'], np.zeros((2, total_time))))
            physics['delta_save'] = np.vstack((physics['delta_save'], np.zeros((2, total_time))))
            # for m in range(8):
            t0=[3.0, 4.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0]
            #t0 = round(max(tau_0_cal).item())
            cbar = np.round(C_bar_cal.cpu().numpy())
            for i in range(cbar.shape[0]):
                t00=int(t0[i])
                bb[i, 0:total_time - t00] = (physics['qd_save'][i, t00:total_time] / cbar[i] * (
                        1 + (physics['mu_save'][i, t00:total_time] + physics['delta_save'][i, t00:total_time]) / cbar[i]))
            bb = np.nan_to_num(bb)
            # bb[:, 0:total_time - t0] = bb[:, 0:total_time - t0] + tau_0_cal.unsqueeze(1).round().cpu().numpy()
            # bb[0:6, total_time - t0:total_time] = np.repeat(t0, t0)
            p[:, (n - 1) * total_time:n * total_time] = physics['p_save']
            v[:, (n - 1) * total_time:n * total_time] = physics['v_save']
            qu[:, (n - 1) * total_time:n * total_time] = physics['qu_save']
            qd[:, (n - 1) * total_time:n * total_time] = physics['qd_save']
            mu[:, (n - 1) * total_time:n * total_time] = physics['mu_save']
            delta[:, (n - 1) * total_time:n * total_time] = physics['delta_save']
            beta[:, (n - 1) * total_time:n * total_time] = bb

            check = D[n - 1]
            print(f'\n{n}')

        qu_phy = torch.tensor(qu).float().to(self.device)
        qd_phy = torch.tensor(qd).float().to(self.device)
        p_phy = torch.tensor(p).float().to(self.device)
        v_phy = torch.tensor(v).float().to(self.device)
        mu_phy = torch.tensor(mu).float().to(self.device)
        delta_phy = torch.tensor(delta).float().to(self.device)
        beta_phy = torch.tensor(beta).float().to(self.device)

        self.qu_phy = qu_phy.clone()
        self.qd_phy = qd_phy.clone()
        self.p_phy = p_phy.clone()
        self.v_phy = v_phy.clone()
        self.mu_phy = mu_phy.clone()
        self.delta_phy = delta_phy.clone()
        self.beta_phy = beta_phy.clone()

        ratio_qu_phy = normalize_max_and_min(qu_phy, self.qu_max, self.qu_min)
        ratio_qd_phy = normalize_max_and_min(qd_phy, self.qd_max, self.qd_min)
        ratio_p_phy = normalize_max_and_min(p_phy, self.p_max, self.p_min)
        ratio_v_phy = normalize_max_and_min(v_phy, self.v_max, self.v_min)
        ratio_mu_phy = normalize_max_and_min(mu_phy, self.mu_max, self.mu_min)
        ratio_delta_phy = normalize_max_and_min(delta_phy, self.delta_max, self.delta_min)
        # TODO: change max and min for beta time later
        ratio_beta_phy = normalize_max_and_min(beta_phy, self.beta_max, self.beta_min)

        self.qu_phy_norm = ratio_qu_phy
        self.qd_phy_norm = ratio_qd_phy
        self.p_phy_norm = ratio_p_phy
        self.v_phy_norm = ratio_v_phy
        self.mu_phy_norm = ratio_mu_phy
        self.delta_phy_norm = ratio_delta_phy
        self.beta_phy_norm = ratio_beta_phy

        # predict flow using NN calibration
        # and using NN hidden (note: NN#2 estimation is based on each previous condition of DTA output)
        # ratio_qu_nn_cal = self.qu_ob_norm[:, 0].clone()
        # shaki: it only takes first column -> p,v,qd at t=0
        ratio_qd_nn_cal = self.qd_ob_norm[:, 0].clone()
        ratio_p_nn_cal = self.p_ob_norm[:, 0].clone()
        ratio_v_nn_cal = self.v_ob_norm[:, 0].clone()

        # normalize
        ratio_tau_0_cal = normalize_max_and_min(tau_0_cal, self.tau_0_cal_max, self.tau_0_cal_min)
        ratio_tau_w_cal = normalize_max_and_min(tau_w_cal, self.tau_w_cal_max, self.tau_w_cal_min)
        ratio_C_bar_cal = normalize_max_and_min(C_bar_cal, self.C_bar_cal_max, self.C_bar_cal_min)
        # ratio_Q_bar_cal = normalize_max_and_min(Q_bar_cal, self.Q_bar_cal_max, self.Q_bar_cal_min)

        # inputs_cal = self.zip_inputs_cal(self.i_time,
        #                                  ratio_qu_nn_cal, ratio_qd_nn_cal, ratio_p_nn_cal, ratio_v_nn_cal,
        #                                  ratio_tau_0_cal, ratio_tau_w_cal, ratio_C_bar_cal, ratio_Q_bar_cal)
        inputs_cal = self.zip_inputs_cal(self.i_time,
                                         ratio_qd_nn_cal, ratio_p_nn_cal, ratio_v_nn_cal,
                                         ratio_tau_0_cal, ratio_tau_w_cal, ratio_C_bar_cal)
        output_cal = self.nn_cal(inputs_cal)

        # ratio_qu_nn_cal, \
        ratio_qd_nn_cal, \
        ratio_p_nn_cal, \
        ratio_v_nn_cal, \
        ratio_tau_w_cal, \
        ratio_C_bar_cal = self.decompose_outputs_cal(output_cal.clone())

        if check == 0:
            log = "Infeasible!"
            print(log)
        #     ratio_tau_0_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
        #     ratio_tau_w_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
        #     ratio_C_bar_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)
        #     #ratio_Q_bar_cal = produce_initial_guess(self.n_edges, 0.0, 1.0)

        #tau_0_cal = denormalize_max_and_min(ratio_tau_0_cal, self.tau_0_cal_max, self.tau_0_cal_min)
        tau_w_cal = denormalize_max_and_min(ratio_tau_w_cal, self.tau_w_cal_max, self.tau_w_cal_min)
        C_bar_cal = denormalize_max_and_min(ratio_C_bar_cal, self.C_bar_cal_max, self.C_bar_cal_min)

        # tau_0_cal = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        # tau_w_cal = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0])
        # C_bar_cal = torch.tensor([78.0, 78.0, 78.0, 78.0, 24.0, 24.0, 378.0, 0.0])
        # Q_bar_cal = denormalize_max_and_min(ratio_Q_bar_cal, self.Q_bar_cal_max, self.Q_bar_cal_min)

        params = {"tau_0_cal": tau_0_cal.detach().cpu().numpy(),
                  "tau_w_cal": tau_w_cal.detach().cpu().numpy(),
                  "C_bar_cal": C_bar_cal.detach().cpu().numpy()}

        scipy.io.savemat("./output/params.mat", params)

        if check == 0:
            ratio_qu_phy = 2
            ratio_qd_phy = 2
            ratio_p_phy = 2
            ratio_v_phy = 2

        # loss_obs = (torch.mean((ratio_qu_nn_cal - ratio_qu_obs) ** 2) +
        #             torch.mean((ratio_qd_nn_cal - ratio_qd_obs) ** 2) +
        #             torch.mean((ratio_p_nn_cal - ratio_p_obs) ** 2) +
        #             torch.mean((ratio_v_nn_cal - ratio_v_obs) ** 2))

        loss_obs = (
                torch.mean((ratio_qd_nn_cal - ratio_qd_obs) ** 2) +
                torch.mean((ratio_p_nn_cal - ratio_p_obs) ** 2) +
                torch.mean((ratio_v_nn_cal - ratio_v_obs) ** 2))

        # loss_phy = (torch.mean((ratio_qu_nn_cal - ratio_qu_phy) ** 2) +
        #             torch.mean((ratio_qd_nn_cal - ratio_qd_phy) ** 2) +
        #             torch.mean((ratio_p_nn_cal - ratio_p_phy) ** 2) +
        #             torch.mean((ratio_v_nn_cal - ratio_v_phy) ** 2))
        loss_phy = (
                torch.mean((ratio_qd_nn_cal - ratio_qd_phy) ** 2) +
                torch.mean((ratio_p_nn_cal - ratio_p_phy) ** 2) +
                torch.mean((ratio_v_nn_cal - ratio_v_phy) ** 2))

        loss = 0.5 * loss_obs + 0.5 * loss_phy

        # logger.save_stats(self.iter, loss_obs, loss_phy, loss_hid, loss)
        log = 'Iter %d, ' \
              'Loss_O: %.5f, ' \
              'Loss_P: %.5f, ' \
              'Total Loss: %.5f' % (
                  self.iter,
                  loss_obs.detach().cpu().numpy(),
                  loss_phy.detach().cpu().numpy(),
                  loss.item())
        print(log)
        loss_record_all_cal.append(loss.detach().cpu().numpy())
        with open(f'log{self.iter}_NN1.txt', 'w') as l:
            l.write(log)
        # backward and optimize
        self.optimizer_cal.zero_grad()
        loss.backward()
        return loss

    def loss_func_hid(self):
        """me"""

        # obtain the initial/updated parameters
        params = scipy.io.loadmat("./output/params.mat")
        tau_0_cal = params["tau_0_cal"]
        tau_w_cal = params["tau_w_cal"]
        C_bar_cal = params["C_bar_cal"]
        # Q_bar_cal = params["Q_bar_cal"]

        tau_0_cal = torch.tensor(tau_0_cal).float().to(self.device)
        tau_w_cal = torch.tensor(tau_w_cal).float().to(self.device)
        C_bar_cal = torch.tensor(C_bar_cal).float().to(self.device)
        # Q_bar_cal = torch.tensor(Q_bar_cal).float().to(self.device)

        ratio_tau_0_cal = normalize_max_and_min(tau_0_cal, self.tau_0_cal_max, self.tau_0_cal_min)
        ratio_tau_w_cal = normalize_max_and_min(tau_w_cal, self.tau_w_cal_max, self.tau_w_cal_min)
        ratio_C_bar_cal = normalize_max_and_min(C_bar_cal, self.C_bar_cal_max, self.C_bar_cal_min)
        # ratio_Q_bar_cal = normalize_max_and_min(Q_bar_cal, self.Q_bar_cal_max, self.Q_bar_cal_min)

        # bb[:, 0:total_time - t0] = bb[:, 0:total_time - t0] + tau_0_cal.unsqueeze(1).round().cpu().numpy()
        # bb[0:6, total_time - t0:total_time] = np.repeat(t0, t0)

        ratio_qd_obs = torch.tensor([]).to(self.device)
        ratio_p_obs = torch.tensor([]).to(self.device)
        ratio_v_obs = torch.tensor([]).to(self.device)
        ratio_tt_obs = torch.tensor([]).to(self.device)

        ratio_qd_obs = torch.cat((ratio_qd_obs, self.qd_ob_norm[:, 0:total_time * n_demand - 1].clone()), dim=1)
        ratio_p_obs = torch.cat((ratio_p_obs, self.p_ob_norm[:, 0:total_time * n_demand - 1].clone()), dim=1)
        ratio_v_obs = torch.cat((ratio_v_obs, self.v_ob_norm[:, 0:total_time * n_demand - 1].clone()), dim=1)
        ratio_tt_obs = torch.cat((ratio_tt_obs, self.tt_ob_norm[:, 0:total_time * n_demand - 1].clone()), dim=1)

        # ratio_qu_phy = self.qu_phy_norm.clone()
        ratio_qd_phy = self.qd_phy_norm.clone()
        ratio_p_phy = self.p_phy_norm.clone()
        ratio_v_phy = self.v_phy_norm.clone()
        # ratio_delta_phy = self.delta_phy_norm.clone()
        ratio_beta_phy = self.beta_phy_norm.clone()

        # predict flow using NN calibration
        # and using NN hidden (note: NN#2 estimation is based on each previous condition of DTA output)
        # ratio_qu_nn_hid = ratio_qu_phy[:, self.i_time - 1:self.i_time + time_window - 2].clone()
        # TODO: add different demand scenarios
        ratio_qd_phy_0 = torch.tensor([]).to(self.device)
        ratio_p_phy_0 = torch.tensor([]).to(self.device)
        ratio_v_phy_0 = torch.tensor([]).to(self.device)
        ratio_beta_phy_0 = torch.tensor([]).to(self.device)

        ratio_qd_phy_0 = torch.cat((ratio_qd_phy_0, ratio_qd_phy[:, 0:total_time * n_demand - 1].clone()), dim=1)
        ratio_p_phy_0 = torch.cat((ratio_p_phy_0, ratio_p_phy[:, 0:total_time * n_demand - 1].clone()), dim=1)
        ratio_v_phy_0 = torch.cat((ratio_v_phy_0, ratio_v_phy[:, 0:total_time * n_demand - 1].clone()), dim=1)
        ratio_beta_phy_0 = torch.cat((ratio_beta_phy_0, ratio_beta_phy[:, 0:total_time * n_demand - 1].clone()), dim=1)
        # ratio_delta_nn_hid = ratio_delta_phy[:, self.i_time - 1:self.i_time + time_window - 2].clone()

        inputs_hid_col = self.zip_inputs_hid(self.i_time,
                                             ratio_qd_phy_0, ratio_p_phy_0, ratio_v_phy_0, ratio_beta_phy_0)
        output_hid_col_flow, output_hid_col_beta, output_hid_col_traveltime, output_hid_col_gamma = self.nn_hid(
            inputs_hid_col)

        ratio_qd_nn_hid_phy, \
        ratio_p_nn_hid_phy, \
        ratio_v_nn_hid_phy, \
        ratio_beta_nn_hid_phy, \
        ratio_traveltime_nn_hid_phy, \
        ratio_gamma_nn_hid = self.decompose_outputs_hid(output_hid_col_flow.clone(), output_hid_col_beta.clone(),
                                                        output_hid_col_traveltime.clone(),
                                                        output_hid_col_gamma.clone())

        # beta_nn_hid = denormalize_max_and_min(ratio_beta_nn_hid_phy, self.beta_max, self.beta_min)

        ratio_qd_phy_1 = torch.tensor([]).to(self.device)
        ratio_p_phy_1 = torch.tensor([]).to(self.device)
        ratio_v_phy_1 = torch.tensor([]).to(self.device)
        ratio_beta_phy_1 = torch.tensor([]).to(self.device)

        ratio_qd_phy_1 = torch.transpose(torch.cat((ratio_qd_phy_1, ratio_qd_phy[:, 1:total_time * n_demand].clone()), dim=1),0,1)
        ratio_p_phy_1 = torch.transpose(torch.cat((ratio_p_phy_1, ratio_p_phy[:, 1:total_time * n_demand].clone()), dim=1),0,1)
        ratio_v_phy_1 = torch.transpose(torch.cat((ratio_v_phy_1, ratio_v_phy[:, 1:total_time * n_demand].clone()), dim=1),0,1)
        ratio_beta_phy_1 = torch.transpose(torch.cat((ratio_beta_phy_1, ratio_beta_phy[:, 1:total_time * n_demand].clone()), dim=1),0,1)

        mask_qd_nn = ratio_qd_nn_hid_phy != 0
        mask_qd_phy = ratio_qd_phy_1 != 0
        ratio_qd_nn_hid_phy = ratio_qd_nn_hid_phy[mask_qd_nn & mask_qd_phy]
        ratio_qd_phy_1 = ratio_qd_phy_1[mask_qd_nn & mask_qd_phy]

        mask_p_nn = ratio_p_nn_hid_phy != 0
        mask_p_phy = ratio_p_phy_1 != 0
        ratio_p_nn_hid_phy = ratio_p_nn_hid_phy[mask_p_nn & mask_p_phy]
        ratio_p_phy_1 = ratio_p_phy_1[mask_p_nn & mask_p_phy]

        mask_v_nn = ratio_v_nn_hid_phy != 0
        mask_v_phy = ratio_v_phy_1 != 0
        ratio_v_nn_hid_phy = ratio_v_nn_hid_phy[mask_v_nn & mask_v_phy]
        ratio_v_phy_1 = ratio_v_phy_1[mask_v_nn & mask_v_phy]

        mask_beta_nn = ratio_beta_nn_hid_phy != 0
        mask_beta_phy = ratio_beta_phy_1 != 0
        ratio_beta_nn_hid_phy = ratio_beta_nn_hid_phy[mask_beta_nn & mask_beta_phy]
        ratio_beta_phy_1 = ratio_beta_phy_1[mask_beta_nn & mask_beta_phy]
        # t= torch.mean((ratio_qd_nn_hid_phy - ratio_qd_phy_1) ** 2)
        # nan_mask = torch.isnan(t)
        # t[nan_mask]=0.0
        # reshape and merge variables to make the previous shape (decompose output_hid_obs_flow and see corresponding obs)
        loss_hid_col = (torch.mean((ratio_qd_nn_hid_phy - ratio_qd_phy_1) ** 2) +
                        torch.mean((ratio_p_nn_hid_phy - ratio_p_phy_1) ** 2) +
                        torch.mean((ratio_v_nn_hid_phy - ratio_v_phy_1) ** 2) +
                        torch.mean((ratio_beta_nn_hid_phy - ratio_beta_phy_1) ** 2))
        print(torch.mean((ratio_beta_nn_hid_phy - ratio_beta_phy_1) ** 2))
        # if torch.mean((ratio_beta_nn_hid - ratio_beta_phy[:, self.i_time + time_window - 1:self.i_time + 2 * time_window - 1]) ** 2)<0.1:
        inputs_hid_obs = self.zip_inputs_hid(self.i_time,
                                             ratio_qd_obs, ratio_p_obs, ratio_v_obs, ratio_tt_obs)
        output_hid_obs_flow, output_hid_obs_beta, output_hid_obs_traveltime, output_hid_obs_gamma = self.nn_hid(inputs_hid_obs)
        ratio_qd_nn_hid_obs, \
        ratio_p_nn_hid_obs, \
        ratio_v_nn_hid_obs, \
        ratio_beta_nn_hid_obs, \
        ratio_traveltime_nn_hid_obs, \
        gamma_nn_hid = self.decompose_outputs_hid(output_hid_obs_flow.clone(), output_hid_obs_beta.clone(),
                                                  output_hid_obs_traveltime.clone(),
                                                  output_hid_obs_gamma.clone())

        # t0 = round(torch.max(tau_0_cal).item())
        # tt0 = torch.round(tau_0_cal).transpose(0, 1)
        # cbar = torch.round(C_bar_cal).transpose(0, 1)
        # bb[m,0:total_time-t0]=t0+physics['qd_save'][m,t0:total_time]/cbar*(1+(physics['mu_save'][m,t0:total_time]+physics['delta_save'][m,t0:total_time])/cbar)
        # bb[m,total_time-t0:total_time] = np.repeat(t0,t0)
        # bb[:, 0:total_time - t0] = (1 + (physics['mu_save'][:, t0:total_time] + physics['delta_save'][:, t0:total_time]) / cbar[:,np.newaxis])
        # bb[:, total_time - t0:total_time] = np.repeat(0, t0)
        # bb = np.nan_to_num(bb)
        # self.tt_ob_norm = normalize_max_and_min(self.tt_ob, self.tt_max, self.tt_min)
        # tau_0_cal = denormalize_max_and_min(ratio_tau_0_cal, self.tau_0_cal_max, self.tau_0_cal_min)
        # qd_nn_hid = denormalize_max_and_min(ratio_qd_nn_hid_obs, self.qd_max, self.qd_min)
        # beta_nn_hid = denormalize_max_and_min(ratio_beta_nn_hid_obs, self.beta_max, self.beta_min)

        ratio_qd_obs_1 = torch.tensor([]).to(self.device)
        ratio_p_obs_1 = torch.tensor([]).to(self.device)
        ratio_v_obs_1 = torch.tensor([]).to(self.device)
        ratio_tt_obs_1 = torch.tensor([]).to(self.device)

        ratio_qd_obs_1 = torch.transpose(torch.cat((ratio_qd_obs_1, self.qd_ob_norm[:, 1:total_time * n_demand].clone()), dim=1),0,1)
        ratio_p_obs_1 = torch.transpose(torch.cat((ratio_p_obs_1, self.p_ob_norm[:, 1:total_time * n_demand].clone()), dim=1),0,1)
        ratio_v_obs_1 = torch.transpose(torch.cat((ratio_v_obs_1, self.v_ob_norm[:, 1:total_time * n_demand].clone()), dim=1),0,1)
        ratio_tt_obs_1 = torch.transpose(torch.cat((ratio_tt_obs_1, self.tt_ob_norm[:, 1:total_time * n_demand].clone()), dim=1),0,1)

        mask_qd_nn = ratio_qd_nn_hid_obs != 0
        mask_qd_obs = ratio_qd_obs_1 != 0
        ratio_qd_nn_hid_obs = ratio_qd_nn_hid_obs[mask_qd_nn & mask_qd_obs]
        ratio_qd_obs_1 = ratio_qd_obs_1[mask_qd_nn & mask_qd_obs]

        mask_p_nn = ratio_p_nn_hid_obs != 0
        mask_p_obs = ratio_p_obs_1 != 0
        ratio_p_nn_hid_obs = ratio_p_nn_hid_obs[mask_p_nn & mask_p_obs]
        ratio_p_obs_1 = ratio_p_obs_1[mask_p_nn & mask_p_obs]

        mask_v_nn = ratio_v_nn_hid_obs != 0
        mask_v_obs = ratio_v_obs_1 != 0
        ratio_v_nn_hid_obs = ratio_v_nn_hid_obs[mask_v_nn & mask_v_obs]
        ratio_v_obs_1 = ratio_v_obs_1[mask_v_nn & mask_v_obs]

        mask_traveltime_nn = ratio_traveltime_nn_hid_obs != 0
        mask_traveltime_obs = ratio_tt_obs_1 != 0
        ratio_traveltime_nn_hid_obs = ratio_traveltime_nn_hid_obs[mask_traveltime_nn & mask_traveltime_obs]
        ratio_tt_obs_1 = ratio_tt_obs_1[mask_traveltime_nn & mask_traveltime_obs]

        loss_hid_obs = (torch.mean((ratio_qd_nn_hid_obs - ratio_qd_obs_1) ** 2) +
                        torch.mean((ratio_p_nn_hid_obs - ratio_p_obs_1) ** 2) +
                        torch.mean((ratio_v_nn_hid_obs - ratio_v_obs_1) ** 2) +
                        torch.mean((ratio_traveltime_nn_hid_obs - ratio_tt_obs_1) ** 2))
        loss = 0.5 * loss_hid_obs + 0.5 * loss_hid_col

        # else:
        #     loss_hid_obs= loss_hid_col
        #     loss = loss_hid_col

        log = 'Iter %d, ' \
              'Loss_O: %.5f, ' \
              'Loss_P: %.5f, ' \
              'Total Loss: %.5f, ' \
              'Gamma: %.5f' % (
                  self.iter,
                  loss_hid_obs.detach().cpu().numpy(),
                  loss_hid_col.detach().cpu().numpy(),
                  loss.item(),
                  gamma_nn_hid,
              )
        print(log)
        loss_record_all_hid.append(loss.detach().cpu().numpy())
        with open(f'log{self.iter}_NN2.txt', 'w') as p:
            p.write(log)

        # backward and optimize
        self.optimizer_hid.zero_grad()
        loss.backward()
        return loss

    def train(self):
        self.nn_cal.train()
        self.nn_hid.train()
        i_record = 0
        for epoch in range(1, n_iters):
            self.i_time += 1
            if epoch <= 2*self.total_time:
            # if epoch <= 1:
                #     # TODO: Should we increase this warm up since we have more data (more demand scenarios)?
                #     # reduce it! because you have more data now and it should converge faster
                #     # warm up the calibration training, then we train both NNs
                #     # shaki: loss_func_cal is main functioning of NN#1

                self.optimizer_cal.step(self.loss_func_cal)
            else:
                # if(self.i_time+2*time_window-1<total_time):
                self.optimizer_hid.step(self.loss_func_hid)
                # else:
                #     self.optimizer_cal.step(self.loss_func_cal)

            # shaki: I think at each epoch this code only runs for one time step in NN#2, and after 60 (=study period) epochs:

            if epoch % (self.total_time) == 0:
                m = sum(loss_record_all_cal[
                        0 + i_record * self.total_time:(i_record + 1) * self.total_time]) / self.total_time
                loss_record_cal.append(m)
                m = sum(loss_record_all_hid[
                        0 + i_record * self.total_time:(i_record + 1) * self.total_time]) / self.total_time
                loss_record_hid.append(m)
                i_record += 1
                self.i_time = 0

            self.iter += 1


def show_results(T, p_predicted, v_predicted,
                 p_ob, v_ob):
    """
    This is a temporary function expected to be removed sometime soon.
    At the moment I just try to plot some preliminary results,
    but we should come up with more meaningful plots based on this function.
    """
    folder = "./output/"
    from_node = 6
    to_node = 7
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.plot(range(len(loss_record_all_cal)), loss_record_all_cal)
    plt.plot(range(len(loss_record_cal)), loss_record_cal)
    plt.figure()
    plt.plot(range(len(loss_record_all_hid)), loss_record_all_hid)
    plt.plot(range(len(loss_record_hid)), loss_record_hid)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    df = pd.DataFrame(loss_record_cal)
    df.to_csv("{}loss_cal.csv".format(folder), index=False)
    err_p = np.zeros(T)
    err_v = np.zeros(T)
    for t in range(T):
        err_p[t] = torch.mean((p_predicted[from_node, to_node, t] - p_ob[from_node, to_node, t]) ** 2)
        err_v[t] = torch.mean((v_predicted[from_node, to_node, t] - v_ob[from_node, to_node, t]) ** 2)
    plt.plot(range(T), p_predicted[from_node, to_node, :].detach().numpy(), label="predicted")
    plt.plot(range(T), p_ob[from_node, to_node, :].detach().numpy(), label="observed")
    plt.legend()
    plt.show()
    plt.savefig("{}train_cal.png")


def read_obs():
    # TODO: make this 8 general
    p = np.zeros((8,n_demand*total_time))
    v = np.zeros((8,n_demand*total_time))
    qd = np.zeros((8,n_demand*total_time))
    tt = np.zeros((8,n_demand*total_time))

    for n in range(1,n_demand+1):
        f = f"./data/obs{n}.mat"
        observed = scipy.io.loadmat(f)
        # TODO: make this 2 general (a network with multiple org dummy)
        fft=np.array([3.0, 4.0, 1.0, 1.0, 2.0, 1.0])
        fft = np.repeat(fft[:, np.newaxis], 60, axis=1)
        observed['p_SUMO']=np.vstack((observed['p_SUMO'], np.zeros((2, total_time))))
        observed['v_SUMO']=np.vstack((observed['v_SUMO'], np.zeros((2, total_time))))
        observed['qd_SUMO']=np.vstack((observed['qd_SUMO'], np.zeros((2, total_time))))
        observed['tt_SUMO']=np.vstack((observed['tt_SUMO']-fft, np.zeros((2, total_time))))
        observed['tt_SUMO'] = np.where(observed['tt_SUMO'] < 0, 0, observed['tt_SUMO'])
        observed['tt_SUMO'] = np.where(observed['tt_SUMO'] > 40, 40, observed['tt_SUMO'])

        D.append(float(observed['D']))
        p[:,(n-1)*total_time:n*total_time] = observed['p_SUMO']/60  #it's veh/hour and should become veh/min
        v[:,(n-1)*total_time:n*total_time] = observed['v_SUMO']/60  #same
        qd[:,(n-1)*total_time:n*total_time] = observed['qd_SUMO']  #number of vehicles at queu for each minute
        tt[:,(n-1)*total_time:n*total_time] = observed['tt_SUMO']  #it's already in minutes
        # mu = observed['mu_save']
        # delta = observed['delta_save']
        # qu = observed['qu_save']
    return p, v, qd, tt


def read_given(f="./data/given.mat"):
    observed = scipy.io.loadmat(f)
    tau_0 = torch.tensor(observed['tau0_save']).float().to(gpu)
    tau_w = torch.tensor(observed['tauw_save']).float().to(gpu)
    observed['Cbar_save']=observed['Cbar_save'].astype('int32')
    C_bar = torch.tensor(observed['Cbar_save']).float().to(gpu)
    observed['Qbar_save']=observed['Qbar_save'].astype('int32')
    #Q_bar = torch.tensor(observed['Qbar_save']).float().to(gpu)
    adj = torch.tensor(observed['edges_save']).float().to(gpu)
    priority = torch.tensor(observed['priority_save']).float().to(gpu)

    # TODO: the demand here is not updated to the chain network (should be n_edges*n_times tensor dimension)
    # padding demand to be the same dimension as other parameters
    demand = observed['d']
    r, c = demand.shape
    demand_pad = np.zeros((tau_0.shape[0], demand.shape[1]))
    demand_pad[:r, :c] = demand
    demand_pad = torch.tensor(demand_pad).float().to(gpu)
    return tau_0, tau_w, C_bar, demand_pad, adj, priority


def network_init(network_name):
    graph = network.make(network_name)
    edges = graph.get_edges()
    nodes = graph.get_nodes()
    node_ids = graph.get_node_ids()
    edge_ids = graph.get_edge_ids()
    tau_0, tau_w, C_bar, demand, adj, priority = read_given()

    # update network
    for i in range(len(edges)):
        edge_id = edge_ids[i]
        edges[edge_id].set_freeflowTravelTime(tau_0[i])
        edges[edge_id].set_shockwaveTravelTime(tau_w[i])
        edges[edge_id].set_priority(priority[i])
        edges[edge_id].update_flowCap(C_bar[i])
        #edges[edge_id].update_queueCap(Q_bar[i])
    for i in range(len(nodes)):
        node_id = node_ids[i]
        #nodes[node_id].set_demand(demand[i, :])
    graph.set_adj(adj)
    graph.refresh(gpu)
    return graph


def gpu_support():
    if torch.cuda.is_available():
        device_support = torch.device('cuda')
    else:
        device_support = torch.device('cpu')
    return device_support


if __name__ == '__main__':
    gpu = gpu_support()
    # initialize network
    # network_name = "six-link"
    network_name = "six-link"
    network = network_init(network_name)
    total_time = 60
    n_demand = 8
    D = []

    # configure the network destination
    destination_node = 5
    origin_node_dummy = 7
    destination_node_dummy = 6
    n_links=8
    time_window = 1
    n_iters = 60000
    loss_record_all_cal = []
    loss_record_all_hid = []
    loss_record_cal = []
    loss_record_hid = []
    layers_nn_cal = [7, 64, 256, n_demand*total_time*3+2]  # see description in DNNCal ### shaki: the inputs are only
    # at t=0 but the outputs are all time instances for p, v , qd
    # layers_nn_hid = [11, 32, 64, 16, 6]  # see description in DNNHid
    # hidden_neurons = 16
    layers_nn_hid = [4 * n_links, 64, 256, 64, 32, 3* n_links, 1* n_links, 1* n_links]
    model = PhysicsInformedNN(gpu)
    logger = TrainLogger("./output/loss_cal.csv")
    model.train()
    print("Done!")