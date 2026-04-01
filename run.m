% assume we just need delta and mu as output
function run()
    clc
    clear all
    N = 60;
    hlength = 1;
    time = 1:hlength:N;
    nNodes =7;
    
    marks = ['k--','k+','k*','kv','k^','ks','ko','k-'];
    
    NLoops=20;
    store_sigma=[];
    strct_sigma = repmat(struct('sigma', [], 'conv_err', []), 1, NLoops);

    %iteration start
    for Iter = 1:NLoops
        %Result from GAMS
        system 'gams DQDUE_Priority_CapIn lo=3 gdx=DQDUE_Priority_CapIn';
        r_links.name = 'links';
        r_links = rgdx('DQDUE_Priority_CapIn', r_links);
        r_N.name = 'Number_h';
        r_N = rgdx('DQDUE_Priority_CapIn', r_N);
        r_p.name = 'p';
        r_p.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_p = rgdx('DQDUE_Priority_CapIn', r_p);
        r_pi.name = 'pi';
        r_pi.uels = {{1:nNodes}, {1:N}};
        r_pi = rgdx('DQDUE_Priority_CapIn', r_pi);
        r_qd.name = 'qd';
        r_qd.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_qd = rgdx('DQDUE_Priority_CapIn', r_qd);
        r_qu.name = 'qu';
        r_qu.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_qu = rgdx('DQDUE_Priority_CapIn', r_qu);
        r_v.name = 'v';
        r_v.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_v = rgdx('DQDUE_Priority_CapIn', r_v);
        r_d.name = 'd';
        r_d.uels = {{1:nNodes}, {1:N}};
        r_d = rgdx('DQDUE_Priority_CapIn', r_d);
        r_lambdah.name = 'lambdah';
        r_lambdah = rgdx('DQDUE_Priority_CapIn', r_lambdah);
        r_F.name = 'paraF';
        r_F = rgdx('DQDUE_Priority_CapIn', r_F);
        r_FPi.name = 'paraFplusPi';
        r_FPi = rgdx('DQDUE_Priority_CapIn', r_FPi);
        r_tao0.name = 'tao0';
        r_tao0 = rgdx('DQDUE_Priority_CapIn', r_tao0);
        r_Cbar.name = 'Cbar';
        r_Cbar = rgdx('DQDUE_Priority_CapIn', r_Cbar);
        r_Qbar.name = 'Qbar';
        r_Qbar = rgdx('DQDUE_Priority_CapIn', r_Qbar);
        r_deltap.name = 'paraPiEq';
        r_deltap = rgdx('DQDUE_Priority_CapIn', r_deltap);  
        r_eta.name = 'eta';
        r_eta.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_eta = rgdx('DQDUE_Priority_CapIn', r_eta);
        r_delta.name = 'deltah';
        r_delta.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_delta = rgdx('DQDUE_Priority_CapIn', r_delta);
        r_deltabar.name = 'deltabar';
        r_deltabar.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_deltabar = rgdx('DQDUE_Priority_CapIn', r_deltabar);
        r_mu.name = 'mu';
        r_mu.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_mu = rgdx('DQDUE_Priority_CapIn', r_mu);   
        r_etap.name = 'etap';
        r_etap.uels = {{1:nNodes}, {1:nNodes}, {1:N}};
        r_etap = rgdx('DQDUE_Priority_CapIn', r_etap);   
        
        links = zeros(nNodes,nNodes);
        for i = 1:length(r_links.val)
            pairi = r_links.val(i, 1);
            pairt = r_links.val(i, 2);
            links(pairi, pairt) = true;
        end
        p = resize(r_p, nNodes, N);
        pi = zeros(nNodes,N);
        for i = 1:length(r_pi.val)
            pairi = r_pi.val(i, 1);
            pairt = r_pi.val(i, 2);
            pi(pairi, pairt) = r_pi.val(i, 3);
        end
        qu = resize(r_qu, nNodes, N);
        qd = resize(r_qd, nNodes, N);
        v_r = resize(r_v, nNodes, N);
        d = zeros(nNodes,N);
        for i = 1:length(r_d.val)
            pairi = r_d.val(i, 1);
            pairt = r_d.val(i, 2);
            d(pairi, pairt) = r_d.val(i, 3);
        end
        lambdah = 1000-r_lambdah.val;
        F = r_F.val;
        FPi = r_FPi.val;
        tao0 = r_tao0.val;
        Cbar = r_Cbar.val;
        Qbar = r_Qbar.val;
        deltap = r_deltap.val;
        eta = resize(r_eta, nNodes, N);
        delta = resize(r_delta, nNodes, N);
        deltabar = resize(r_deltabar, nNodes, N);
        mu = resize(r_mu, nNodes, N);
        etap = resize(r_etap, nNodes, N);
        
        %gams('DUE_UnCa_SF');
        %Read results from matsol
        %[ p pi q v tao0 C deltap] = ReadResult();
        %Adjustp;
        %Eliminate extra p (for flow conservation)
        %Calculate sigma and Write into gams file
        sigma=WriteSigma(p,pi,qd,tao0,Cbar,time);
        strct_sigma(Iter).sigma = sigma;
        strct_sigma(Iter).conv_err = ConvErr(strct_sigma, Iter); 
        
    %     
    %     i=2;j=3;
    % 	figure(500012)
    % 	axis([1 N -0.2 0.4]);
    %     hold on;
    %     plotfunc(sigma,i,j,marks(Iter));
    %     str_y = springf('PD \sigma_{%d,%d}', i,j);
    %     ylabel(str_y);
    %     str = sprintf('Iteration %d', Iter);
    %     legend(str);
    
        fprintf('Iteration %d DONE! \n',Iter);
        %Pics
        %link 3
        %i=1;j=3;
        %link 1
    %      i=1;j=3;
    %      figure(Iter*1000+i*10+j)
    %      plotfunc(p,i,j,'k-');
    %      title('p_{13}');
    %      axis([1 N 0 80]);
    %      
    %      figure(Iter*100+i)
    %      plotfunc2(pi,i,'k-');
    %      title('\pi_1');
    %      axis([1 N 0 40]);
    %      
    %      figure(Iter*100+j)
    %      plotfunc2(pi,j,'k-');
    %      title('\pi_3');
    %      axis([1 N 0 40]);
    %      
    %      figure(Iter*10000+i*10+j)
    %      sigma_fig=zeros(1,size(sigma,3));
    %     for r=1:size(sigma,3)
    %         sigma_fig(r) = sigma(i,j,r);
    %     end
    %     plot(sigma_fig,'k-');
    %     title('\sigma_{13}');
    %     axis([1 N -0.3 1]);
    %     store_sigma = [store_sigma;sigma_fig];
    %     
    %      figure(8800+i*10+j)
    %      if(Iter==5)
    %          sigma_fig=zeros(1,size(sigma,3));
    %         for r=1:size(sigma,3)
    %             sigma_fig(r) = sigma(i,j,r);
    %         end
    %         hold on;
    %         plot(sigma_fig,'k-');
    %         title('\sigma_{13}');
    %         xlabel('time (min)');
    %         axis([1 N -0.3 1]);
    %      end
    %     legend('\sigma^5_{13}','\sigma^1_{13}');
        
    %     
    %     figure(Iter*100000+i*10+j)
    %     plotfunc(q,i,j);
    %     title('q_{12}');
    %     axis([1 N 0 25]);
    %     
    %     figure(Iter*1000000+i*10+j)
    %     plotfunc(v,i,j);
    %     title('v_{12}');
    %     axis([1 N 0 6]);
    %     
    %   
    %     i=1;j=3;
    %     figure(Iter*1000+i*10+j)
    %     plotfunc(p,i,j);
    %     title('p_{13}');
    %     axis([1 N 0 6]);
    %     
    % %      figure(Iter*100+i)
    % %      plotfunc2(pi,i);
    % %      title('\pi_1');
    % %      axis([1 N 0 15]);
    %      
    % %     figure(Iter*10000+i*10+j)
    % %     plotfunc(sigma,i,j);
    % %     title('\sigma_{13}');
    % %     axis([1 N -0.3 1]);
    %     
    %     figure(Iter*100000+i*10+j)
    %     plotfunc(q,i,j);
    %     title('q_{13}');
    %     axis([1 N 0 25]);
    % 
    %     p1=zeros(1,size(p,3));
    %     for r=1:size(p,3)
    %         p1(r) = p(1,2,r)+p(1,3,r);
    %     end
    %     figure(9999)
    %     plot(p1,'k.-');
    %     title('demand d_1');
    %     axis([1 N 0 6]);
        clear r_*
        if strct_sigma(Iter).conv_err <= 1e-6
            NLoops = Iter;
            break
        end
    end
    %iteration end
    %%
    Conv_err = zeros(1, Iter);
    for Iter = 1:NLoops
        Conv_err(Iter) = strct_sigma(Iter).conv_err;
    end
    
    nEdges = sum(sum(links));
    p_save = zeros(nEdges-2, N);
    v_save = zeros(nEdges-2, N);
    qu_save = zeros(nEdges-2, N);
    qd_save = zeros(nEdges-2, N);
    mu_save = zeros(nEdges-2, N);
    delta_save = zeros(nEdges-2, N);
    
    k = 1;
    for i = 1:nNodes-2
        for j = 1:nNodes-2
            if links(i, j) == 1
                p_save(k, :) = p(i, j, :);
                v_save(k, :) = v_r(i, j, :);
                qu_save(k, :) = qu(i, j, :);
                qd_save(k, :) = qd(i, j, :);
                mu_save(k, :) = mu(i, j, :);
                delta_save(k, :) = delta(i, j, :);
                k = k + 1;
            end
        end
    end
    check = sum(sum(d));
    save data/dta.mat p_save v_save mu_save delta_save qu_save qd_save check

    %% plot: comment the following for now by Ohay
%     figure(3141599)
%     plot(Conv_err,'k.-','MarkerSize',8);
%     title('Convergence Error')
%     xlabel('Iteration Number');
%     
%     lgConv_err = log10(Conv_err);
%     figure(3141600);plot(lgConv_err,'k.-','MarkerSize',8);
%     xlabel('Iteration Number');
%     title('lg Convergence Error')

%%%%% below we generate testing data until SUMO has been set up
% % save the variables in the size of the number of edges
% % in the six-link network, we have 8 edges
% % edges: (1, 2)
% %        (1, 3)
% %        (2, 3)
% %        (3, 4)
% %        (3, 6)
% %        (4, 6)
% %        (5, 1)
% %        (6, 7)
% nEdges = sum(sum(links));
% edges_save = [1 2;
%               1 3;
%               2 3;
%               3 4;
%               3 6;
%               4 6;
%               5 1;
%               6 7];
% 
% noise = randn(size(p));
% noise(noise < 0) = 0;
% p = p + noise;
% noise = randn(size(v_r));
% noise(noise < 0) = 0;
% v_r = v_r + noise;
% noise = randn(size(mu));
% noise(noise < 0) = 0;
% mu = mu + noise;
% noise = randn(size(delta));
% noise(noise < 0) = 0;
% delta = delta + noise;
% noise = randn(size(qu));
% noise(noise < 0) = 0;
% qu = qu + noise;
% noise = randn(size(qd));
% noise(noise < 0) = 0;
% qd = qd + noise;
% 
% p_save = zeros(nEdges, N);
% v_save = zeros(nEdges, N);
% qu_save = zeros(nEdges, N);
% qd_save = zeros(nEdges, N);
% mu_save = zeros(nEdges, N);
% delta_save = zeros(nEdges, N);
% 
% k = 1;
% for i = 1:nNodes
%     for j = 1:nNodes
%         if links(i, j) == 1
%             p_save(k, :) = p(i, j, :);
%             v_save(k, :) = v_r(i, j, :);
%             qu_save(k, :) = qu(i, j, :);
%             qd_save(k, :) = qd(i, j, :);
%             mu_save(k, :) = mu(i, j, :);
%             delta_save(k, :) = delta(i, j, :);
%             k = k + 1;
%         end
%     end
% end
% save data/observed.mat p_save v_save mu_save delta_save qu_save qd_save
% 
% tao0(length(tao0) + 1, :) = [5 1 0];
% tao0(length(tao0) + 1, :) = [6 7 0];
% Cbar(length(Cbar) + 1, :) = [6 7 0];
% tau0_save = zeros(nEdges, 1);
% tauw_save = zeros(nEdges, 1);
% Cbar_save = zeros(nEdges, 1);
% Qbar_save = zeros(nEdges, 1);
% d_save = zeros(nEdges, N);
% 
% k = 1;
% for i = 1:nNodes
%     for j = 1:nNodes
%         if links(i, j) == 1
%             tau0_save(k, 1) = tao0(tao0(:, 1) == i & tao0(:, 2) == j, 3);
%             tauw_save(k, 1) = tau0_save(k, 1) * 2;
%             Cbar_save(k, 1) = Cbar(Cbar(:, 1) == i & Cbar(:, 2) == j, 3);
%             Qbar_save(k, 1) = Qbar(Qbar(:, 1) == i & Qbar(:, 2) == j, 3);
%             k = k + 1;
%         end
%     end
% end
% 
% priority_save = [1; 1; 2; 1; 1; 2; 1; 1];
% d_save(7, :) = d(5, :);
% save data/given.mat tau0_save tauw_save Cbar_save Qbar_save d priority_save edges_save
% %%%%%
end

function output = resize(input, nNodes, N)
    output = zeros(nNodes,nNodes,N);
    for i = 1:length(input.val)
        pairi = input.val(i, 1);
        pairj = input.val(i, 2);
        pairt = input.val(i, 3);
        output(pairi, pairj, pairt) = input.val(i, 4);
    end
end