function sigma = WriteSigma(p,pi,qd,tao0,C,time)
    sigma = zeros(size(p));
    for i=1:size(p,1)
        for j=1:size(p,2)
        if i==j
            continue
        end
        if C(C(:,1)==i & C(:,2)==j,3)~=0  
            for r=1:size(p,3)
                sigma(i,j,r) = 0;
            end
            %sigma(t)=[pi(t+tao(t)-tao0)-pi(t)]/[tao(t)-tao0]
            for r=(tao0(tao0(:,1)==i & tao0(:,2)==j,3)+1):size(p,3)
%                 t = smallest_s(r);
                 if r>40
                     ;
                 end
                t=time(r);
                tao2 = qd(qd(:,1)==i & qd(:,2)==j & qd(:,3)==r,4)/C(C(:,1)==i & C(:,2)==j,3);
%                  [t_p r_future_i] = inverse_t(t+tao2, time, zeros(size(time)));
%                  eta_future_i = pi(j,r_future_i);
%                  eta_future_i1 = pi(j,r_future_i-1);
%                  eta_future = eta_future_i + (eta_future_i1 - eta_future_i)*(t+tao2 -time(r_future_i))/(time(r_future_i-1)-time(r_future_i));
                if tao2~=0
                    tu = ceil(t+tao2);
                    if tu>size(p,3)
                        tu=size(p,3);
                    end
                    tl = tu-1;
                    pi_future = pi(j,tl)+(t+tao2-tl)*(pi(j, tu)-pi(j,tl));
                    pi_now = pi(j,r);
                     sigma_temp = (pi_future - pi_now)/(tao2);
                     sigma_temp = max(sigma_temp,-1);
                else
                    sigma_temp = 0;
                end
                 sigma(i,j,r) = sigma_temp;
            end
        end
        end
    end
    
    
    
    Sigma_id = fopen('gamsSigma.gms', 'w');
     %fprintf(Sigma_id, '::r_Sigma\n');
     %fprintf(Sigma_id, 'd %d %d %d %d\n', size(nodes,2), size(nodes,2), size(nodes,2), size(links(curlink).traveltime,2));
    fprintf(Sigma_id, 'parameter sigma(i,j,r)/\n');
    for i=1:size(p,1)
        for j=1:size(p,2)
            for r=1:size(p,3)
	            fprintf(Sigma_id, '%d.%d.%d %1.8f\n', i, j, r, sigma(i,j,r));
            end
        end
    end
    fprintf(Sigma_id, '/;\n');
    fclose(Sigma_id);
%save('links_result.mat','links','smallest_s','nodes');
end