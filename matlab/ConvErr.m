function Conv_err = ConvErr(strct_sigma, Iter)
%convergence error 
%the difference of the PDs between two consecutive iterations
%Iter: iteration number
%    nNode = size(strct_sigma(Iter).sigma,1);
%    N = size(strct_sigma(Iter).sigma,3); %# of time steps
    if Iter ==1
        Conv_err = 1;
    else
        den = sum(sum(sum(abs(strct_sigma(Iter).sigma))));
        num = sum(sum(sum(abs(strct_sigma(Iter).sigma - strct_sigma(Iter-1).sigma))));
%        den = sum(sum(sum((strct_sigma(Iter).sigma).*(strct_sigma(Iter).sigma))));
%        num = sum(sum(sum((strct_sigma(Iter).sigma - strct_sigma(Iter-1).sigma).*(strct_sigma(Iter).sigma - strct_sigma(Iter-1).sigma))));
        if den ~=0
            Conv_err = num/den;
        elseif den ==0 && num ==0
            Conv_err = 0;
        end
    end
end

