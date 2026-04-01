set nodes /1*7/;
alias(nodes, n, i, j, k, l, kk);
set desti(nodes) /5/;
set dummyori(nodes) /7/;
set dummydesti(nodes) /6/;

parameter hlength /1/;
set h /1*60/;
alias(h, hh, s, r);
set links (i, j) /
1.2
1.3
2.3
3.4
3.5
4.5
7.1
5.6
/;

parameter tao0(i,j)/
1.2 3.0
1.3 4.0
2.3 1.0
3.4 1.0
3.5 2.0
4.5 1.0
7.1 0.0
5.6 0.0
/;

parameter nh(i,j);
nh(i,j) = tao0(i,j);

parameter nomegah(i,j)/
1.2 8.0
1.3 10.0
2.3 4.0
3.4 4.0
3.5 6.0
4.5 4.0
7.1 0.0
5.6 0.0
/;

parameter Cbar(i,j)/
1.2 21.0
1.3 28.0
2.3 9.0
3.4 9.0
3.5 5.0
4.5 9.0
7.1 441.0
5.6 0.0
/;

parameter Qbar(i,j);
            Qbar(i,j)$(not dummydesti(j) and not dummyori(i) and links(i,j)) = Cbar(i,j)*(nh(i,j)+nOmegah(i,j))*hlength;
            Qbar(i,j)$(dummyori(i) and links(i,j)) = 5000;
            Qbar(i,j)$(dummydesti(j) and links(i,j)) = 5000;
parameter nhead(i) /
1 3.0
2 1.0
3 1.0
4 1.0
5 0.0
7 0.0
6 0.0
/;

parameter nbeforehead(i)/
1 0.0
2 3.0
3 1.0
4 1.0
5 1.0
7 0.0
6 0.0
/;

parameter pi0(i)/
1 6.0
2 3.0
3 2.0
4 1.0
5 0.0
7 6.0
6 0.0
/;


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
7 440.0
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



