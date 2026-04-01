*gamsCapa_data2.gms
set nodes /1*7/;
alias(nodes, n, i, j, k, l, kk);
set desti(nodes) /6/;
set dummyori(nodes) /5/;
set dummydesti(nodes) /7/;

parameter hlength /1/;
set h /1*50/;
alias(h, hh, s, r);
set links (i, j) /
1.2
2.3
1.3
3.4
5.1
3.6
6.7
4.6
/;

parameter tao0(i,j)/
1.2 3
2.3 1
1.3 6
3.4 1
5.1 0
3.6 2
6.7 0
4.6 1
/;
parameter nh(i,j);
nh(i,j) = tao0(i,j);
$ontext
/
1.2 3
2.3 1
1.3 6
3.4 1
5.1 0
3.6 2
6.7 0
4.6 1
/;
$offtext
parameter nomegah(i,j);
nomegah(i,j) = 2*nh(i,j);
$ontext
/
1.2 6
2.3 2
1.3 12
3.4 2
5.1 0
3.6 4
6.7 0
4.6 2
/;
$offtext
parameter Cbar(i,j)/
1.2 3
2.3 2
1.3 4
3.4 2
5.1 14
3.6 1
6.7 0
4.6 2
/;
parameter Qbar(i,j);
Qbar(i,j)$(not dummydesti(j) and not dummyori(i) and links(i,j)) = Cbar(i,j)*(nh(i,j)+nOmegah(i,j))*hlength;
Qbar(i,j)$(dummyori(i) and links(i,j)) = 40;
Qbar(i,j)$(dummydesti(j) and links(i,j)) = 80;

parameter nhead(i) /
1 3
2 1
3 1
4 1
5 0
6 0
7 0
/;
parameter nbeforehead(i)/
1 0
2 3
3 1
4 1
5 0
6 1
7 0
/;

parameter pi0(i) /
1 6
2 3
3 2
4 1
5 6
6 0
7 0
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
5 40
6 0
7 0
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
parameter gamma /1.5/;
parameter Delta /1/;
parameter R_para(i) /
1 15
2 15
3 15
4 15
5 25
6 15
7 15
/;
*parameter epsilon /1e-3/;
parameter epsilon2 /1e-3/;
*parameter epsilon2 /0/;
*parameter epsilon4 /1/;


$ontext
parameter d(i,r)/
5.1 0
5.2 2
5.3 4
5.4 6
5.5 7
5.6 7.5
5.7 7
5.8 6
5.9 4
5.10 2
/;
$offtext
