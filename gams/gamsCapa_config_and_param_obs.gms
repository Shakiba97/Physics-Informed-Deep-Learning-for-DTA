
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
1.2 3.0
1.3 6.0
2.3 1.0
3.4 1.0
3.6 4.0
4.6 1.0
5.1 0.0
6.7 0.0
/;

parameter nh(i,j);
nh(i,j) = tao0(i,j);

parameter nomegah(i,j)/
1.2 6.0
1.3 10.0
2.3 3.0
3.4 3.0
3.6 6.0
4.6 2.0
5.1 0.0
6.7 0.0
/;

parameter Cbar(i,j)/
1.2 3.0
1.3 6.0
2.3 2.0
3.4 3.0
3.6 1.0
4.6 3.0
5.1 10.0
6.7 0.0
/;

parameter Qbar(i,j)/
1.2 30.0
1.3 75.0
2.3 6.0
3.4 5.0
3.6 7.0
4.6 4.0
5.1 40.0
6.7 80.0
/;


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

