*With queue capacity
*With inflow capacity constraint

*$offsymxref
*$offlisting
*option limrow = 400, limcol = 400;
option limrow = 0, limcol = 0;
*option solprint = off;

*$include gamsUnca_data_SF.gms
*$include gamsUnca_ToCa_data.gms
*$include gamsCapa_data2.gms


*$include gamsCapa_data0528.gms
$include gamsCapa_config_and_param.gms

*$include gamsCapa_data_LinkChain_04172016.gms


*$include gamsCapa_data0826_onepath.gms
*$include gamsCapa_data1217_onepath.gms
*$include gamsCapa_data0403_2ori.gms

*$include gams_data_Ca_SF.gms

*$include gamsNetSZ.gms
*$include gametime.gms
*$include gamsd.gms

*$include gamsSigma.gms
parameter sigma(i,j,r);
sigma(i,j,r)=0;


* variables
positive variables mu(i, j, r) 'slack var for exit flow';
*positive variables mubar(i, j, r) 'slack var for exit flow';
positive variables p(i, j, r) 'in-flow at h';
positive variables pi(i, r) 'shortest travel time';

positive variables deltah(i, j, r) 'nonnegative slack var 2 for exit flow';
variables deltabar(i, j, r) 'slack var 2 for exit flow';

positive variables etap(i,j,r) 'inflow p cap slack';
positive variables eta(i, j, r) 'upstream q slack';
positive variables etadesti(i, j, r) 'upstream q slack';
positive variables v(i, j, r) 'exit flow';

positive variables d(i, r) 'demand';
positive variables zeta0(i, r);
positive variables zetap(i, r);
positive variables lambdah(i);

* MODEL SECTION for NCP DUE
equations
eq_mu(i, j, r) 'mu_bar complementary',
eq_p(i, j, r) 'p complementary',
eq_pi(i, r) 'pi complementary, flow conservation',
eq_deltabar(i,j,r) 'definition of delta_bar',
eq_delta(i,j,r) 'definition of delta',
eq_etap(i,j,r) 'p cap complementary',
eq_eta(i, j, r) 'etaqu complementary, UpQ_Cap',
eq_eta_desti(i, j, r) 'eta complementary for dummy destination link, UpQ_Cap',
eq_v(i, j, r) 'exit flow v definition',
eq_d(i, r) 'd complementary',
eq_zeta0(i, r) 'zeta0 complementary',
eq_zetap(i ,r) 'zetap complementary',
eq_lambdah(i) 'lambda complementary, demand conserved';

eq_mu(i, j, r)$(ord(r)>=nh(i,j)+1 and links(i,j) and not dummydesti(j))..
(sum(s$(ord(s)>=nh(i,j)+1 and (ord(s)<=ord(r))),p(i,j,s-nh(i,j))-Cbar(i,j)+mu(i,j,s)+deltabar(i,j,s)))*hlength =g= 0;

eq_p(i, j, r-nh(i,j))$(ord(r)>=nh(i,j)+1 and links(i,j) and not dummydesti(j))..
*(1+sigma(i,j,r))*(tao0(i,j)+(hlength/Cbar(i,j))*sum(s$(ord(s)>=nh(i,j)+1 and ord(s)<=ord(r)),p(i,j,s-nh(i,j))-Cbar(i,j)+mu(i,j,s)+deltabar(i,j,s))) + pi(j, r)- pi(i, r-nh(i,j)) -sigma(i,j,r)*tao0(i,j) =g= 0;
(1+sigma(i,j,r))*(tao0(i,j)+(1+(mu(i,j,r)+deltabar(i,j,r))/Cbar(i,j))*(hlength/Cbar(i,j))*sum(s$(ord(s)>=nh(i,j)+1 and ord(s)<=ord(r)),p(i,j,s-nh(i,j))-Cbar(i,j)+mu(i,j,s)+deltabar(i,j,s))) + pi(j, r)- pi(i, r-nh(i,j)) -sigma(i,j,r)*tao0(i,j) =g= 0;

eq_pi(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) and not dummydesti(i))..
sum(j$(links(i,j) and ord(r)<=card(h)-1+1-nh(i,j)),p(i,j,r)) - sum(j$(links(j,i) and ord(r)>=nh(j,i)+1),Cbar(j,i)-mu(j,i,r)-deltabar(j,i,r)) - d(i,r) =g= 0;

eq_deltabar(i,j,r)$(links(i,j) and ord(r)>=nh(i,j)+1 and not dummydesti(j))..
sum(k$links(j,k), eta(j,k,r)+etap(j,k,r)) + sum(k$links(j,k), etadesti(j,k,r)) - sum(l$(ord(l)<ord(i)), deltah(l,j,r)) - deltabar(i,j,r) =g= 0;

eq_v(i,j,r)$(links(i,j) and ord(r)>=nh(i,j)+1 and not dummydesti(j))..
deltabar(i,j,r) =e= Cbar(i,j) - mu(i,j,r) - v(i,j,r);

eq_delta(i,j,r)$(links(i,j) and ord(r)>=nh(i,j)+1 and not dummydesti(j))..
deltah(i,j,r) - deltabar(i,j,r) =g= 0;
*deltah(i, j, r) =e= ((Cbar(i,j)-mu(i,j,r))/(epsilon+sum(l$links(l,j), Cbar(l,j)-mu(l,j,r))) * sum(k$links(j,k), eta(j,k,r)))$(sum(l$links(l,j),1)>1) + sum(k$links(j,k), epsilon4*eta(j,k,r))$(sum(l$links(l,j),1)<=1);
*deltah(i, j, r) =e= (Cbar(i,j)-mu(i,j,r))/(epsilon+sum(l$links(l,j), Cbar(l,j)-mu(l,j,r))) * sum(k$links(j,k), eta(j,k,r));

eq_etap(i, j, r)$(links(i,j) and (not dummyori(i)) and (ord(r)>=nbeforehead(i)+1) and not dummydesti(j))..
1*Cbar(i,j) - p(i,j,r) =g= 0;

eq_eta(i, j, r)$(links(i,j) and (not dummyori(i)) and (ord(r)>=nbeforehead(i)+1))..
Qbar(i,j) - hlength*sum(s$(ord(s)<nomegah(i,j)+1 and ord(s)<=ord(r)), p(i,j,s)) - hlength* sum(s$(ord(s)>=nomegah(i,j)+1 and ord(s)<=ord(r)), p(i,j,s)-Cbar(i,j)+mu(i,j,s-nomegah(i,j))+deltabar(i,j,s-nomegah(i,j))) =g= 0;

*Effectively eta=0 for dummy destination link, since Qbar->inf
eq_eta_desti(i, j, r)$(links(i,j) and dummydesti(j) and (ord(r)>=nbeforehead(i)+1))..
Qbar(i,j) - hlength*sum(s$(ord(s)<=ord(r)), sum(l$(links(l,i)),v(l,i,s)) ) =g= 0;

eq_d(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and dummyori(i))..
-alpha*(Delta+hlength*ord(r)-R_para(i))+(1-alpha)*pi(i,r)+alpha*zeta0(i,r)+(alpha+gamma)*zetap(i,r)-lambdaup(i)+lambdah(i) =g= 0;

eq_zeta0(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) and dummyori(i))..
-(Delta+hlength*ord(r)-R_para(i))-pi(i,r)+zeta0(i,r)+zetap(i,r) =g= 0;

eq_zetap(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) and dummyori(i))..
2*Delta - zeta0(i,r) =g= 0;

eq_lambdah(i)$(not desti(i) and dummyori(i))..
-D0(i)-hlength*sum(r$(ord(r)<=card(h)-1-nhead(i)+1),d(i,r))+Dbar(i) =g= 0;

$ontext
eq_p(i, j)$(links(i,j) and (not desti(i)) )..
*tt(i, j)-sum(n,LN(i,j,n)*eta(n))+epsilon(j)-epsilon(i) =g= 0;
tt(i, j)-sum(n,LN(i,j,n)*eta(n)) =g= 0;

eq_eta(i)$(not desti(i))..
sum(j,Connectivity(i,j)*p(i,j)) + d_b(i) =g= 0;

model NCP_DUE /eq_p.p,eq_eta.eta/;
$offtext
*Hard constraint on inflow rate; NOT working
*p.up(i,j,r)$(links(i,j) and ord(r)<card(r)+1-nh(i,j))=Cbar(i,j);

mu.fx(i,j,r)$(not links(i,j)) = 0;
p.fx(i,j,r)$(not links(i,j)) = 0;
pi.fx(i,r)$(desti(i)) = 0;
mu.fx(i,j,r)$(dummydesti(j)) = 0;
p.fx(i,j,r)$(dummydesti(j)) = 0;
pi.fx(i,r)$(dummydesti(i)) = 0;

mu.fx(i,j,r)$(links(i,j) and ord(r)<nh(i,j)+1) = Cbar(i,j);
p.fx(i,j,r)$(links(i,j) and ord(r)>card(r)-1+1-nh(i,j)) = 0;

pi.lo(i,r)$(not desti(i) and not dummydesti(i)) = pi0(i)*(1-epsilon2);
*pi.lo(i,r)$(not desti(i) and not dummydesti(i)) = 0;
*pi.lo(i,r)$(not desti(i)) = 0;
pi.fx(i,r)$(ord(r)>card(r)-1+1-nhead(i)) = pi0(i);
*pi.fx(i,r)$(ord(r)>card(r)-1+1-nhead(i)) = 0;


deltah.fx(i,j,r)$(not links(i,j)) = 0;
*deltah.fx(i,j,r)$(desti(j))=0;
deltah.fx(i,j,r)$(dummydesti(j))=0;
deltah.fx(i,j,r)$(ord(r)<nh(i,j)+1)=0;

deltabar.fx(i,j,r)$(not links(i,j)) = 0;
*deltabar.fx(i,j,r)$(desti(j))=0;
deltabar.fx(i,j,r)$(dummydesti(j))=0;
deltabar.fx(i,j,r)$(ord(r)<nh(i,j)+1)=0;

v.fx(i,j,r)$(not links(i,j)) = 0;
v.fx(i,j,r)$(ord(r)<nh(i,j)+1)=0;
v.fx(i,j,r)$(dummydesti(j))=0;

etap.fx(i, j, r)$(not links(i,j))= 0;
etap.fx(i,j,r)$(dummyori(i)) = 0;
etap.fx(i,j,r)$(ord(r)<nbeforehead(i)+1)=0;
etap.fx(i,j,r)$(dummydesti(j)) = 0;

eta.fx(i,j,r)$(not links(i,j)) = 0;
eta.fx(i,j,r)$(dummyori(i)) = 0;
eta.fx(i,j,r)$(dummydesti(j)) = 0;
eta.fx(i,j,r)$(ord(r)<nbeforehead(i)+1)=0;
etadesti.fx(i,j,r)$(not links(i,j)) = 0;
etadesti.fx(i,j,r)$(not dummydesti(j)) = 0;
etadesti.fx(i,j,r)$(ord(r)<nbeforehead(i)+1)=0;
d.fx(i,r)$(not dummyori(i)) = 0;
zeta0.fx(i,r)$(desti(i)) = 0;
zetap.fx(i,r)$(desti(i)) = 0;

d.fx(i,r)$(ord(r)>card(r)-1+1-nhead(i)) = 0;
zeta0.fx(i,r)$(ord(r)>card(r)-1+1-nhead(i)) = 0;
zetap.fx(i,r)$(ord(r)>card(r)-1+1-nhead(i)) = 0;
zeta0.fx(i,r)$(not dummyori(i)) = 0;
zetap.fx(i,r)$(not dummyori(i)) = 0;
lambdah.fx(i)$(not dummyori(i)) = lambdaup(i);

*p.fx('5','1',r)$(ord(r)>=26) = 0;
*p.fx('1',j,r)$(ord(r)>=26) = 0;

model NCP_DUE /
eq_mu.mu
eq_p.p
eq_pi.pi
eq_eta.eta
eq_etap.etap
eq_eta_desti.etadesti
eq_delta.deltah
eq_deltabar.v
eq_v.deltabar
eq_d.d
eq_zeta0.zeta0
eq_zetap.zetap
eq_lambdah.lambdah
/;

NCP_DUE.iterlim = 1000000;
NCP_DUE.reslim = 36000;
option mcp = path;
NCP_DUE.optfile = 1;
solve NCP_DUE using mcp;
*display links;
*parameter vl(i,j,r);
*vl(i,j,r)$(links(i,j)) = Cbar(i,j)-mu.l(i,j,r)-deltah.l(i,j,r);
*vl(i,j,r)$(links(i,j) and ord(r)<=nh(i,j)) = 0;
*vl(i,j,r)$(links(i,j) and vl(i,j,r) >-1.0e-3 and vl(i,j,r)< 1.0e-3) = 0;
*parameter xl(i,j,r);
*xl(i,j,r)$(links(i,j)) = sum(s$(ord(s)<=ord(r) and ord(s)>= ord(r)-nh(i,j)+1),p.l(i,j,s));
*xl(i,j,r)$(links(i,j) and xl(i,j,r) < 1.0e-3 and xl(i,j,r) > -1.0e-3) = 0;
*display mu.l,vl;

*v.l(i,j,r)$(desti(j)) = Cbar(i,j)-mu.l(i,j,r);
*display d.l,p.l,v.l,mu.l;
parameter qd(i,j,r);
qd(i,j,r)$(links(i,j)) = hlength*sum(s$(ord(s)>=nh(i,j)+1 and ord(s)<=ord(r)), p.l(i,j,s-nh(i,j))-Cbar(i,j)+mu.l(i,j,s)+deltabar.l(i,j,s));
qd(i,j,r)$(links(i,j) and qd(i,j,r) >-1.0e-3 and qd(i,j,r) < 1.0e-3) = 0;
parameter qu(i,j,r);
qu(i,j,r)$(links(i,j)) = hlength*sum(s$(ord(s)<nomegah(i,j)+1 and ord(s)<=ord(r)), p.l(i,j,s)) + hlength* sum(s$(ord(s)>=nomegah(i,j)+1 and ord(s)<=ord(r)), p.l(i,j,s)-Cbar(i,j)+mu.l(i,j,s-nomegah(i,j))+deltabar.l(i,j,s-nomegah(i,j)));
qu(i,j,r)$(links(i,j) and qu(i,j,r) >-1.0e-3 and qu(i,j,r) < 1.0e-3) = 0;
parameter Qbar2(i,j,r);
Qbar2(i,j,r)$(ord(r)=1) = Qbar(i,j);


*display pi.l;
parameter paraPiEq(i, r);
paraPiEq(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) ) =
sum(j$(links(i,j) and ord(r)<=card(h)-1+1-nh(i,j)),p.l(i,j,r)) - sum(j$(links(j,i) and ord(r)>=nh(j,i)+1),Cbar(j,i)-mu.l(j,i,r)-deltabar.l(j,i,r)) - d.l(i,r) ;
paraPiEq(i,r)$(paraPiEq(i,r) < 1.0e-6 and paraPiEq(i,r) >-1.0e-6) = 0;
*display paraPiEq;

*display lambdah.l;
$ontext
*p.l('1','2',r)$(paraPiEq('1', r)>0) = p.l('1','2',r)-paraPiEq('1', r);
p.l('1','3',r)$(paraPiEq('1', r)>1e-6) = d('1', r);
vl('1','3',r+nh('1','3'))$(paraPiEq('1', r)>0$(ord(r)>nh('1','3'))) = p.l('1','3',r);
p.l('3','4',r)$(paraPiEq('3', r)>1e-6) = vl('1','3',r)+d('3', r);
vl('3','4',r+nh('3','4'))$(paraPiEq('3', r)>0$(ord(r)>nh('3','4'))) = p.l('3','4',r);
p.l('4','11',r)$(paraPiEq('4', r)>1e-6) = vl('3','4',r)+d('4', r);
vl('4','11',r+nh('4','11'))$(paraPiEq('4', r)>0$(ord(r)>nh('4','11'))) = p.l('4','11',r);
p.l('11','14',r)$(paraPiEq('11', r)>1e-6) = vl('4','11',r)+d('11', r);
vl('11','14',r+nh('11','14'))$(paraPiEq('11', r)>0$(ord(r)>nh('11','14'))) = p.l('11','14',r);
p.l('14','15',r)$(paraPiEq('14', r)>1e-6) = vl('11','14',r)+d('14', r);
vl('14','15',r+nh('14','15'))$(paraPiEq('14', r)>0$(ord(r)>nh('14','15'))) = p.l('14','15',r);

p.l('16','17',r)$(paraPiEq('16', r)>1e-6) = d('16', r);
vl('16','17',r+nh('16','17'))$(paraPiEq('16', r)>0$(ord(r)>nh('16','17'))) = p.l('16','17',r);
p.l('17','19',r)$(paraPiEq('17', r)>1e-6) = vl('16','17',r)+d('17', r);
vl('17','19',r+nh('17','19'))$(paraPiEq('17', r)>0$(ord(r)>nh('17','19'))) = p.l('17','19',r);
p.l('19','15',r)$(paraPiEq('19', r)>1e-6) = vl('17','19',r)+d('19', r);
vl('19','15',r+nh('19','15'))$(paraPiEq('19', r)>0$(ord(r)>nh('19','15'))) = p.l('19','15',r);
$offtext

*display p.l;
*display vl;
*paraPiEq(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) ) =
*sum(j$(links(i,j) and ord(r)<=card(h)-1+1-nh(i,j)),p.l(i,j,r)) - sum(j$(links(j,i) and ord(r)>=nh(j,i)+1),vl(j,i,r)) - d.l(i,r) ;
*paraPiEq(i,r)$(paraPiEq(i,r) < 1.0e-6 and paraPiEq(i,r) >-1.0e-6) = 0;
*display paraPiEq;

parameter paraF(i, r);
paraF(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) and Dbar(i) > 0)=
-alpha*(Delta+hlength*ord(r)-R_para(i)+pi.l(i,r))+alpha*zeta0.l(i,r)+(alpha+gamma)*zetap.l(i,r);
paraF(i, r)$(ord(r)>card(h)-1-nhead(i)+1 and (not desti(i)) and Dbar(i) > 0)=
gamma*(ord(r)+pi.l(i,r)-R_para(i)-Delta);
*display paraF;

parameter paraFplusPi(i, r);
*paraFplusPi(i, r)$(ord(r)<=card(h)-1-nhead(i)+1 and (not desti(i)) and Dbar(i) > 0)=
paraFplusPi(i, r)$( (not desti(i)) and Dbar(i) > 0)=
paraF(i,r)+pi.l(i,r);
*display paraFplusPi;

*display qd,qu,Qbar2;
*display etap.l,eta.l,etadesti.l,deltah.l,deltabar.l;
parameter para_eta_delta(i, r);
para_eta_delta(i, r)$(not desti(i))=
sum(k$(links(i,k)),eta.l(i,k,r)+etap.l(i,k,r)) - sum(l$(links(l,i)),deltah.l(l,i,r));
*display para_eta_delta;

parameter Number_h;
Number_h = card(h);

parameter para_eq_p(i, j, r);
para_eq_p(i,j,r-nh(i,j))$(ord(r)>=nh(i,j)+1 and links(i,j) and not dummydesti(j))=
(1+sigma(i,j,r))*(tao0(i,j)+(1+(mu.l(i,j,r)+deltabar.l(i,j,r))/Cbar(i,j))*(hlength/Cbar(i,j))*sum(s$(ord(s)>=nh(i,j)+1 and ord(s)<=ord(r)),p.l(i,j,s-nh(i,j))-Cbar(i,j)+mu.l(i,j,s)+deltabar.l(i,j,s))) + pi.l(j, r)- pi.l(i, r-nh(i,j)) -sigma(i,j,r)*tao0(i,j);
*display para_eq_p;

parameter para_eq_p_p(i,j,r);
para_eq_p_p(i,j,r)= p.l(i,j,r)*para_eq_p(i,j,r);
*display para_eq_p_p;

$gdxout DQDUE_Priority_CapIn
$unload
$gdxout
