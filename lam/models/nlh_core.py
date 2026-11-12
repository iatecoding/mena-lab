"""nlh_core.py: Pre-tabulated Nordlund-Lehtola-Hobler (NLH) short-range nuclear repulsion & smooth distance floor.

Implements Suggestion 1 from SO3LR Tauri (Kabylda et al., arXiv:2611.06237v1, Sec. S1.2, Eqs. S4-S5) using the
pairwise 3-exponential screened nuclear Coulomb tables of Nordlund, Lehtola & Hobler (Phys. Rev. A 111, 032818, 2025)
for all element pairs Z = 1..92 (with smooth Z^0.23 Thomas-Fermi screening extension for Z = 93..127).

Supports three Short-Range Repulsion + Descriptor Floor regimes in MenaACE internal units (Angstrom, kcal/mol):
  1. Native SO3LR Tauri mode (energy `mode="so3lr"`, clamp `mode="tanh"`):
     - Energy (Eq. S4): 0.5 * w_sw(r; 0, r_off=1.5 A) * [f_cut(r; r_cut)] * E_NLH(r)
     - Descriptor floor (Eq. S5): smooth C2-handover tanh floor below d_off = r_min + w (w = 0.05 A), E_NLH(r_min) = 5 eV.
  2. Mena C3-gated mode (energy `mode="c3"`, clamp `mode="c3"` or `"tanh_c3_matched"`):
     - Switches on NLH smoothly below r_out = scale * (r_cov,i + r_cov,j) (e.g. scale = 0.45).
  3. Zero-retrain r_min-matched C2-shifted NLH mode (energy `mode="c2_shift_rmin"`, clamp `mode="tanh"`):
     - Subtracts the 2nd-order Taylor jet of E_NLH(r) at d_off = r_min(5 eV) + w (identical 0 for r >= d_off),
       giving 0.000 meV/A zero-retrain equilibrium perturbation, Z-adaptive heavy-element protection (up to 1.25 A),
       strict convexity (zero dead-zone and zero switch-derivative bump), and exact nuclear Coulomb asymptote as r -> 0.

Zero learnable parameters and zero state_dict keys (all lookup tables are module-level non-persistent constants).
"""

from __future__ import annotations

import base64
import zlib

import numpy as np
import torch
from torch import Tensor

COULOMB_K = 332.06371  # kcal * A / (mol * e^2), matching MenaACE.coulomb_k and zbl_core.COULOMB_K
EV_TO_KCAL = 23.060548  # kcal/mol per eV, matching lam.units.EV
KE_EV_A = COULOMB_K / EV_TO_KCAL  # 14.399645229592983 eV * A / e^2 (SO3LR uses 14.399645351950548)

DEFAULT_R_OUT_SCALE = 0.6
LAM3B_R_OUT_SCALE = 0.45
DEFAULT_R_IN_RATIO = 0.9
MAX_HEAVY_ZBL_COV_RAD_A = 0.75  # matches zbl_core.MAX_HEAVY_ZBL_COV_RAD_A for Z >= 19

DEFAULT_NLH_R_OFF_A = 1.5
DEFAULT_FLOOR_ENERGY_EV = 5.0
DEFAULT_FLOOR_WIDTH_A = 0.05
MAX_Z_NLH_TABLE = 92

# Exact upper-triangle (4278 pairs for Z = 1..92) NLH [a1, a2, a3, b1, b2, b3] * 1e5 as little-endian int32,
# zlib-compressed and base85-encoded (0.0 float64 reconstruction error vs Nordlund-Lehtola-Hobler PRA 2025 / so3lr).
_NLH_TRIU_I32_ZLIB_B85 = (
    "c-"
    "mxKby!qiv|Yo@oq+*{Vd#>QG*J=7BJJ)DFu(v4y9KNtHg+p^V2fSYjomGFcf2*8^WHzt@ADJZ9M3&>pS{;w`^I#N`|!6~j2+B>^s@~3Gu8Ir&({4^nC0"
    ";AM~rdO;eY=;UJdNMBYvK-)fQ_@tN;D{qb873MfLCJjdk4dx0mMM&k^20ayNkw-"
    "Bt#-Qj~u`4>AJ5X#$Tc-4Hj5YXAMbh10zmBk;zmu`0Va0-wvjx(tE;z5iJmI9{x2iNwr*ZiZX_-_QM=fD(PgXJIRA;9WT}{vc2Fa<U2ro@S{VZQ=b<hv"
    "FvSyuAt+`-"
    "}furj0e0_=xlBWdStJ2k!3!y625nZ98s&!A^ro;{m5XVtmi;PFPqd&UZp@z{Ob{cSlKgnCgo2E!&cFl@xecA3sDM5_oQ1;B7^LA72%O`37;mmwN$cT?K"
    "y3tq9J9i}4GF02&*C$BuDF7mdKb_W)M83%uZl7k;-9_}b>cmr??+xh4R$9|-(@ea0d};D@dSV?`-"
    "(+~M3u|EdBHc~K0f2MN4h9MB|2;A5VYK&jmV|4^0tT|@4HZy4HIh&byianI@q{N=&2s8n6xZWVyc`U2N~4M+8H0{>ExYppAAP3N*OtP*%yIiP+u;Fd{+"
    "M?1=+nWr_5w$dRf5b(5C!Lu3ACZRfZdgx&}UV~F^JRjz#(-h{Cdq?aTZUd`81s$0W<IXX6mjTaPr^2Pc7F8N4D8_tn-"
    "43)n%=q7#iP%a;{M@7<ZWYk>o;a;tTO2V~BhL5UY|j5i;0vN%5LrRsrEQpJSAnW47}sM*pz3GD+ie=WV4nHj1ZFJ)w!O^3`npCO7^=a!Gr*<YVxEq79F"
    "THJ<WG;yK=@UGUux@uIgP}A1>~@P?h8EYw=0k#@RKZzYp(>}zKsX^9~HPq7Lfi;;EimGKrPo_HiLNx5oeR95AH<>eAh6>#Y*74zxrWdUxE9j0V9k8f8Q"
    "VgOP2^dasbfPUf^v{1YqMPftMT#^tBea`k6l#%@Me6BKMndnBoUCi1J3(HZ4{**5Jutu6qpQ&(F`g`Js=E6<Rnc_|XP%?hZ^n$j@E|p%}TQ>`(=k<fK<"
    "~fc2bfd&3?Wv)vi4TNE5*{2$CGH*oL!#4UzRpeqvgD0oOps7O5hA@?ow4c_c@!_$)rk|KeJxj?gdz?!`kkT}zTD?>Ec9tJGh!klBCls-"
    "}oofg^Q)eH?Ddoc$$@ECbqmv<ZZ-!UQ8mh+Rxu51Jf$q|}w6S3FV8QW9}4wMD9u4g>BZ$B4AqhvWdcsVFo9}b-3_(0BI|8*3qpD-ZOSHactfIW}9dm;H"
    "CKMLRa7~oo5!Se{_E%UQ1_j%pyD0l=J@Sa@Y$a)FoS~|@DidT+8)62H#AE}@>&uilW*63vB($Z>JILsCn#NPw~OF57I8sJ4@Qv{VYV!<{A_gGtln2R;Z"
    "EBV2dk-5kQn~N&Qj3q~0V(iFggLhWM-%>W%V`3e}1IFu&O)`%$wgQ&_w8mMTg7!5So4Z_3N1%Mi2-"
    "qL6#>F=p>>zIQoOMGUx?L**YZh4J?;Z`>aGncafdRFEl^GQfv&<TcTr^nA{CIGZ^~-"
    "(fFd`AdGmU7pRl!=;b!DDc9`|5%<<Y3N)&#{)i!Cm|;=PO|W0w*;32sds;2Nk!Eb}C4C*Z};&eq?EB8ME&rjZs0Snma_F<b7}i}Tm9y0J5+jL_m6>okJ"
    "<)tj*>SM(bmrn+GDVlAvT0kf}(I<iUfz|GSlpD!|Z&KC&$TWJq$-79d08>=|~@Dq%`my>Gmb{8DU&|(1BpHHmI3Nu%X#rlR0Xmy_Z<-"
    "`0VFJ+Qfo}5U+dpi@-"
    "#&cgh0qtd;Z#LJJkp}G$6AtuHpyT=Gb6*M;0R8i`QDKGyny@ZPIkFyz?_Es&YQ7o1l^yYDIB|R6=|y1Da>jY^O*GUyVcj?dStY1dPx!YVU^lK2;=kBoR"
    "a*_#m*U)y$yt3__d)g1;DsH&SJ2?DJMiuqIcY5G_)!K1r<u@YtOjSOojo`{nSA?n|0!H|aYXO!8tf|s9xV{@iHvu_@t1<%G8z9V$AF29U$3X{@XEsldy"
    "gr|H&6??zahJTQCgkq-VYZn`K;jkI>z&+)Y&$+Sa8Eacn0QE+tPw(@?N>(ikp}}rV9C_mcRq9_+b1-"
    "fnQ}_ol2BCHLnaTCJ3H<$oTB<FL0gG6hoSdecZ=?9*q?E$n3jV?jrQ)C;zTY7x*9gkOSuhUSLH(K|iyQ|F7|hX~=pk@LM4&9BV5MJ7KdPc00xT$G&B5&"
    "K7vtU|U?u5`JSab2jL_zz5mbqic{@f3aDtmG=TK@!b&<mJ0myK=Oh?tS7ynJDl9ad3Ef<{Pz|3-"
    "tyiUGEm^$YIueV{IFvXzC9QCLDu<wCxN>U^1$@20<T$zd_|t07{dBH7Jvq}I(U6np=1Nbn&ZP5*Bf2|$iJ<{!%HemWxf0BRCrC^x)U6XdP8+coofMq*8"
    "3$zg?01=)wYMA%2FLHZPeJn{aQ~SRkJpA!LB6A`&#1W0^;P!rp(X9^fgmzl*Ng^mWXeu!IIKkAM5ey2=Z%u1X31SA?>sVgNgyKcQDsjU#+H8M$_4P4DO"
    "{s6-"
    ">S$=Y8D<EN>Qt?+vX{<gkKyzVxq*f4{S4y{d(U2W;T)sYNaF{37P<F7m+aqmA&g3HPR)7R6b2>#0R2$lWXUbcS21E#~DY7)YLJ_L_bz9k4hz7&f16pgE"
    ";M=p%ZU<zm17J7I^r9wMG@tm(|H0>8)Nbf_<IXX;eU9-zh%U}iBF)erK(Jn}$Ia#Bl~|8+8)aQucC-<*D>GRN9-%i6DUK-V4u524Rkb%@+Ve-ymg8S^Z"
    "4NE>Q_HO!x7^svpSOZ8n{QMbDeC$3xI;dEd+V^f?uFowS39rtB)pc;jl^hNZzi>d#8v)pkm&<az_XmEHM{RX*THwip<cl6@E)L*E<R%`ywnoc8k*Uw#y"
    "u_+{^1`5Jh2IZ)Mm2U9Z?uyFup);P(SMY~Ax9qt%uYD`s@$HFN{}F2Yo!0_CztsZ=Ek!-{ArF>%C-"
    "CQ?p7`J`@CKZ}4A0H`1+{AVOPn0)hM^8xEMYz;(^q6uACtEFp=eP(s?%3=NnoB+L)!4XG?xS6U0sh!12kyHe72+Kt1^}QKB+j;YTMx3E)6=y00+2_qn7"
    "~VGs3WhLeb{21{<lT4lB8b^l-Y?G050w57!_CW0)tqI8V3Z)Vq=GP;rVQe4Eful2=vio!U@O2G~`>h}}lS-"
    "`8L`*YcgZ5JDdj`N9ou_Z2+)ph8pnt6=V9KtryxoIi#=vc$_-"
    "YB)0X1G$e*)LG+*Kzul41;aZv^c<fwnla+~1MP~U@U$Kwi5m204F0l?N)(|s#rmNAG9CS_1vXRflNsks;{6=`(CDoWL-txgPd<71ft*79@zMEXfTJZg<"
    "XYe;d!s$nYhUWX{Ct0CLoJcM&;moL6H7>n>9nM0GK-"
    ">iyd@e<vp_$_e;Vt00ppQaA^?Snme@Jj0;~IwPswA0{n_Vj^hcE)mbltj4SSC7sZzn8KD{l+=S{H0rhXPs(*yczR2ahfpZEvi=N(Jz%dx;$@>^B%Ne>V"
    "DuNlE`^0&g>^A>2*0hq(_JIG7*|Ab=3Tq~6HRU@c1U}>R3-%#L}sRS}~dYBT_cu8(<^M#zjx`^5xiggY2P-"
    "m;Lg7t8b>nz6p!u>=gtSM;&B~60_`l~MV;Wx-jzi(DUwT8Bs{#yeJ?ynDdY4|Aee62c&xnn?fYXx-"
    "!sB_f!XRI^F1+6f1j1l|RDKIj=`{_Y0GPl;a^oCP=6Lx%3kjC@zKFa=^Ika$GA6OkW;kcrOp7?&&{B`Egy@!2~t9L*)dqo}nLnn@}KtHMEbw*x{38!Bv"
    "SWbU9kGh{oKW?fT1+R4mSQS&CCZE5if7{GDh#XrMz3pxBd%6a+e(<7yYtxhd#HT7|lrtcl+G#;veaZ9P!hET=sUco|x5L8%4PKK!V~F<|#~fYM2}`2wG"
    "2@E{Zp@DYdhl)RKf3FN;M@um8d8&TxZgt_u#T9c_tv+=#G3XPy<39?%+1lP?J;`#^U2PbKU0D7QH9!^qa?W?nEI@q=7JJ;Si8+Fu&W-B%D4??-"
    "hKHPhGCNpD5%Chs|@>7`muzOfT3j^uB|j;FY7vi>ug2tc|yKAHMav2=o^c#)1WVTd_HS$2J@lI1}|J)pu@p8YSbc!)#3a$^eHE{dQpRQX#L&-"
    "$<(iiXrYhiFMDF%457ch`Tx}oRVcy!qiX9AMCA$|tIB;zZKuKq@_N&+WiYFv;43pf_#~_Fhr=Fsjz;_A0(awip6;W<KH7lRBbuT`3*jH)%Bj${vEaQcn"
    "O1moLimK5Tw}|F(ti%H#mJjtJ>hHV$8HO}WO;jhmHwv}IkK$u4-GVq7<EnH>C_7k$-~7bI-rmA6<x{mk)H+bK;8LuMd0fg-_KeRhYI^$krX7(w;1zgfV"
    "}?jsh1a8jTiXB_CT!w74m3%YW?!TkJkbpLEk@yzGX>C>idyE{H&&f(`E})WF78&N-"
    "xOymiH<Rw?0+~=%~gu=3xkXxFv&F*DK4TptK$t8EPD)&aLJ7mZvYhw4*ZWozP<kW7LuH??CVWfZCt6Ee_ohY%n2RgVV$#>CYFi7dp8n9(C8)peg$ldoR"
    "&*459a0Z&e5P)dp<oO<zhry3ZW{@sd1$q7`bcGNQ&a1*z11dwLKb#wgQ}h)zrG(Wa#KU%rkT&KQs@SBIv;vbqVq+i77tfw}fu_7NkT5HeBh`%lJs<tKS"
    "<?S>=5h6&u6K6v6ef&bN;u;iJDTLWkM&U^Hj<oR>syQ0GkTO9bI!E*ZWGK|A%dYT#22O+740qN|^4$xOeUtrwGC;hx<;rTv0jIO94fqMOm`VvF^3P{|K"
    "(yi^`HAq28`p-%{Zh3BHrG<{TdqNy{9^>AgF<ae|{Zw%eZ24h{9GwO?x-o92$PZ2F7v>hhy=hixIG=o*!r0TF%pu;VOcAW^V1?&X)W}Mr-qGi~F)xm*e"
    "DL?673MSPYm;YZ(<kho2yAjGia(#M5zGD}j`eONch=?lr`!*~(@nNm`AdUh<lz<nkw5nXci$GnnOa61Zl_>7dE^~+`R5ht#=zAm7T}Br+ZC*19VW9kno"
    "N!>IpPdfXS-l3Jx3dsS?GM$0DG7!3*X`4R#&L$IYtwI$XHfk&$HmsGo+Su!#Q6q%2W4$O%<GO_qj4I-"
    "!tLtNd@(px3kE>`^>#xZUpWfv_lyEPYwEE3t}zUALTqQiIq=m(1U*UQ&sLCx#lkG(%@AVF^6qobwh(!<yeQr$J1}b*;YiJogI37)?g3wydCEmPycYYbU"
    "due+T)O?f)x7l0CJfv@qNKDc<*nIQMU9D{JjTb)0n!sGsz7-9kp<z-"
    "`+{R>B4<aCogWV>4E44TC6^3fsyP34^b~VMsv@4_#^kWB}}c=NTAQyPmer```+_aFgnslEqkcOex6Sma>pj(cT>D^_lXXXKh+56#lDojLRXxgcxM1M?6"
    "yQ*KQ%6NW#7nJ$tPc`tb&pH))K9XtMQuoQ;W6IopD{YI0!4ASz^Aw8Xpts6TWe;IKIN?U~CPsLe<)8oT0wwe5apb|JreAD4wKR!LPR()!G2ZesHhYA9R"
    "f=jv@!F5JeqNZU=m%Z;Y)#y;g;y_A4u#`@!)|sJrwhyBP0K$I|E?uSW&)h7bFrtiQkq_8)tThoM-S9!WRUI9{LYBOb<iy0s{S9l3gJrCv0qp3VgMjrn-"
    "6X*u-&rpHy#PZD3j@vGSHWJi|6Pe*I)wa~zs^G6bISe}0PXepF@t;dC(YQ(Z%y2*8pWnDaI6od44HV8kgK`PHDojSXo{w^xE7IwBXz^1Z-"
    "!R%{%uL4?n^!NLlAa}nV4o*{Ws3>`lxjBlyx6S-"
    "kIK9z`%lY)B<h}0n=G)lYjA@;KW6SMv?~Q_I<h_B^k{IUh&W%YZ+sK3iIxUV6KS5vN#C~eV#U!klVuG{17SlNY7V=3u^475Y*2uTBNALy(f%HEuSX=&s"
    "$%*N8@T{pF9{DP0MITavy^uS7Tz20m%!)VQ#z76PvtEu8*Q5f4*W*#-"
    "o&mp3XyC>3`btmmANl6wvt}3_Y((|j8kFR5^ErM!IoD`vhr*ji{CKND0DXmuzG4V{?weliky^(dbKYs7<@|2+Iiu)*&aRDvm!lEm8PCS_<rd_5vo9Og%"
    "ME|opagiRvF|kf6Z82lec6#tZU{P|MaU@&^ki*?vEJ1+$$<+!@%*YK`V~-j$R!KNRdMu>k+q5-$4QS5r8TI)eA~=A-Np4Q$;DtSYd}2vqg{E-"
    "TgEenyi)wF2a4otu_)34@$9`ib6>uZpXL<tg5w`8g2vK6GyZj3sPKS#a&}@F^z5faJh?xZe=EkT5Jf#S23CWTu0y|vDg+iX1`CB=bXu)P><FRPZRr=|?"
    "h3rh5L;|+Ao$#hetxz|=yk1p6L#JecpB?DRNkj(rFg*IS?E!v&g2b0fzN&I0m~g?{6y-Sv(z<{w-eR`2z(a(@lhugYVg-JEj+ONfWQN(>$$|a)riu0!c"
    "*d_s0)u>RbY#TupwSJd`rP&_96=F`lE{qtEsm>yL}LmtVOAE7I;;KK8%|1ggm)9Gz2wR?;U1okjOlFwpjGXBUVSCxUDU2<ZAGe{`m-"
    "X>LWjMOsRqF(KhgQ(Lh7LI+XeB%eq)usXj)`wZT~@4X#o5&vA~Q^oZ6sTOp9TQ-*%>1>?Cx-v1iD#RSVLV*SPGFZ=%#c$-"
    "jX=r0M}p8dcS>6h9Bx#EzQh-WzKq=&7*?=Nt}pf&=}Vx61hb@gw}-"
    "QkfXudCC4_m%g39>=)h`AUJ8V1D<L*V|nhx#8A6frqjl(li2}bK4aU*9m+Gb$<jkU^RP{a=Nup>Y4D}mbt}PX7aoW_+OSxbHwJ#V*D%OdGyb;80d4x5i"
    "gr-F(yof$>fU?oZ}gNZ~QhV*k@~TtsDDFdZh-8<x=)%cMNWb-eZYRt2Fq_y*x}ky>Gsr(9Hv(<E@Z=QiF-"
    ")B|Z7amistnMI6Fi>@fVR2HE7>wP(oj%%hPr0#M*?jr~>{3?OgU=04tMzRcGL!0DkLSH7yzlOFjfeMKtsx^I2}w%4-"
    "8rZDOzb>jRM<|w^I!IE%T*0sahZyJ!+v6C9vkH<(oy&Wxmobcad1x1LT<{t073>0<P2?fsBx|2M{IMk5$gYt^IVVm@^BbPFFo(Q~IS$DK?5OK>Um*}Oh"
    "KEBf(H;W70m*+Y1slYeRcSrLOfd{Vx2Hg_4#|KwT`y%GgV7>QPPo1TgS~5Qo5o;WfpRFL5zUVN|sRX@|wo-AtKW~G@FEnUGeOW_%75lHg6^h|cmJPbGH"
    ")z5BWfRX~-"
    "dyThrP6qK))uSTudzkORnC9L>>+ZZG3}8Nl}agiO@1_Ad+I@c$_}Uj%QN<<Q&qvDL~}xsLz(YC?*(A)3v2A){wAgaN%WFe=y!e{@kWo~mUuHkjbH4$Ym"
    "$fCao=nV#gJEBkEsXLSVw)lLBHXd3{+nb2Jh+CSXotrb}gs}Gs!*V=mwXIVYi(v1|HR51N}o?^4Kc+6WfMCFbuQ8L;8@B)JuEnqmlePu|_EJzFA?tg9e"
    "tIX9jgaVIHYEmB8TsdTe5!xSPNG)9W4Kdah-J;puNZDt6PLQf>MMdcD8&s~KM9kXX|i!9z6vct4ol-"
    "<)@$yTkFbt2Jv{gFNc`LLRpkb?xS;%2<+PgZUZ-AF7M%R3jVEdu=U`eWR?ALm%6Z_!#zI>v;TMnh3bev4(EE2Gh7NjX!cvxS#cx5Z_{rY~t65ccouSWS"
    "q2<BCvU@H8u{{Aer%<{)Jq{b$PcckJR<nIFq2kE$00xuJZ@?%{tx>SEgFwW-Ioy)QM*7n=HsTcDEa1#aIJU8!A}tO&wvcbCfk%@?ckFuyI*S&P^sCR%L"
    "yCV_tV~9|(Oj6Mky7SY^KNz&b6<+Sqbr3_^V!P{&`3NzBg+)U|l#ZxQS97&gNJQ_E`cgZMJ$teQF0uSq(ZZ+1Z6NG+-"
    "o&pXC_WS`p7Z6cmsaexx4#R&RTg}%j$^$~x46!zIUU?h7U53Vzl`SXYQe7#tAET3x+W2S;1vGg+ZKOe~tL;s7w?Yp+Pc3*>vjPpCzZW?u_kD)wL9@?TI"
    "d)<%pXNlyi+Kkg!m#TR1(0~iiG^j(LFo^Tq9>)H%d_3&W+QCw*;41yhD9%%xePxw*HBqmZ5vSc1L{T3raQ(r|+Z)a$F}JEM0<$&v$~a!xL5`u1TQ%4lL"
    "rPhpPP!Tk$uW%>n_TY0-"
    "$tG=I9MW~k{TVjj+X543R|&<nHY$pv#hZqM1%Xxzaxy>Z~Ewo{Yv5DNn4y|{tS$xzMP<E=RWGIN8m<jJ7nc)P=xhZid@4G<E~RNygF}-"
    "l4%MYJ8>WB2kO%w9(-0DhHtj$IE(!tbMwJ0!7~YYddvzJK4Ce}|F^vV>snw$ua$y-"
    "zOpXXR1$b`TMq<x6nr&`KG;Xz$4`t2!iVhwU&6iuT?C#mA|9W8#C42c?C(~{`!y9xdn3k8jPKmuoU<y#Sa1zzyiiwO?`@I*yzo{bk;@xGKd{k8;6)k#p"
    "N=Ywc4HjY`QYqoEqr1vFqiy)g6CF^c`=;%@7Yj?>g&~b%HOAwPeRFW3)%!?_k7j`{b4wL-"
    "dOWJZ}uv6Vk={IZzJ}we+(o)4w2WN7a1IJIbOt}9W^9Y-tT+Letg6u5w`)<(^t&pYV;6Gy=Gw3Q6uu-YmmU6vg%7QK6r~G#&j3+l%T%%lltCqk_T+u#d"
    "&3Lzunyg9yG}df2AMjOWrH#Bk<GLz0rEUz+D)h2}K328s>}N7X&_!@ze$jJngg}#;N3eDE3Xq0tMc=gdb`Z5qNXXQ(Ml{Y^yJ>h6vo;*B>t;@Gtv(@Z3"
    "}2tJ!mZlg|}YtnP|h(x2QVewqCCj=yelb46vYXSAgSW;4cb$QvHq2k%gK?2WL*ly>AR`j%YA&5`|O^Q-"
    "PC$g#vA)~V5vdqR#`Lr=UWvjZ&ECN!ou{ccD9%so(%YX({k#Mu<~BkL5<g5muh^21JG&!1#m*lZ8a4ho90|4gUX|3-aWsH=ebp$0_lr%r_di`oA!Vt-"
    "@`F9)O602lTnt(mub=+P%}O+LPHI9A<=2sZ^9^R*)RJVvEME-"
    "=aGc7E`mdh}7P=$URsR8}4S?1Ghr3MR7_9i{H~f8vf9PZ6Kb=KJ^ZIgQzOJ#bh0t210*!ACiMoEO@65qSD!#_Wf{16p|_b)3NOWRM^J3cPfGUo@B}@IT"
    "D)W%ByonX_I9?jZ0~o==#(9_TXO4Xf1x-^;#oCb_oJcKYxykqFK-B5k;W`E|MXoU8as=63ty$a-"
    "Lli**?X_GJ&bFL@jv@*s}=xjmi?P!P}kD#h9=!F|cs_QUuwjwoE9AdY&GMLs-"
    "7jx)~cj(Z&(piWTGi#+&^KA{rzrQG9KT&!S(`nLwQ^a%?X>#sx2^%#Q3Kdh0NME#}S-!SVG>!w0P1bk=Opy^-@9+oseCqZw=-mK-Xc#Pa_z_z~{G^fws"
    "OV6IQkNlt39!nlNAhofAiu8fE_5zdXO<P^931b&K-"
    "0r0yx(sWVzR`!A`1?vMVz(MlsfS{|MU0X3|C#J_3!cPc?nDFL^;NK@jJQ6$f%$%~Z5*0(Goafb1;wZ%=a{pr*bg;IBw1H8Ks`*sPvY@rAIp7v^g9*>-"
    "UbXD#pBZVF5o;z*pE~`-4Rn)8Zqvrg7M)%G5Vs{BUwk^V=(H2E$)s{{yhh^lzKFQ{cnvLaVTxWb!I3yYQCRGUCUzJ;`+y9hOGfsqZQO6KPPkFU-"
    "Tibw1|dwf-UZm-#$grSJVGkP#>y|k3`K?Hkf5lFo^xaO#06m+~;Q*-"
    "iV8~!o^c+WN{z2Qja=P!!Ebaz?A7GWS7(81U<s_J(4GHI3u{9$g2a)harq}SLVt4XVXyF$PwX5T0CZ*_Gf+7;@)rZUyR7#j+iurK7o0Yb4biHz`_*?v&"
    "B4h4~pmR{$2C@=!&H|vfts{{Z0vdx`jI`Y!LVx_8}X%9%Vm0>&my-V&#HMDO!Bs`Ui4+#0p?{wQ+cs?SQi16y(sq4d#B08cn&{SsCw2*r70y^;-?lP-"
    "l&d{{$rx=^qV<WzYPvJon@@&yRlKY@=vY++&B&%&R*g%qjYo2>O$lQw^~o+#c?%i<%rinVN8h^Za@cgCU3QFs7)2494F`9ntrtze=uzzsn4me_n$(<oQ"
    "Qj&pGO3`)Z*$cFP)fD`=ok0fsRT-"
    "I@a}iUvcs%o=_X8dRgtsKR|OI*9t46$_6XJG5s%)tmd5OTRRdeqiCUCP=Mrj~_Y(OZ|a{=KDSqf!?NeP=Bz;2VVtm>8}D9>&fgHR?h8&`kx(O4J~rWhu"
    "Pe(&g75iVja;i!V#7}TBxgX-kL%ewvO?}cgg=-"
    "0~wzV0`D;?9Nip*UX)_|`=koI?%39tC!hCQ;%<JRL*RYuwS+^mz=zZS_?Hp*g`)we+C<D>o#)@Zqzdo&QE*)!ys0bj1@r^CAu2SX9M3fQ;J?;dR3tW>J"
    "od;!^)F7ZHWkI|Hae7^qs9To|3oiwZD`f|KvXhVV_t6sQy52k`5eW~cCNT{UGTh${&KO0z^jIO!CpQ`=gWS4r0mC&cDNzpvV6{yy3$MX`GE6I_%T%AHy"
    "QVQd40gPwF`c36!<yj!6gUz{7DN>_y&miKT;Pr1PFXsV?RupAn<^;^t+(~9}-j)Hirb>lRmOs33-"
    "32egND)$bO%`D6pKsy_67W%ZNBnWMAPdecjp$CD62&z+16DsZl}Tua*?UrQuTd+0T3{FYu-"
    "#iotraz;n5tw;=+bbRd8nA^isP{+PV(ajuL9K3);)i6DQ)QP%<t%-"
    "c#H2&kaLgD5rDGR_ytwE^VMi_g6<XonTn(&vw3KfcWD$=Ji(e6}1t9Gy_6u@={<y(b=uc#dsG?JY0n_n~f$dLZy;uUv3m`m2&#nP-"
    ";;Uay8Tnq83l#XRh0zK74xgC{h?Jfl6Pu|EoCKN3R@E6>lKq%_A{_P>TC1s&+;qvi8p6=u3%gWUHh^K~w2|4y!H`OL<uOM6}6W~)VPHoc>KZejX#PgvE"
    "I>!fat`Xcal`XV^fQ|2@4R-+PeOaA49+sg$0i~PJoEuZrl=L^#wfuAPtU(rk7=*2w0E1zFtoFC}~KC6pAz87ev=j2*t9$Kq?uv7ZUo#e+t8MkuP-"
    "7&bVd|uRx`o+(J+4B^7G{)7^4%l~wyvMqJvxdhee^>t03C&Ir&sR{|$h@Lfv?f<xN|}HKU7WDtrh;qqZ=LAhk~scj#c4>=I3ag~f(6tC17rM!T+{aPD0"
    "tX7qAhjv5b+G^a;t5Ocd4<Mv(W+T$+uIf?>#xka`M%^^X*a0!UV;h<AazVT%(q`(qvG5Brml?DSrk1*>A3*H%zAwx!Q9O`c`(p@%8ir-"
    "0yBOAJzrAL$zJr=VD**gva#d-"
    "sgQ!#PwDtbibkC1@WK6_iQsiThken>zQ!jHTyxvCyQe*alXqbopGariT+H%UE)88ALlt&J)4N0UM3_NwdhLx6Y=UCe`Hw~G>JB$9{cD6oPRj|QX=tT6_"
    "Zf0iwO<62PxFXGRMR^_s?_2(-C4{R&)Fvo|Ahn@bXMD4kz=t)wPJ>{3ED?59oosgOX7YZbAm}`Scsv=IdjO+rErW=sVgT6T-"
    "E4MBbiDeJ{Ge{H#wiR2gZ+5f?2Eo3GDPkK9?0t#-"
    "!2^O6BU1qxcQr+mjXwPL(qJ9(f%e@o;?Y0#4UHJ;~s(0p&{c@7@uIpRUK7Dstbx8?I}zk0f1!FjR%V_f$w*-yC-cgNbNlJ}T98|3p&o6dM(knCTka@|J"
    "x{PVJ4FHCk7_;dQmU5^FcWmXXklKsnzRlp8;-"
    "TNSYbx#|C7iFFeJ0tL<I_|KSed}}Pi4WI4hhFx*$0Tg3>4<~~Elx5H9?Z8r)c2*PI8=5wBHH}iEd6H^`-@hr<2})}v46G^-"
    "I&iWshba&`&Z_$E_^y6)Z2uh0}2ioW&U5{v8PbihPA<karT%)e$x{_OFlVCeW?AT9h_6`v9!JNZ(lO$$I6dmZl$+^=93YFN|JvfSr0dWztoMP&!aFGcI"
    "a22!7}>BLeAqfm_ETG9?D8POngEA%z6puJV{*Vo6|$k<hco%^z#|)pZyr;@9aPH=Z2w!j{{oKAL^V~yVS;NtfMjgXQP?k5w8c+FVRE&kbduQvKxxZ>i}"
    "iR<N3_P9OhPqbC=PnpDX?wt;KKh_VS^Ef5sNB#-2+e{u}Lq5Au1-6AnY^$pt=*`Dxcx;GTWzV0wbU!>Lm}q)x3o8-"
    "(;saoqeStj{nNy!ny!17BpN2>d0-|78E0$Kx~^<Btt*wU{;00t?wkI53~zlP`-"
    "*3x?MMON^mkYRdEd+nwHoT4hx#7N<rzVEAJN{itgtq|S!j^F`~%V*dZwFFVM7`JtB&ylcolE0N>;1%7s$5BAm;_-"
    "6Wwck()a35^H7{{8>YOQaPR!TcC8z9ReXZr%c4+|LKKddt3^zDO&5QN5AD7<@zE-"
    "H2C|_|$H}Fv#~Wiq<pph`_HcDvthfvW{x9j^z6trFWIWuE7GI#QxkVR^Z|F%i`QxfoH_9KH~)LoL&xPPYS$NL*|+E1BWw<Va{TKzh!)KWgNEG@W;Bg0w"
    "2k^U!<-zwBnkJc%rV84jbNRaHJizg}vQ2^4=bMU(|H7#ujS!X>0a~%+0y%rI*c|jB)W!sLCE#PoBztA@XpvpDPOch&Y^LA9Yn;SMLzvir?}%-"
    "k;|CI)wu7%l`L99ob*dKQw(LUt78A3Tr>9r{wo57v*y=3C>8lBKB*P`MQAYqqAzc!t`6<>z1*%m-"
    "D#qaYJi)f2TZq=<Xi`?)t|QiS6WjJ=FKJ^8JgiPQGwnB=?v4u9f#|UcU0h;9c_i82gjTihM8Qr!QWf75Ey~`B&*bqvsdJj;~S|2GXZVA3JGe0J>;I{Il"
    "rCOwtcTy8GkmLor_AI_JxE_8shnwbIZ3V6UGe?{jrdbYn~f-"
    "iZ3XoS&^f%UFet$AV}l4Emv9F#E5%^aHop^Y|Rtf~W@0SgzK>o7z75teEHHUKb3L&#@GcOU}qXuK7I|95^lT-Q<yHvhTaU-"
    "35)e2)rD*xu?`Y(?=JySt{@fwp=Isqr=phuVMQzxxf)yyC?`FXQv((<NKsLW2Bs4L#|pNpR1YM)D>~I;=IzRm!UTVUbm+Q?mZIY!>QFhWIn9f<BBL**Y"
    "mcD=kx!)=km=JE9G^Yv(&)L5+7o6L(Xu4ClaqB_3l-"
    "m8%)y$ewFwsIlfL&HzY2VJhoLl&;L)G^llg{`&=*P;aHi6N*{Nekh)n&kF!&rSN)l;m?Hh5n*H)Z8Rw%`u2?AZYKH^sh{tHcc$Z6>ft8ybv2Tg?pRYU1"
    "SbpJrRqG9h?`8)qVoqp^05hnaR@6t6DG7E1Ow=mYBm2jjoMRvLcDzMbOx2m7XAky+`ko`tw-"
    ")12`MEgm4C=V8)bS0=JYn({`FWmobWzq(!6q+ElYPo<dg`OkWPKekg1hmuznDi4A+N9geBzB?%>~|zcv*R!?R^O!)M+O08f(P0x_>^Qz!Re#1@1wesLw"
    "b=Fjp#---&Oj&d6G>MF{;>BKu-T#yjf%0DP?OfF-"
    "N}|4{Ng^Rq5}eUZV9umbjoyssdIaoBf)oX+)5_}vZ5(@a?Zf%;BA(4YRtkK;G5>w@8L?J;?;f+OrN-gDnyQD>`P7>kwe4mh++=yK(2;<=#53rFGcOB1%"
    "uRnU;}ce^RBGb9{H!Q-MP#HA?sYJUEn`EW6Vygn)c$$fZ2u?kLe{(IE@FYKvqoz1{TO9y!LQ1F>P=P>oXkn0&Yaw;Y)c0k5h1<x4!-"
    "OFS><~XC*W|9A8I7iAlfrq?yLHVl!Z^#^JctPObJ-o2spxEEu%&QTsu`SG%jl&nBCl8oU-u_{}ek<SibXzwJb@E00lNq-"
    "N`F`ksV;bUcAAu{?$xBfJ_qY)XADRDUspCcMRj5^td)hq&pC;+Bz*mFb70BD{r&_mQZf`G!;hC2B60Jc;#-"
    "}B9<hl8p+O*2ZZ(s|bo(h^$XX{FxEx6@{I`Vq#;AZTj!UVpla{&6L3tYuMda>-IpZEmhw|oz-"
    "2>b3avhUt8Hvr=#e=3}RzMQ}10Uubkm)FTt7=IUmU!PnA$E2@#)s-43#|Lfph0Az>cWq6+E-"
    "vs(Cj+6oE${Es|D=@__(H!>to0IcbEO{`Fa1D`Mj=>LMb=9r){9(!?DSBK=_2sD?CYD#_dpN4DTetI1wMdrEGFL@ZGA2X6-UbVXX$h9%jZD8-"
    "}OU!JAwC3qo0uP33e*ug$we&PaO5Gt?Xy(FK|bL2V(s(%oB^r?0s3=2d1>d1basuo~A`D>d4votRZ^$o=KB%X{;lx611?SpPnP%yWBq54Q^xPd#SU@VQ"
    "*!B)Z87f<ol|H-"
    "1D9CI!x9t5A>2eJe&MrCHY};t~;!n3w#Crz%O|{VO2Lbyb2ZgcIN&|*^eX^cSW9^z$dfsaFqA`r|aF(PhNl6Ypz?_?=N5D37=j9k7wWAM)uvlBfOEjQ1"
    ";zpxmF8#z3YT8ejF3H&2V6*ydURbQxqFt3A`Hhy_221&Qm22$E>CAqEE<`{=uz*9|k-Y<5w{b-"
    "(?)C&+<a;6#~D+T0bZGc1?SCq)9)&p8WnvUf<}s+7<3H4*$^?IY>P^@Ye-"
    "h<a^3a^#2d3VQZ+(V>{#^+0hBvcNLV|$Nzs)jL#YBf`U7OH?EKuPM;R|mn2ubz9#TF^aGX>Puk>)E9c~WH$_|nc)^~{Z_y>ys61Enzo6j#e%9x4>95Ya"
    "vZt1QjK}?aLf!|VfB5=Vz8AwV<xrz8^4!)Gc2s3mb;F)2T5MU*zDVkR;Vlmszl-"
    "t57>8Rj4);CX(M9^q7xW1`uFLn|!`)#cpW}T+e4E6ZlyHZu^mQ(rXM)5#mvTp2`JU`e;`ws^G|rzSpWAWS%<=MmQ|M<mTs$HDD%ajrK7X<x*aIE^%KJf"
    "#%@OHOQfj#2;AMg5kb{3n-*sS@E8blcco^lOIgjDS{Tk%55+gP`!SApZE{@D+?&APz-"
    "k8}_(Bz{dO63u!jvS{x9^$^azZrrau8we8r^Sl^_Fp{4FRrW8p@vwv*&gSa>vcG`ne;OkY~0ZCy?h>Ky?FokUw)=~z)HSPca*w(Tl$&hB|H%jDDbcJ32"
    "Wr}PJHi)sq+2(qx>h08rzi~BBxPjRpYN7u#eQihVh>%_wj|FCtl0z1GSh}Eo5HRJmrQt*Ts6C^1S-"
    "f8&#)I+4g=c?0!1nK@Tndcmc(kS81&8W1sut#t;)+J?V3ZuV)SxvY$v#NX6%L6H5M4V5GiWrM~CUN4M#dfL@#I(VV&an>t>Wap+BbuW)|^bJB!ddeGzK"
    "j2lbDdXCI=!W)_YTiCn2SuF5TxlY(|O5iE%UCPMk`i4(+!g~4Kod<iMboqW|m#t11x<=qt$Zd)9<aOp*PIx|7z9-IFzb5s4^loS9_X)hk1IB2jz+*o-"
    "V~ael`I<owsrx@4yWpyPKdTmXzlGF&gW3ywWk2G-"
    "iR+X3eEoJ2RQ@7)BbdC=UhvY=dzEon>QO`X>Ak{Lc*tpOF9zV#Gc7uCEqT=Sh1B0#KIDYdFw}UXL&{nUETaFnkpBOCJv%fmD%SIjabGR(dxnKN;=6oq;"
    "YUB_hkU;AtKI`m<?{wf)Y%I1e!9i?B6!zK;1fEr_T=^Uw)1?EEAMBaGq6PJ-RDWZ_^-"
    "e8?c?a%rLOsey5X5h;AUD}MqPVH&UH4v!;^<jh$>WYivH}Xyl)%Z$qi@Z_g&N-!Me5-"
    "^E6NNz=}la&j&G&<#mIr+dN?{@AGC;XZ2EN8|l2T|FFQX4dEWk=K#;m^*}|*1I-"
    "x@FZSLgv?`3B<Dp8q>k5xsT7<C=|2c{^@C)czas_Vfa7O!cS_~wgccx|^V^7#Ht`}<QouFnfF`9kfv5E8o^xS@yow2OBBdk|xv6Om!iXJnZefk~GQAko"
    "b;p7x8ZjeuouP`@6K&hs!@xo-nybvw+kp~{hIvwlbjN8@3Iwx|hv((?MPh8MX`l5lXi4f`kTUU2OwCtnTvnD>t`|O`|?r10d)ja0DR_eQ*)&q^?eeiH{"
    "RS(%8xz}}Ok0A2If;{k&Yud{Es`|JYtoNI+=cIxqjKgT+BTt*(r5}S@D;&|_g@V%L?cS2l+u69{a}ntixc-;&`uV^EZn#!U-"
    "~+h+Vsia+54mGVl6>Be>l`H4nfTHJ`V4{FvZiOq>l^WtJn&53_xB{vj+1=oGSCfKZ3X_DytiDgXVw%~gjW@Kaq2>Usn>PiyP%xBE)mRre2~<I9%EgwS>"
    "|m??yrGd*_nT}Ioa6!(g9w1%D-!z6%L5;C(1da(_VSsiv7~1Bl3NgGA?K*-"
    "xIk?{XHf1_h3s`^u6%^^?kFY8@|iyDn*Eok@$=xH?)`dzp|vp*Wqz4e9uvB%5=eqwF=6x_ezoPJ#{?ljPL?6KAv3OM)tYCBVF-a`n2)nD_6-"
    "^vr4<ea9fPmQqShe>%hOpxnjP&pE!x@agyut4R^&M8RwZiFB|D+7O_dLoh-"
    "*wPv=lKhjX3A=y^C(#tC=%Sq0{QQ);0DWA`^d6Z;l9VgUQ)cI>Z9aBp0AoQ#A)xJTVl`fB0L-"
    "$%>ij?uazB~Q%1k^PY`&*2g^JIl8N3P+g`)KiOYfq;&98TO}L;~L{hmOb3a9r0o0G9LF5^()yo4(FQ~;n!4)x6J?MJl{5?h!rzju~ef&_l+uaWL{a(KS"
    "VI!$K7#9&viQdKBC3|&fA3ZU#HF=D(8;hMoY}h)xgG`ev+}6&pL2!-XH$6Ot2lSMRA_*1;$|o&-dW1>G<5#3C-"
    "wf%<rIt%X}ES#ubI~x@9k(Z&`W1z3aMRvAm8^)x`7VJkgA275#E(!<_Nsq!tf&-"
    "1j_(bNp=k`~GP0#Q`akwU|R)j$}TZTTMQ6OhI?{(OIpunCVYG<bK6bf8QSKgq(Ty@Uqq-g5wjY*OsimgcfNyTGNDq-"
    "xLH=$6e{WI<dd*&?o`d+S=plZUv>NL+7Y#%d^ZEF_Z9RmOWZ+QZR!0{*ZMSMg8$EKL*Waa^X7_IC(I)$ZzTN-EM7%;L<P??yghtm-"
    "@a`>ihF^PWU71;ywMqCg}&t>Zr@o_cr1D@6GRH^0-OqnW&U)LhWG+&ar>LP5kpz`Xl{p+%P(z{3rz%*>jwi&-"
    "FD8bwcQWQXkoOHz98{W1sZ3`c<s=bws;)3ZlrJ9T&;(87SclU+JG4aIYHem%N(eh79R%pL0KU$$DS1#~p{)%l8>W*waFVsJ6iF)Urqp(_`mwHENK<T1k"
    "I$U2DLXA>w(tFy>CGtRwg5Ml@8!ahI62Lh}DUFGu($2)qV+!B+A*PnCU6aG4=+YsPn$d~Z(u%oXD$FZE%(O3CNbk~3Z5DEUN7joST59Aeld7nGKM=5Ho"
    "@`>#B{&a4lvIMP6ipRK5s)L%FHia(yLjce{$T||e=XDu+MC;5x<Oo`-"
    "vw(`WZuR63?t;P=a>CKo|vlxeSZGGSoWQF^!H8|Rg9LBtw#?QMB^h522R^)FDF4hD_(Dyc??`V5J2<_ilVbXF9ylT<g(eov-kH6a|7`6wj;4n^u1J&pU"
    "?r;y2xX%-dqxdvE3Lj}Ox;%M@d~P?^d=0%cN*}St_IL$(CD{Kk*6xg7y=CFB47Nr7Yz3Q&@_blBd+66<RaHpp2CTiI;5zgC)=tsK#BYzrktTM?VLrEH-"
    "EBKc{v%JkJlg;(^X$+zOF<0l{V{W52<IvPy%W;k7%`pRt37q(lll8ts55t7r6Q)E3Hb*VoFPvhr`NV+4#iaOgF(9;(3L%5A$eyX@t^c|8~2YxeXS#Ia&"
    "NTchnwVcv-"
    "C`U>V&Sp6*$x5)S!o*!FtT6@0feb6?+eAv7xsG!s)p(3+OG#SCjlbacGwg+Xk!Af&2B&oYRfSi%xzhonZyPS{j7W2ZnH+HR$gyE%(Rsl6oYr(O_EwHJk"
    "An*axUMxG1Kt)?>(X4aW5%FKp*p6UlEa{NQ=b3eV4LaJM_qoqXGZ`?56JA0Y-"
    "kY8Gk`$^C0c&UPNdW0VX<*ePptDytwmf_y<9i!#3p5mym6s<JjZE9lO3x8Z)>TF&Ektc$i`MmUa9aG5;tjX}6VFLH8FSA4l*k76?!PwK*TYT<8kic`~;"
    "=)B2@pjHYFdYkX_0UyY34wG7;cBT=lA{BUXo>S!i-"
    "Q=1N1AC!Yx;<<fa6Jw{Ip$9>_FKbK|HGasCRle;@aqb>e6zeSFwhCLhlqNK=6*TR6V&FOy{@qz+1nk_V}yeH%!f~`>od%gz!sS(*WLjEEfiGbp8Vz5B("
    "s(^aYozWVxA=PJEn}Mg&L@QgP!rRf~LDvsKB+PQ+FO(aL@KS!H52&+a?w47=uoXPiZ~#C)yR>MlB{?v_LNN$>yN>Ss!3<KNsxlq(!L=6+Ef$C%DeNq#e"
    "J7u9%vu#f5h&jL8+xYnd;<yX<nqgCAO4t6+gU>~*5puZ<-S7zVkca!DPoHMc+q_NVEL#U3+1>~lx6x;of)u)t!*YZ3D>(GpOsJ&={6L-"
    "YU(1ah9m^j)`z#}9Qw{hm5}*<*pL+$SCD{Rr1T^qmJH_v=tT&jQ!DPesWmE11vr`#iDeoesU9T3{!2<sHfDJL4U5(+lfDEaBj!#@I>BWBSRN+_%^~Zv>"
    "69M5nfDj3bYEKW09#UTY8Y!SvIXc(YB7*3-nb&-VW2>pfmbxo(Np1`VDt?>ACMY+~6bl@CI-`FgZIqCo@Vjm-D!o0#7d4Z(yW);Juj;7t!|CiiA-"
    "J)qi{AlU8I<As}o0lmq$%(q{)Y1cXg(zEI@z{va|4|+4-GTJfz>Jqqi(Hbx6DtJYH%jf)WSeI4Xlt+U|TV$mvh^WLl82^-"
    "kJWk1I_zg0^f3ku{^fP~`rw3U3wXW7dT7(^9sdv8Q-"
    "~C+wA=YL6$+1w?F<@{w>!mW#?>;pn8Mtg)4R0I_Xziq6J9%<8@usOjP(d7`HyZFPM8R3|<V*I(&-"
    "(+7!y7_B#12<DkHWt73h{g%cVPckm{iq>g|!q^qwo64eMuh9daBwE4;C2_Q(wVL&XdRae{%i4j~ijRVnkR81tVkFtFV_Izyg@|rV5Hg8t~@62G^=mN66"
    "=H<d2EFqtJht0sV<DtiXC>A2E&biL4)wW;eup3f0LQZ#aL?6z1sqXxRNS;NyGxF;ch{^J*gF(Bxtz1Z+3JgZN#>%s|fAW_}0lk~12!Z<@;e%VX@1vF^-"
    "Y4fHt56&u)B?klB&BlmE!yuW{_gdM6p6ncHp{EV&i|6$HXXkQ4tIP0sw^v}bV+vByjsO#V6@7a;pqq5^n7#1z?6~uo@{MAN#v~Mo(UCe(M>btpBxjEYr"
    "kB<oaBlG|4YjPy{!E%rj;w-"
    "eVu~%X50AR~o`d(k=VQDvvsiVVA*3qr@%vXA;4ve?*(GwdMTOxm?8j<Av&Fr&w)HJ_O>4V>ut?+EQ8U<1GUDR1G=IQ#o0ccuXkFG8n6fMX6Al{7mQ&_V"
    "E?)zFpouq+pX=21f8Rv72!*C_d8s}$d(2@CYl>0JlB;%YO3BQRp==Ms3vqb<O^2S#Bl%FNyv3`v$)UgVdlAl{}A6C<sUWko@<!@Ugrz*(h_*?8j9my+>"
    "d*U&b`B{Sf&QbOYo9M?%kc+?fZiX4RxHree_kDep`dip90aM!=5x0%`7DO*|LVTx2a@9yYay8&?8wLA=#dizMXS}^CR7BGj20ZAlz_kqh5BD;Yc=E|G4"
    "CH#OQxptk9fg{8c@A?ot2j>evBlF~3U*T)XK?-I+w0q(J0Xx>`|4~9JT^a9&HC+3e`@#G74`HwG-"
    "{|uAM(`>^4k#lV%1>}bjsFYhm9KfgE`M#vo`W`cW*@XutdelY7Fd4Kf}6nW!-"
    "<U@x_u{OZaE2(UEoH%Q$;;U1ciyBR#|lL!YZLiTi$#b<~XfGw*pIh7?$#c_s1~^|!)J`k$8Mx6h$iGggnGW9Y{??h7^PDfK_>OG!*@ZjCyJG#JkM+Dcv"
    "MJ&J2O6^Xvywpi*x|HJ*gL?4nyZ|z#T9?JRH;eC4r%gD2F)R#Q^)(6eI!kzx-"
    "J$=!4_7{IxOKTX9v^VXry}1!RdnoA0eASu1Pj3e6@>_GPJz$3$RTQk@zHDQj#Ev(6xY~H~#DJ!@3c8W6Z*rX-=w~-"
    "SY>bg>?C{r5fsXTRWSn1;*I&$Sfr~yyj4P#JEaN+r_%iwm<J>Mt{A<J&&Ywly@5bY<<NmH)(F2Ev+G8F4;coUjKiL<*Vt%%n-"
    "Wa1t8qp^}!9E^S!`|*5>n>oo6P6xO&}6d;y~!D$>6a=o@2Z|~!O_oJET>K-uQB&mKuzXp#}GH{F@HCwl^Sh0z7O-"
    "ZfNfFHN3M9fOpE64Eij(zt3z%&>qNg$-4m6s=}=lxqxmes=aV@;^@=At-_>F36ALWj`JbS!MN;q7QAM!E$r7)=S)e-"
    "eDvdh$*ungMtvCKwvcwnWRWRc?mvwrN>)deM8}1b71x3|(Gmd#cp0Qzm4x8?SGm|WlS4E8p^o^f5k3Wyw^R*XFwdVYO<PGxsH1<?iUCCRU{IKZ}``<ol"
    "SPlbBAE;}@o0RfL(nm{J4OAnCI&t75<3oJIL4OQWTcOb)HENmPd*%39=I>KFT@-68Sz-1DHJ(v7+mpxIFu%^v3B-l<R;Z%Uz=8A3U~l%_-"
    "+XV_4>_4us7Mk#(2cR=c-vC+YwQa;JLyp>N`sor<4f$dGQ*kER>ct4gn2$lgGuZwrtrKns1J>m64<g?k2C8uxX_$D$@=P17dU-"
    "_^*+KHmp<{h4OlPcK9Th@`9L`SmbSq-=ELb|dNk@+0r_x7jjA}2VhdGQ1rF4yNcO)?(wJKtt7F0l1OBq-"
    "T*u=%vK}AMFI4Ft1J5(IXwy`|ZQ=<$ZbBF4bmLg~OtZxh&L38tdd+qI;&GQnRYJM*HrRSsgAep2%474rDf0i6%6L}U7P~)c5KW(u%JJ{%!;MqoFz$pc3"
    "ZURbMaG%@@60$IdmRredjrns6+C5K@8@xUkw5-Us)6iY)LBQ?BmILr@j#x}Xq$Lw8<7t!xc+4E{c^X=-"
    "&Z#(6vjaMpC%eOvA+sry>})59Ng;%B}9)uUK$u#>#^*k>$C2?PP(9anhsaF?<R8UQ1d;)V)U{V-7qZ968EpEkxBpEjP-St>zdKu6LafWVUdRhm8h|mSu"
    "ep>)bbQJm}={gv&90(%-"
    "?T9T`)gxl&833afA*N=u1a(j%n;Gr&7mNWlSh+A#@WfS*P5`b;LivaKNBUft$bgwikWNK;nzi95Ha7!2QWnNsOEMz5BW=9kJ$?g0Ev$_(A=hPQG<u-"
    "_~t{6WUeM;`v?+{Nnzp=@XXl-;Z2Ru(K691uzce)g!Fmv-Eqj$9m(LMvq6Q)yQQW2T{id(oY`O`=O4a$GLfGgfNaTKQq^wnxDf7#GwA{o7OVV$%ijlN1"
    "wS5BkPxe<xOi?`-}JbOR$cNteat#Dj>I>4T9NUKV@CDX8w1g{+=2ai#s)Jk-"
    "b@iQRKZ5_r<mNZwG6myoUiBe{lQ*!P(|NkMfPE)KR|2O%C?WmG@2d*yE*qzNjpDL7~njQTK*DOvJ>EMx4o`Z)C4?d9Qpw<+ll|Z;J6<$vgA8my6BUsg}"
    "3JW$MD(9SYu3d-a@$?GY;9bil)JV*DibG`*QuE4VKMvmG(pM~iplse$Z)RMgpE^Y=jwaKfJ2TAZ1p!tzyu&$HPtxWu}^ajzCP3oT$^pJ_>tHK8H-r-mn"
    "%<y+#{0@geI;y14K8h!hdG#~hMwL+z48cbz<?qw{RvTnA#_eFA`6_&GKXwLCj&&2y-"
    "#SZ(z;}z%Wt;Q|$bBXj(?dcc3p7ckfVpbT}UyU;4legsCk*s6iIU%SRu1CRVHS-#%$GqCpjPd+k0^3y9s9HyZp5&8zW}adGWQUc-"
    "*FtL)%+?^5dVh+#Fk>kBEvqW>{<Fp85A4&)k58#d6Ug`fWz<2<*9N$RDOlwt?n5m%e@EEtHn<pL#7p+AA+F*%DD(Gotkm?vfN*<s9IT)>_30G-XV-"
    ";{?dd+qjI@Up{m()3^VqDTSJTW7cC^FC2qTJBP|%3{@P>NYX%M+4A_C>pZE^3q2J2Y2=NbP%>OfF!_U}(^k^WqRmGq7OasE{LhoJmQsN2$jm)|sKMn7i"
    "Hc_NvISHIQ5mWy_%?ylekec%`3hp4}Ao;Amdi$)krDtN*<rkn41P#@iXIH7u&7M+@^v6uN{+CWY+>t>b<$}ZKS)jbQmUSPhj33TUtRtX-^-"
    "qAtrqDJ^caXqsM<LAAU=5m51N^_n$^Z{3?&0*|^_Dt}^%_vKh+@t>Yz0iBP@41X$WULRCCtKnI^|u-G>d#B_cScfU8vCODL`zJej(;HDo%z5N#QSo-IG"
    "bgOhMz3ZYc%VG^}dt%l*ztuTT5K0h7ZRNC$1)bFu@nAH(O$dHP^{_eR#v(op|p@zPND465YMj_(>gn_s;AM*=J<?;+wq{tP|DnByLH57{>aZpY4xTE>_"
    "r3OO3be-"
    "!_s@)Z~L6K>@hc#0oQds&S<c{l^#99YgT3R3IG2T47L@8jqRh$3Afn=!4XOLGa&ag~5!&2<k*%^33KC?#swvymHhd`@Q<#_uZVLt_|Tj=N$7w(nU*L3T"
    "Hg2Pir~8J@t2XOfj_1)uUZ)4W6|jH}iZ$sC!A*OCrO@8V55q&@|?GQQz0IJ|=W2i#g=CX*V?R=kM>SN0VEyUSF2Sp-DCv@1S5Y>&2Dz@|^WjaIG?&$J^"
    "ptL+Tg#Gmri@gZ*sZX;ElY%>XC%(H>@9Bi@5P`ogyuOr2`LHRgXJaTVvO%>Kn;S1eYxHlRf_1%23us#vEbsizMcMq|QZTUfBKf5CM&rT!+EeL+SU=pt="
    "k%GF>XeM2bmzpU#U>hkDy+y?tjYrsR{$WxXO`>Dc8m2ou97KsluNM)au!Sh;|Kn~hh1)rYSqWf3&kc>-x@=s&-"
    "#YIa+!6DfIYwC4;YvSYqTk6Hn0i}@C)dtNMYH*0$nYx-eLZ7?%t1Ik}>QF6Ojcx2_A5&+|&#l+k>w=qgb;xCY7O*e6z@F(>0FapEfh)|<N)yzum`~qMZ"
    "EhWAe*e!4mP4)JP@qOl>ic){a7Xj^I=K3w#9%!d8#HjCp6(}qp5ng0P4>d`PL`O!Jj~-"
    "U`<uTzmU!DkMewPQCBCPsag_aA5$fQ6^7)4c4rt7tpapX*X9|!`Uon7ndOFJ)tA=SYCrFJX*2OmVg5#KXsW)8E^@$D#253;9ar0vT9?p8b)T1a)y|+e-"
    "d<}lFo`<pCgXx$16f2Ll{<i3~L<8p_;bUjhZ?=z##F1zl{4JuvR`d5RzUMkw2SEp`U~p?2RMl%RjB(U4Z@t)u)j3%Y5dpTiSy6*BUxAb<@;=l`JFHzL`"
    "h8p0=!`k?Ijh=6xI7d1dG6swsi!&Q+JU8nuB|6$f0BASzrG3Q<o6FRVk|B83A|K!2l(%l-"
    "+xP=Gg00rYMktdbsq%YgBlwy@9X(5bVQ*}+;@1ri~X9sj~QLw30vj;{Z8x$Y|P)6&iS6-Nx^0}6J8Z6u-"
    "HkDLtU7_ed#dK2|4sLk!lt4=*L{BBNv!IbHBUbszHYf^z&Jaix=1bnRWc)tQX2gSz*;x4gS_=9<%2;`5!%5mtZ6{V}2&Ek7HdF%M<T=4)qGfS9=?jE~5"
    "PVp5%Qz=k2UJpYTw`kD}i`sDY^!_my?Aw-"
    ")tU8;r^o^~it*cUe~o^>kkiVAskJI5yYAqfm{W^jFi#!^^1erAC#+7sVQv?KQZ~x<1UhvmifgXc~c84Q+5^um+LT@2%XMt>ncXSE?aD$reS9v5x4k4%5"
    "$jG4Jl!*F;VF;YDf%rJ0xAuZsI5cPF+$&J{a2#wzGb&sb9KZ_RKAJdxk)*pI%@o8H&3mVeX#!}JaGpgFAHYvMa@|9ywRGDj?u&#MeK(qD4FuCazQ-"
    "_AzpM-y@vDF}2G@56Ryo{wA83u#^L5z|G%1N!DX;tF-iZ*?5rJJ?}|i-"
    "N1<<?6)8vCq3%zXq1}rk?sLSWbVqkb2sY{bDpp>v^CN3E>LT%+J$u{2S(O^YHHYMsK&diGnNSg&iF4#C?3yu`cSjG-8gAf|=a+DC+MS?pNFE-"
    "WWI18oBE=|NLAHazS26#*UNL+@ZswX=<$NNnT()wdB#v@c|fbtw-BSY8<8yUdR0{OaJ@fQ7GOW(4)QXf2exvuqw7UZ1~1KvxVK8ZV`|WX;3KzBt$?(#X"
    "`Zt?(XiwPV8>Q#;&8s?(XjP*!teX%<uc&_m6X(%g1r{nrA)F9TRI24|#m@Zt9I@^tCO@;aI2^|FskF4}Whm`82%(u=|D&J}$OG)j29ek*7|w_ZITH>#K"
    "a>G0O^tvsK9D@qNjUC;g<_@<rM*`otwF%;51Z?Dr4EiEMXYj4QOl-Nh<6^7!8DcWdH=-"
    "6miB*=B{&o9NT2122+KdoUi?&<~~0TcPIx6;|X(n|lz0>GQ6|1i<sO23qPy$zQD?&pzRJ{66_(uY(5npQ(`7iulQRb&h|4bpXnyXz-"
    "+zfIRZbdh+ZH=FM?<09Lo5Kc(MsVEi-"
    "3Ig>biWo7`rjMc!YqJXSs^81q=$fw&^RlvG`H26h64QCy1r_SHaadxfmjU<%@pRTAdlznlFJe$nEjQTeMD=O=da8p1n@}>jvFeX>3|2X`J(L*FQ>FF!O"
    "S<`v+$7LJFqFrr0Vp!*E$Vb=6?`4^fz^r)q|JLJB7ZJmF{pZL>wfXxW7A9fTW&^Iyk`^hW=B5Aj7z7O6m4Tv}My%N`Vj%0b4UgYT9aOxa9*W~kC2O4Zp"
    ">B>LFTCwV3<}6VjZ`E0ju3HxI_v_+W8(Z=3apM9yA8;!B_c9K{yxQ?b>t8bjlu2osQ5=fd)CuR_VtuJYOHS2xVBi2x`GJns>C0jrzLS<Xi!y@jxgXBee"
    "OJtzqfQe>#ix1F!`|o!<YvH^K+fY7n7H(HI2g_djmF;mnRc{u2UyOWwEaBhN9|x9W=89jHdp+Dm_o<bq7<wIX|?*Wb$UrGGZ?4S(Mgww06TZ>ah!pR9M"
    "CH9;E-O%zV4o4aBT}wD=rEej)C)r+yHLdkxcqk=96uTD1jO5GRYNi=J{GJG75L%1<3Wb`y|EK8s-gx|5gV5BMN<6YHn5fH=mhQHOhQ-"
    "i{JK7wn}UpCI5X^SiwVYm2&Ab<+{!sp~2{5pa`uwUOg=q<1ac+Z6e>CX99y(aA<$4`e2>zE^f*6z(?SDnw+{_gV1#Y4pvrpR~k-"
    "WFyWT74VK)a*M)`+s7GkS=r~P0^>cW$Z^-"
    "3oLl92_j1<eumADpWE1Xhk>d}Tg9A(CxKFqlD?iBbEMjmi<++<@XKP$eRMyjD?zIYUwq9z3o)hHw4f=xP|NG9x%NFI7=im*<cWKIZmx%-"
    "Iu<eh+v&6CS%6A#r_VyU&BlG4`a@rr|J<Nn$2TaaV*7;e-94qURA;baWmG7$j4de;>k-6mNKI13i)G-"
    "@eEUQKixuOX@%nRcEh9U<Xf2c<HQN$(IYboYZGm6*k?L|GRMI-X2hV^lmxVnqJtc-ggc2Ce_=W-F7t4i-"
    "R(jQM}UCa!Imi7O*hlp3f>=j<`iW$VF)6sa@&j7DH5q8wW`Q&jQ=^1hNaP)qtD|uG2H$i&YlbAvr$aoZr?&q}dZzA9pdDEA8vXD4XIG;L-"
    "dZ~IYeJFYJ2Kjd+`{kHzRg@3X<LM*;HpEkbb2g5?@6pE^=-<Ww<81-"
    "kJ@}jKmt*9qq_BFhs%S)lNyIc_xj}gkGcMN}rJBh3U&EgGs=ROcaH0)#%KHohi5cD-l=rbSY%x!HAEPbju<}l2J?BMsn6A7>{eoP&bC)tN-"
    "|R5#vGV@sCHWiTlJ`r;+QF>6PjZ|Xa6{qor3Uiy3}yc*@_xt_&dF5inQ?st6`HW0{vv}us(^KJj5>2cUM`kZF{5o$5j<td`dWURubvh9V0}All<p(%OJ"
    "XD*j-O5aeyckknyg{dP=t;;vo-4~m_74vK|bC#HKPXcXEE`8C~JKv=PP$YG!EQ0pcnPkWcuWM>H9L`PmL9BDDR@d>i_|v#ETBZ2Nm(jc4s&|%IdH-"
    "i9AAl+bKOiBHmQ?kH^Y3dc2%LzF?hHWljSJv(7&z;YbG~mR=K3!1FHQ+;pu@KXo|-"
    "^9O5jDn&q9;!<y($BI72v_25ethJ~eEnqqQ$O^_AQjd3u4uqAUC5H&8!FUeO^NV%*ad047nY5Ti982SQ8nNg9GXB~s2*<s(Sj)O<NME;=JU)}>Nz??RU"
    "$Pdi#IfPTFAv85@O)(=qVd?4I*mTaiT+1Ve(cZjtjrC?!O2>bW*uti%d8lm%)S|@4ujQNEk4W_u$0$z^Brr5{7^Y53}HvL=uO@7l=1dFe{J%~>2hIsd_"
    "{|)I|UT;Ji{4pCFzrwp*VL@i@;3+hDh(DaQw$A5#Rp!<Dsn<Tk8nOCVzKj9&WNPGU77vX`2Dz9Ynn4aYxA`((~a@amjdA*MPdcM4Si*I<hZ15{rvhq+!"
    "|<1Kj7*Z*pFi+>+O;Nq$ii?<N{y+%MuLdELbM+ChBy{=PAG%r+sDHT{D9)#RG|9#q<y_OM=MMyQ1v@nxy6SZ~`|FEi`3$E!j!I(?S)VMFq4kHONnu8lA"
    "@#e~{hM8t6Xp5)VN<cB6Smur?9QDHEBF?nSm`}%Ne;(ufUrsW!NzoLl7QGgS#D^S{3?Rhn<KVd)~bpj;@4ziaVx)Kk*R>#{~M$~7&tYIEfdH&5jzS_5B"
    "bkG^$A1-1PaVCj6<eKzcT}YyqHQ<?AgqAvfBz?D;^V2sx68qlj;Kq4u%Q_y*@ksdBrePp{Y}BHfpMbZ-&pkZ;E9<9xRuJyq)?%-"
    "ZfE%oXw$y3Oc)szs<6v8$$D%$0a*2o8)MIbzG1uM+m}O%?k7)u<G2VuHtTX#4_Io@gxEWAto`7lfIUe@u1lHY`WufSDjQsmnfGv;v$vjVO$U3PXfiNpQ"
    "cH4`{As;0xd{jHfjE!YwJn2u&@F|kxRvI%J{E~T1+OO-g@;-S(Uo!?ikmC--"
    "+HuPJ?F(m`(0#caug@IRP~KNJhnh>?O(=N=eOJB@Q1Z_FK{HOhljDn6I~EFlM!H&KmWMnZ9X-"
    "kn<+=9CHa1w@U5>w}KisT*Kk{y&E#B;w<LT50*^00H<7S5qFXi}a=~=q+-twZp_J|IYc~HWt7-gM-Az$qAS$S{y+%j3$mF#o3)Bzop{nX~JqVK-"
    "<KmK(<E9Lucqmg+a2V2sk)_vUyAq#9^s-i~Sxx~%O^8OlO`yKHktTgU@QlTyTa4<1-"
    "_juM}Mj2E*rGd><)~(e49Fu|nuGDLP+(^(Phke(Py1WDPaB?BgA}I=MOB*q9CUKknIGn!Qf#-"
    "596$RH616I`(ah*KaioB3Nl=C$>3KiE8m;Mqrq;<~8r?vU}bAMFE?YlbMCEjMRe@k;NYxV}(vHlOe*Q5Vd>VGfdDtY!)Ut*<O27<B-"
    "7{8zR`Hgct`G4yPm~qWf&hrj>k(bky??bnnQ7KxEr|{g3|5qoAt<kZQ9A88%u~6#joyis!%6Aab+7ic<`wtrq+hURO{cmf|ZM3rQp2IOaOjG7}4&#<e{"
    "ODafY*OA2AEPC%vDSTfu4mI9ASS>T4K9eNKpyY8Qdzgn(H66o$;V%y-"
    "yFo>u0zc?woV%yPBX)~n}}3f))RR%hxzx_4TW=tHKIq+pE4d#e0xc)aNIZ)En=*pVvSCx{_M_co6qyyT2z2v9js9ySHwPd>TTBBuyGtuMh^CEH({GQeK"
    "-AVFzc<t5aMG0TF8H3gg5n1omBoVc{ZypG4Vw(9%kub94_Esi1Z#3=a+tANYxZvIHJd@snj`v>>2XHP~zC}r!_D--hdp=m*n?u5+5G${I2EGp(!$={5="
    "5+sjm+6_;}*SsOTvCo}j~H@>D{a^!^_(g1A&%L_Ke-L(8$mnRLoA@`xYx{=-G#Xz){u0@lyK4E~w%YSjOO+lAxgCoO%e07@;CAs-"
    "bm{(M#h+zqTZ@{}{r-;=*9J-5s-R>ZgpI-KA*(>czyJbqmlpo(p6R3%SUYfC-"
    "PJTxFbe&+E@-&8`^dOG@S@+0H>c%EO3&w5x1J+gK1d@P_h&wrTbY0G$3&q}zSs>6;K0;+PHS-"
    "dV2$75a;2|s@w>OU87HIwr#-4EyA=RS|XWIG+sQy;eA^}909p5$Lm&k#KSu0>Ohe-ruWEBojNd1Oo18kj}jw}X5XQ&D;!g*od+O}0D*Z^s$Xj9RT7eOG"
    "zrU*C^49GZa#WsR7+Q$z}{w-0gY(FpQdqlV}hY(k@lBBpo}`#JC4lQ_p+TH|3Pc~em19P4Ky`#PCDc3?s`w7X}<SX(uol4swu4o~v<?&%%jFx-qL^bD2"
    "!059TIiZt(gnqpF{3HOMVO+sXz+Q)JB>Ys%>e#VmbxEfH`{eCQcLk}E!kb=BD1~jiPVoQYlJ{veM`oXDCe>1>?dU7VOcRKm8FYD93M+*KdH(*MLh-LK4"
    "XNeoZ?Wkj|(h>T@z-y)c<ngmOU&E<Wwk=78URuxgyMXEB!9eEuK6TBVzp*IF)k9bz;4tfFJ2Azty0o89Wpv)F!;6OWiPTjizOt`5U-"
    "O4o!S@(F>{<&LOdfy!lm3@HSnhfnnvpjLv2GW$FYf;3JW}5UcxB=0Ap^RvW<Bt_RKGdr<jK<)(y*bH0jK8*Fq5ZG^SX}EcXTd_!?NFc<a&uXMPBd9Jmg"
    "ak#J*3!xnu+WWQd43O|4Wc<H@#2YqYSJ@uU*xs-1#k2R@pS@KxbKVnY`N_kMjh<LzxZ{vS2MTIGH62`OfHDZE+8-"
    "xL)6A27p=V^8Gx67uq2<vi8PTjRZ#{JYo4oBt}~nb^+;1^wiB^a0kN!kh1R*&;{TZ`7B^_EX*mOD1pXKg;o?O|pI~d7nJU9?lUm-;E#-"
    "x+y$Z$=3ltTFLQg^hO5dJDQsf9iUgf+w~=<^;39u$2WU486w9=(vMh>YhB2jr7hF&!rulXYpHRWyj5CSBY-"
    "%z^`#5*X@xVbL^Ps~ddNKF4`+R*c_KAji%wlcEa$wh*+%Y_-kZ%0!5Ciyu97#;lQ+YTO4@`x-"
    "!}@m>kRN?ojcMuRi@9aEq&K7W}@*nBR<j3&Y@4Nz?|Nuj!l`EgoCqK&o&~0y*a<^cai?_tZfZ+Yi~du`kg!E&5Oj159}jxXA1UR)uU*ofL*@wK4@CTD="
    "$vPqnmofQ5)I*;e3o$)+2Q_V_j)wKVN6oCi`yT7;;v8GwfVVes>X3kLS)%eCUhG)>zd_KK>(fzfs|@78h)AyHJi#AxD`M|CkwI2OH)6d1=qdn@W6YeLE"
    "~re5jMOcctP(FWA`An<zX;>^Z002d?R7hfx=l`xs`{DD!5=ISW=##;X0+a3*)QmELFLZzo9m5!npH(3-5pNg~RUPmQdfuFUiJ$hIh&ZiXYd-"
    "^6)$We@b2CA||c5E`8|mgkH3%j1Xh+R6~eKE$*|xp*@!R}v9OJd9^ug%c-"
    "#Hmi#xHB7Lf&#LSu?`zq27&Tt6=5SnTLdt*SFZTO9#*Z=o?Zc9B*V2eH>jY#Gm#$K8_a**(s+WxPWqPa_Enp>&&nN!OB#w2inT}bJ1_-"
    "+aydpjieI&2<Z&fP;oAQl_J}sa<abOB@Y-"
    "K~~JFRFeJ*z|P67nGVXbW+#2X(t)Y9wyu>X62Hzd+s$p#NP?d~1;q4s(zWwKfa5Nc`HyaXK@;@mm<`laCgV2c`A>I`a56j9>T>jthx8v^ytYBk}Md<0j"
    "(U)XNdLOTPQ(u7E$pc?~tz+@74znA+%RWyBonlX^V=WFEhtK1Dq-"
    "0$);eSoT4HEBWsee|IYRZOhGYs6urpOKhsaKYw~EKR4WJ5sr@b<jo%fqB)L}#HK=NUH`N&WR}vQ&Tj#0`R5d#e>0CCF*O3eB6avq&#;twVhZ&@=OkeD$"
    "UxjZuEkvH$?xn_1AljJ8{$)BJTxEmsF){WVWj-ti=OkfqJ1j7XBv<+TP&Hc^LMCms7re#r6WtauFbrz0L};NCy6+*J3JeA|1-"
    "k&x`+$q<nR0P*kcc`<YB})6N0{qNMilWVjUi*-"
    "(2q37CrOK*e0sckbOO!{JWjUzuw#u>$GMpXAhiW|1P3NtVkdFWPUE*WSa1nyxEyNu4kY6aGbYC)Pg?1h={%-"
    "4s!hch|z62Q!D2s!}x+ew6%!5%EWEvtqOT?{kCcdUSvR%6cL-rJ6oBbIQrJVHL62%z<^iY<X@h@FLlmE>N?lC8Cdm`b?7AGEP2zFb-SY}xhgOflR6mSM"
    "craWKlmU0;1cpz;mt&NKG(yOdG1HwI`9W|6zAP0A{lP?^{6sgKuhYqmgL7<)JI*GRK>O$dJJqVpdJ147Gl$A&S#%VshG**d-"
    "V|DM4h8jSzsj3U;A`zyjpKSKK-VaxO%&k1$H;!b+4+6%TEnxu|~i%p64Lz@H>6*=A#MF7Z^}4PDFRcdr51QFb{rFsqlYqfMZAYFL9==a=+B7j5V|hKW4"
    "EPYt2&DIUHcZhN;SPOcVV)arHIltInFLC}Mv8?INJU2lk=z{gipJ33)~G@w+%D&y{^V`xlupLL=i?Pu56<jmr0ot*vpdo*b8+N59^od>?ILgQc^S?~=%"
    "w!ODBoZ&unMTfxJ=(s!ox%d3vjpY_Yc%}>@?qETZ9ang^RcZi&-tCWK7Nj8X%QJ1{exSIU#L=26)lYq#48;okAMn6}ci#WfLzdvb76gD5S#;+D?G-"
    "W;fpw>$}%=t|Z#J1ihOp8`yJoVWL>WZ4gt2<N5pv-$K{0<diLmjo5KJgBD<dsbT&hOD-LI)97*mq;;gQ}6&BA$d|Onp5nr;D)TJa#8HI5D54$Ht&flmS"
    "nxiug+3cb$E4f$>aT3TE9h;1_kBk?}_o_YjAFv`j<I3kD==c^=MhG4b#M^-"
    "`vDMNDg{gLN|jhggTNS+^gFlaI3_@q_q~6(hic`m;Lgwjt}J^tviIZK215R-"
    "7~P?^f3T8|vydvFW&FYd}xd^Oj%qhvSsnrESbu=&XF_{)Ks;s_cI!TEp>w_uD1zovY}voz^(7tlyr*dT65Ps6U5nV6jwr-"
    "c4=(d5s+J($yB0d*rwuF{ZDguZp(XqRBBi-"
    "b#8GW0mqgJjrkEHaR|D+OtBrKisynEgr2>o|_Y&CzBt0((nH}FNvPTh!trf5}E&{)QEdW5}&`sqjG@}bAv>@rBBagyzU_2#@8sUd~HB=ort~UqnX5|TG"
    "Syg?NYFQlo7MZcMDtq4{6V4^5C@6`N%nLLcE2D>(cjc)DY8np2L${AbYk6%bp21PJUU<YYydndDafXxF0&Ga|9$+AwP0n#?+O3YcyJ{)nQ$V0LvI^2+r"
    "e0&SPfFIBYQJ;nGXM2hRJ>$I_h88=sEF=_h)O94DZFeDIO@Rwi9~_7jTj7qzIAFQ6&)w-fQdJpI(oR~68|p%!k8k6|4wAwKk>?!GhLADPcInA23i2-"
    "f!r<}{l)x4+02gO+O0sF8pd#F5F2`*U1xdig?_tAWr$zzyQ>3(n15)_bC_56<UnusT;jGI?k;$9aYLnQ|clbHB4M!o-"
    "sAQvTx{OV1@7LcK8Ll@&tL1#}><w4n|>OP=ZQr!2fTS>bN1fZ@z<J8AFw^3*fW%95<Ca6e8!+yGEhF-"
    "O!9KbL!9+Y>9CuERL>YZkB9ns`5DhaYy2)u3KG0e&sz=e$kHZxOxxvH67tt9X8C|K&@}TP*XRDOAMZLLL4JB2?_7y5!{=<csN-l5k9AfO{Je>-g`@%-"
    "du3%YeQasD96Y2{T2kW?dbSo&)mw<|fp`u1!XG?4$1D_&hkyZmge*L3Od=oDt7Xh=^xhxw0Or@%p|FZUol`CLF#gB91($=J8Fbw+&)07R)ih#P}oj?>h"
    "F9bq`+i{91^vZA4G%J#X^he&%^U@#*R68W?cjfOph&?(}P|8Fy{Qly%~yo!6tg4|$*QKGY8ra)22lVz7F(9?L}X9p`%(aiDf%VDamCeE+P+nrHI9Es3n"
    "d6|BR)9~1D-!hjFd;XRn2t;B-<)EyUBR)dF?0WS}dzc`+8<iWr7rS;L%;F)WH^G*SEnde%J|7Km>ei)5iFLl_@I=3PJRb@PrePC^<ByHM+dNu-_S(h(h"
    "fd$FL(aZ!)%b?D$N?%ES-"
    "$#C|MqL&&t{SGD(&IDhHjw#wD_UR_^;E6>nfSNA0an9>lKaVHIlt4Hhe?a$v3!&P{?xZ8dA%E$AAj=xoI>h^H3nR#zrMrV2e5bNuns@IH$l{@am}d0LE"
    "`Ut;!-yG_rK-#cvxVCCB-WEQSY5TPdsCPd<k{Of~8u#w-"
    "?Zm*Exrta5Ht(;|vd6YoJ5No&w%d2UwHW7V|guXgyG>o{n`Ql<ZHjXDvMdb#ML!SIn)Xh0kLF2N<7EUQ1!U583U8K_)Fi>5mptCoSN3@|*DAtaQd+2P<"
    "6v$b3!)nvvrr{eF6+6P~UtjiFuwDl$Hj9Q>3%Ecvq&9OjjVRRa5NG(E&_>DvOpzP1aBGc6J6z&x;CgUPWocpcHlT=A>e5<QO+ue!4*SpU}K$#q@J!2Ps"
    "k$-ZD<en(Juxl!*7T;he_N34*}^PjIyUB^6!5{DL73_z_UEt*qXyknhI;_?02ud$sYQEI;qp~nU2SpRF-cdz;q-"
    "_}(_sGEVfCSWf0|3&g$B<t{ai%^(eYH_EwfO3sw{9M7h7*pOCK`|O+PEw&1>%)t6H6T%1!#xlmEVZycs6v0%)nmpJiIX$w3!bmlV$ORNdJ!*z{{w!pFR"
    "B@1ka0kV16iCG_V0CSqs?ulwbv36Ia!aEa|8$+XM6gNuwLY^scG=(Vt~(e0X@mri>3Dxhw$IU*2K?XBXmX)|Is&fWNlTMKz+BgAvEPo7@8)3XPQBLt2`"
    "ZeKpwFb%?P7TE0FdD;C#&{r{3Ao8)fdBQERk_-!}C2)CmI_pVzw|UK^}2eWY0Op7Ac~igCmfAFn8sQW-JQU*z@idS**n9`uGz7`4C>+v^Fio<Y8pzCD%"
    "l@WTbB=~j4DUqE$U&#7YeN_|Pcd!RkE*xFTqnt9Z6`~~ElmHk4IHCK-<)TPqC4-cj9L5M%qCdXm;WCM24e?4p<e|KD-|93L_V^gLMTM`9~V;xRn{g3B-"
    "y*ccLvxQa|QmDca&SM+qxi|aL_e%h>e`|2|gbEoPM*-"
    "tUIsd=b2Eso<i^7j8%#iMr5)bPTx34x2#+`T_p7{yTuupx7t6e!ieuq6UvVj!_PF7)&^gb&2y%Ou+$JrfgPFX_8S5a?LpEEw+nYnJ~hDoz5Q8z<{_J#7"
    "gOb_ho^G3R&Y8y*@3Rjh^bCgUi`GoUeJnDk_iIy1W#(G-`G^dt~qYibnbHR8|ODq+6UA*2*9zTUTYsXAC+>W-wBI4U79{-"
    "!T@tJzITC@wI%UELCmr}@GO3iYaJZPlO=e%s?`5#%SO4ez)$$76}&9UaWq1<pwtPN3N0<UY+U1B}Q*~YdE%3rrcEU!yNeu^M&x1-"
    "<k{LdE~^crZcsgO@T>-mj6#QbP}`r-974NBcmp_utECO-zoNPDjaVnC)Acl!!Rrw<**zPn5uKD#3Vjo)jLc}zf1hU~i!QFqL)P#Jed>Pq&REkm4d$NqK"
    "T`4*<dz|K~W27w~dSU;CIUn83X_m@^d!=-vW%ocGxPWFA#>@QEhM0E5q;AD3Zwv1ONpC+{@-_1<K&pHP5=|f!N_+p7CV|X1agOV^c-"
    "GKc(Pczo>6#nk!dgLC0!RFsOOd=k7vtAppZhfSE=_mMOe=jZePZaQo|9(C7jG4T%<hDOvH`C%|8vz%j{m+<(v&6$EVkC4^br{`8KyBifH~B7r_?^-"
    "^3a7v55Ij)8VdBZx@4z$uenmr7oams(n4tm|u|6Zd0%v&s{GSmho2$bNe*q6To)GeE0rij9i}DaXwG_-MoXG=*^8EFPtC2ZDxXp3y-"
    "LAqmX+L%P$BV?vX0hRD`9q7T*Hw7P-?RBk3}!wKl#9fs0v!yc1bDH}E72#Wa9m$1Mx&-"
    "tk1`$tI#ZXNWPP?}f90(A!b%G*Hct{@9xku{Hjn2iJ<$uJwrOx>DRGZ{Q!K6B&i<Qo)&WBfS>ZMNu@`lkn*OmI<L#a}VElM1SiDqWax3CCeO!CudD}Vc"
    "i>DfRw-"
    ">N6Tf#4ttVPi7xI3Ed(L!}jDA}K4$tL<!o^R9IGMIi`i*P&YXV%Rx_P}X+k1HF?K)kHQ&m0lG`>>}M@c3@j{~ug1Y`Ya)8j8rI9vUEFE%9^qCRdc(ph1"
    "-@B3xOo-"
    "^sBPj<8Nry)m}C5u38rI7^)JBzJC}&wgp?j_hPDW`Ad0^#rC$>%$QL$5^>ze{U<)FC)U9=eMSQ=qs(ul~)E+w^?G|1r^@%xE<90znJIuAH320n-"
    "$(rmv(3V7d(^KuZa9q0R@w_ctM=h(ATe%c#Y$~u^}49PI`2FPW)sY1kf+sWSwU}iH5$S9y6#1dh$FK*}q@eUrQbaA-"
    "6z_p`ijYSf68AZ>L#Tr}G1ldPW1cy(&EC^<JfCdq_Vz^kgu)R@Wlqq6+hQo)jK`fc2ZyDTI1hi#_jDculEQ^a&`HBR$KhjLK_t_^&Z_0>}A@<LSeB-"
    "0T_;hh2JHp|`gtj_7z@!&w(SZ)ae{F9YKK66fi6`mukLIbTEGHO9nX6CNdqm`eV#AYK{CQz66p;lX}0_Kp+r*@Tj{5;ia5oHdw+wuRP+Un!z7Jwb}%`@"
    ";O}aOs@v<80aM6P4%M`E6{`eVrVi%5mNyK8MlA6)lX#=iWxd#nO)-"
    "lHNPy91w4tWjO#XN@KuD;{0OrG<|3R^~vIi&S+$1g|f_N0OvP(KgTFNkEl@=zu1pUI|?Yz{BGfO9b>;v*%FRR_w=Z@L_o_#`90zB)MdM$hT_o_J<g91k"
    "VqZrM*fZGd|6Z}hsP_mc<n8q4fR#bF=>q@=G3(e3>F%U*sMYoo_8wi=K}kBQnD|ObkpG81{HFp=Nycy>GQTV@Pl@p241&SSi$oypsvUz-wm4+j2ZO%1t"
    "9`_#xnoRbE$IVZIuV!1X*F%RPx;vdB49*@?G~rcWfpv^dF?cC-"
    "R+{`7b2jiL2d^H{TK`TdQ!6{8EX$Gf!I2*xe0J`dLEVNQENupdI5k?C4?cx}s+%OXSp3;jFYy7xTZ0JZP2gin9$YF*-"
    "$sl!>hIv&=R1bGZz6%<X7}YV>7B>Wvn+<o#{@d%C05AxnH6#_OU#Phmc{b36|_d0?xp6&%;AU`ZWg&3+lp@lV<7iSIq>2XCs-"
    "h5GZ7^j#JCtL!@;*qb%TBcJAQK75I{?$Ukepg{E4p&`c!Xu-bTFYPZ$>sbAoHyWFm|E;Q$_mupI^Ora;^9KduN=+?V(JvI%k=H{$PabPNCjxscb=Y@Dz"
    "!Bp9GuHoq<d?urQTTI3hfHq~ZP{Nh$!oQU-%ZQL;Y)8l))AK?h_e^S%Z2Qtd8?}-"
    "D9?a`0uh98e7q&C5kk&Bl!2CAj5tG|*p2xr&$viD^jqHm_1l`zm%cBBIw_O$uIG7{kFAdT`{}>BvR{b*rzHK%x_#C;3j6HzNc$rojQ$~)^JQZF+?wu<q"
    "9P6I(x*Fi0fMCUkr<!5)E5o4T0Ek@ZN}>z$^7gkZaf|yj^I{0tRZfvQ$L%?N6D$u`?Qs@|E>-"
    "@sRuqX59gTYd%W(iXDj2)RUPU#6R?@%X#AhFcQ@x?Q5dWobZBlTl<Yg(?47h`9&>cKJa)KfQJwj3OC2zi=U+zrtNuF>NAtCaWM8Z${`*m%oTVQBTs;)k"
    "?`kpgmI~d8dy&6^Z#@5j8Q~Z<Oov>xQ1ZUlLh{{M_H_s9_KvZ7<Od5VMU3x8UJE2Xm)_us*VbB`rw@HhoVu|WctkxJyPz!At<hlQMFF#^8;`9ZMm7Zo^"
    ">9G&1}ns|fA5hOHOJ+3XQyuS!Xytp>eByeI4=$NP|uThUK}rnP8ao<LfmW1^FF4wt4F-"
    "C^7O_k`sK=VMGT|g&s|2nM6XpRBLH>BnvgB15#~b7V2=fpo3A_z#r>bwP`6T}4|S%$vR-bi$q|E;b;zxFj@imOw5#7cV282}>SbyDF7}v;y?4|p40#=`"
    "(PV@gqlhyDi<l$nJ^E1|=-5(&S&c;Gbmmy7-"
    "$t@uW~O^U_16;ZzNyfh_%h}>ub1<`w}~%ajn|+N_0J0C{R_vVZY}M55{j`oI#|;0)aJZCW}kK>uUS)XwC$`fxu;yj_(}GWH|uuF-"
    ")Pv=zuTP`@JITNj(+VK`>%sjMHqVP@Pocs@|i2BbJ|dMIrI<4+m>3a_^v`6d2KiOYpwKs$An;%9jJxVPZioR&sx@7g)H{cu`syV>5v*Az>2?{!+MTo-S"
    "s?F71>*f+sg#}!~VbWTz;OYEv$|EfkxC-i!gnZ_q!}vuX3j~?zWcsr6zf&j>0=-w6-"
    "|2T8?+6rwLJ>(^<8(!@Q@;`=a#vH<ahU8{8a_qU>wzOpQNUxxf8=mpxoXIUY|<+Dy6s*(<{igOvN*p2YEx8}dG+<8P<ob*c%Px*`G&0`&`}Jt28r{Ty&"
    "{Qfa(+NE~K8x7*17*dNW>yP<0s=eL~zOU~mO^3gl`@~yx85nrss2<o~T)RzaO@18nQcU_Nz{V@Z~^z{wNcRsx4vgDstH6k$djvkd}QzsK2CUC6McdNC0"
    "0x)j24v*-&cMxZ`oThGPe;prD7Nh2B=ucGG!8#jGJZwvSv1~~>Y;LYWkF&&Ej>C$-`-"
    "#6(ex^Tu(pUDi7vN5vA0fR5#lOE=KMFpF^mxyDt5`sdK;Kl%e0)9N4vo<Yqjsy1MBVn7zuVWHb#dPn-t#OGu}Fnk?2Gv)$ve_}Uk6;!C(RNwhp148J-"
    "w9i4fIFXCc5BkoFys^Rbd?aYah?kn8&N#T=A6tE_sOx<)~Kz`MWi#Yhru5Va9Gt_*|k6r;jyr?r-w{wx0tK;-"
    "|yPnj)%EV+P)n)>37@+k4^E2`ik+WZtMNvY5|~tVf^nKIrYQK@N4rwst(0*Sm*2P<pOEHZ0Vj=6Hdgj((Up>Bzj5KM`26&zgFgfR0V2=l8(i5azUHFw8"
    "%RS3G~MNz^Ok%~9mjE|1-@BbNHSu?o#HIX<4J0{hn|CIahIbnt&HU}~!DkA^fPFWASxpT3|O@oG8otv1iIf#Vu&nM4jZ;Nesei-"
    "P6vbgTzR_it)r_+BH{-xBeiI5wZULPuQ_d!ZqA?ld9HQjIp`Y1JiVACF;OQL2+Q>cpr~&0GHN{`3T5_u*D3C^TcZts1X+{*;IE{^PFw>friLBR<X-"
    "agjJajQPo+E^WRr9_>%+vD`t#>fYo(o+q39^-"
    "sHUn0%gjAWwB@E$xd*{X@RhpDmB39kuLB0deH1*VMOt`2U%g=?DMlu%keLGx0=A{_4;Cbajq}dyF34W|G&KpS{fUNB+G-"
    "<7kwnf85nwKsM{Yn8#1!`Bl^dZ)@t{Nq$*Qy_Zb=u!Gl;?^glUQnlzqow<kOc}e{F$$7Ut7l4>V>cCwp+-"
    "5(1B#vDoZ)7)*z=A3|v=jskB!3;Uus{p`|HiLqY^<h-IYGd`yv`1S1v*l<yJ!+o_L(04<O_(R{)oKJ8gDG^(-?%p*5soU5#xD1E0~{d9M|u<m2f-"
    "G0ONQOCx|82*DKHUVyv-XsQCZwEA(_1=W(yJcX$kbRkTL+=OVnN{j`+zsQ2u%Lx{58dk(QDSy^YbVjU;sY?I^PS%<5X^#hZOTwtl}KPByNdsvAJn_V!*"
    "T8^KgF1_?cjz7+D!DEvgFU9^WH-"
    "}o5*Vg1jEL3A`F>brM<og$AiBHE!dw6?6IHrYVv4}y^`@PgNZJD1Q<Gs+Sv<5?02#DnT6;ijKBc5cXRKTuHS{!`Cc_;3z;_>%bN8*8sDD0@m;!qL3)Lk"
    "mpVQ=Eu-LzN)yw^h?A|laCUJv9P>wo^Z6ogMVz%@X`Ch}!(&hG&7ONA$~n9zd!a#}z=)<-bw#Fg`Wvr0Htc<E5Sp@3M<qX(}yiG8=AQzUXn=-"
    "|_d$Fu)OvrqqNNxgY23YQY}c**fg>(t$OCOs42wFT6{*g;09JvmNtX+Pz?l!%HpNGMR=OCdIIRNn7hu+<i(Gje<cdB;uRojVQeVWCmp7vg*yl<z~=hB%"
    ">%vfl42#<wf$O!~BS!gyuf;8D_E#LD`yx1Al47o&VnYDHho-_WtPI_Fwg)O%!$uWv+rq>gV&+{ofQ7bG>qlip^W8!Vy%y_ZJ92J+MaH)s6GvO>x%0iD}"
    "R-#N3sasjVMFU;WlZYvg0iM(*+Hh+VB(nA%A`K|_hCXU_mm-b5|e-"
    "bys<5N*}q!GoR1e~H?tV4VoLA{~s7K8O`4fuOXKwa|F1>(a%>dh{i2uwSt$M`uy$#ao<5|`6g<*f0=$lqFYj;6oPmEWgIAs$qI?~htdw8%*n@TxXYRjL"
    "!t?>)a@WXJ39mV6pnf!E6MIMbKd|4PNx{zmMQzH{j=@8f2qu5mx_jho9f*hBy5PyNw~Ikl1YCtu`=4-"
    "GWf{SWms{Z2o6*FMzYOUS1V<uwrQ3;2)c`F4~ZmwC(o=8d`^H1u>LidxG1)MT-)tP1_{_z-"
    "a>i#W+Y?~(39@;VMt4Bewn+t)*cGkJ6qecv_Wa8g18++Aown-Ddkd0kzYhp#;T%I7d_qE6^JTEuMTWf1c}QNo$0;kdq3hoL;rgPQVp=-"
    "nkfaG!Wspd(O=sKnz(6CWOBu#e_NqE?y??|ns_m%d*k4!33>wD}u?z!q8*t`<;z6m<f1>{U0`P@xBec~-"
    "DKrNTh!15f6&6zAuFT^!~u*2Afrh;i(@e>t9-%wP3Zb@A^bBX+zMv6t8NlGpWvKC-"
    "7>GhB={qh|#*wA6cA*5~+{^r;E$k(_Cb<OXUiAg@n8C9jEZwCIEK*)}NMUkx*{p+M=Q{!8tVF<q|nEgatz>VamnsmGi3#@ol%s8C0ZO49eR?7brTf;p-"
    "A=wI24im!=F)X8D2!w<yaSeI0+nq$DUrXn(_tDfAE-_OY^AAn&8v>3cgz&G;y93F2X{-"
    "1RYLgm+5tY0W#EqU<YPtv!QfYZ4+v|Xyl1LDj8=I1wcuOH)<KjP8nj2?xIM<&TUwS_$J*ESBxIph)QUPtm@Yx0gK|GrUpG)8^a!Hv8Y%y~&9E(Md1Ue5"
    "_d0LYKd^y%c?%GBqLs6(pl4Z*07T0GPU_{4vo{EL`Ge7(OU4o7S2Q9zzOFMZDf3+PxE%Oa|wQY8ay3k6&%FF&U<u^wkMs*BiKMl`x4pa*qZDEsjj^A;J"
    "~2CK?iV;8Y$F?CoU;>P%Ctc&tvQR##Y^0L$jUkh|nzO#FF-"
    "yTWI`~NFvan6-}Ou|jh<XSoYj6AXX50t%G-B456w<wWyldJ6e>0$D~YGwbU$JB3omG$I(HkKhCD*GQ1zZxm~je4cJpt-Wo0^t||)QI1ty^)3_WByq?Ou"
    "M4S5Y~Ud0(swZ$JBCI@k58XzuAxMA5Zqz74ml3ZQf`zLW8{@1-"
    "#?@ejqm=BL3a18i+i99rjibF@ty#$oRdXtluM1Xl!GErn88SKJs(nAm%OHD;|xP7!cl(`qP)VAw7>KPi20Mhh-"
    "}R?okhvqVDR>eyq@!^}a3zohKMD&|56|{;M2yODoo0UVJ1Hw(0O|xqw!YjJ=@-"
    "A+Gp3SHcYX+_XaihOqC(vp#1Nhu%BHp|Lm5Pv4Zl`P#vGSu>UM_9z#|?<UlmDB>TE{idQfM!m8{#(y#&HKG?!Rn|$~QqBRhm38ckisW@;O7^oj?~IID%"
    "KG-wJ!54Z@+INUu$rjsUqkOzS9w1+`Hc%4SIF^!<S1`tAGDwDPPo-vjvtrqT`BJan}^w>s!5KUomf{Kt0y@r;7%(DFU_!7D54)V$Y<78ICXy792ew%vw"
    "|;q<Q4nnGI3bCe}8F}FZLYL;ScArMu@au1$`NH#p1D5@i)u}AL;u7Pik1=n2~j-`(79C+L_Rc`Bw+X>yettUtg!EU~-"
    "`mex*dL=ez{cU;jg0v21u0=8QC;1N+*F_}rd(4rCqWw+uwOl^!#P3s}g$nk}tcBI%9W0l4=^iy!Rw6YS5wysjJ6q4!Ef;B0s5XD8}97veAH(T=!VYhoM"
    "3t}(-0Rm7SmoP#6GC6Bk=S{{x=bU2wS;vIGGTk`UeUckX0fp|DbkDwwEcEo{$%=IbqRf2OQE^aj-"
    "o4oHyd|%A`+@o(@bT}4%^NpBC51c~3|Na=Sf%shE1@m*+h$TL1ykfnLJWed2f4vr$fP`a4EC^C#EU(E?sh<za?a)%unFHxleCXHyP#1kZTLnqq42XAC;"
    "~@K}^#y4yTVQA17^DRmFy^y}73A;D)RO7EuDeq!<5L$sdh8d`s+O$xqNMd$>?3hxx(><gqb&BrhgZZ1>W&Se!C2c~i>LSKzedU5picB4o*(tV^$k|&N<"
    "1{OFB;SLRq8Lj!<UIaGmOanF5;OV`;>LNmHPKrUPGMpqtLFcMlS#T81nTV`qzsO+QIIyHI9!WZrGCRn71<Y@Sh$RVpbbF>^iQ-"
    "7XGs(hsk&BkB<9yprE2X!Y-+i!P>8<+<&My(gA%B$oXkZ3`kMlyDMMb9w(LkumV_zU71sn`FD=&gpU83VdtX8GUjvRL-"
    "~ES;i?qG5>Kp#iTKfiV<FCrNudrJ6oBpzv^aEKz+u+kde&zc<Dp5xxE-#;oqq+?ByWAA9&XG!-0&k7hjy?|-"
    "V2DRN?av>ZDIWH>8fb(T#s?o;nMxFhQ!tTtm}*#@tA*EkHl*N7W2GT<gc&P^Pf1`L8*H9QD1G8{0{kOFXyH4<Zxu=>rl6zfa%ij@%+trzBAzwxHCbAos"
    "2KypOeUU`{{$Pg(P4Ce|OSG0o6EPU;oJagN<>oiF2t&{Go1KK%R>E4V<9<Y4a)zo!1$0&`2KVwR$Pvo7Z`2591^`pC_mjt}E}M^?UDty+z7?XTw?d%6?"
    "LDS?&l9lKCQ_570~5f9mkRWntMyj&JJ5db3sbN$=!^9|g+(t*q}33Lgb5E`tfmKI0%SE>_lW7ydZoMW7rHF;Ke`vqrHN*9N8{JlY;Lo~bckx_?W*@P*f"
    "~f4C27P(M5^t43wwM!AdRcJ_U0ZaE}e&|p4s>4bEjQ+n=99D8>p7$<M*aJG|(2<q(@%K9N%yI7nrqz_soLdUwAA?>3v5tw&537&O~P%Wj-WSzS*N9}oh"
    "!RP8&+SZ5}<3voMpK4Ct-^A<dv^E*GFATUKt#e2p^pt-8F6;0{QYB<jx4d~LU}YHn2#-"
    "(hLY?!a5)23Q*sT+>oxi)A=MQCHzP*`>kHnd`ZAF}Bk6u>Z11z?;#aiV%wW1R;j+NZ6`(%%W%6_IH^gCS@A0#bW={7?7-"
    "jJTSw6gEX=^0LNnJdS0$w!F_A9Zqf#(ia9u!-zNTZNCxJ#s;|a-a4ZG4fw!{~e>N6M8E9;7IT6uTj4De81ctSG|<|L&--eJWn&~)vI33kfX9jG4Xc%Xn"
    "MRw<TmQz>3;5bc1MH7Kd9@7V~y$8f;!We6bC@xLyrVE5$)-"
    "NEU2R*rF~7u#=zCZh_4kzw59H9K>U9|+_oy+1jBV^9Lo@)Wj<e#kJgZ<lHSxq^l}rbMTqE19TG0BGfCdJ%1Xvu_C>AIBK}Ix@tOa#)U(fbg=2)-"
    "fHy}33@2Y)V_l`vXKi&1g7;w^G=BsPre3|zyd`oxHis*r#b^Us_7>5H8pL0De|YF|TR15DFWez73?Ls>pCo-"
    "35QmM<CX7(2$r(WQ5qX{1yfT^S`NNE<8EVv{U+%&9jb+lBzl~AL!Ul_(>%-LTABmr@SrcPB<e_bf4ZOyaUwHgf#z)hS-n!WYp(-"
    "0(o~Fhv*5^IOpEB3AboH^XzBOVB)Fpd;%-"
    "*i7GqB4Z1<JmhQ5@$#)Ei5Pt5LbtaPGAc&FL>jO7CsbU(_GM`O2z_QCcH3VQRcclKXBN>$&lTD2)7|$3<$R_SEr*iO)TWOGo#I<JV>#de0MKBp&@G22D"
    "$p_Js?=`T{Ll+leLLtvw<>-=w|^ODv0<m#nZPODK6iat(b@BI{*ksahCS&xCi<er>$o)vUu5;(&OwE%pT1;OJU4#+?Bil<zmM3~|J#UkcCin!FXi-"
    "{PbbX4=U(6U|y{t@Kf!QS1v#Ilhd)`CECPJnyUn_FYu=ZKOv$ti1PY(a|0;^OgQ$y_KRq5$QwkPVR`n$7b9JR^xRn^TL1Ah&WJWOh&o+2Hc!3q73VP2C"
    "w%GdBLl=0(5pd%=k~h<%ZJwgX~epovTxXHP<2hr2x-r^7n~{$Q!@D#$ftEJ#@s4VCsZp<f(Oxi~Hj6;ISS}sbgJOpMJ#Eri^!68jsnm=bP-"
    "W)Og|!c`A!M)v$9_)Vim~h+6{cvhPlOB;Im9hrWqKm-{+6O%-"
    "4#y~oI!&PbKsqo|0Dj(QB5Enp}8kqde4EcN86jj4FG+kn+y1<aD3Uw@+}%Hw!?*TANJMr3=6xXV7h{)3*jBmM8@wrF5)#)Nzkme=LI)=RYXT1RRgWj{?"
    "~^Kb=E4)=A!^jC6x8};RDv&<v6b373~P@XRj_Tf;4k6x}Si#lWE_|TF3J7r(R#|C%oZzjk8(g!V3?$@mw;(|7n<@jc^wEj42Nb*6?<59TX4pUyLVIkdb"
    "CGTk1kLQfO7;{dK;!rg-"
    "<m*MOxv%ua&(8YbeTEjH4XHDUSDwU&0^(ACO*q^a>d|Ythz_jp!|c5t{QVeHJXWV0;d@#{Gjc_|@?CRW4Qnh?_D631n)z4mkL|2!jaQfC_$k&~FOISKX"
    "bI<&@S&v<b4QAZ50~Ftct(7?pHmShuIq6;nE9tZ?@hg(NuFBi6pP!(4alK?zsNp1aYbJ5D`0yXh72$vdYxE;dH*Wkd6iymhc3Z#{&y2Io+*5>^P3Y6DE"
    "k{8Eavb1Z{H#>CrnWMPSH;Cl)^i29y?*+N@f2<_UI1fJHb}FoKRjlkGT|eEPc(SOVWPpU&KS@UEsMz4I?pZ>QQB1CkH3Y?yc<WZAI@Yt&d8+di?n#+`H"
    "Prubdj6iIW>yLw}}Ad(dX%aIqP+_KKK9ebSmbd_!C5Ih`k-Jkg>pbwYLq=bl(FuPftK0x@<o`-}Wiopmm0Q!V}G#A;E9H5n0GgZx5VtwNpIfb*Mwt2W-"
    "9H{oF;5%$yxP1(PJQv7cN1Uj4Xi+Z3|S?N7*`ZoIVCrS0;@t+An2_h~<%HQDK;_rGkjm5A`BM!5FV_0v0C2XSp|1&!j&Y$&A=ZM%!zW9gv^yPg0J(Y>D"
    "G&6dzR-}6^mdf{@E1c}{Tlu~~+Am4+cdu4T-"
    "(BWo`Z^mt>Y#@H9C2J(PbRF`0o4?}m&#l=Qt&hAodcR5m*YPC&Hl>wef1AHAVk5#+pOiI+m!uMLmW`7^lupZ{Vefz(JJY?)MmKT+6J-"
    "P)g}9=l~vYZn9<Jxl@+`l6%3r`IMb;I%dV)7`UgyCF_1os{wq*=-c4O}cxOCD4Kku)T{U`2-}!Q^f9c}}^^Jz1rU5VLo5m5x(pk@^$=6?oM!-j-"
    "$AlB)<sABl`_ea3oL}m|$}hASR98ef_Rj#$?_c_x_g6g7<en8aOc#*g%C?f$38b&tQms3dF0@8J*7R-u=20aN$J;ofib@{;SaP+avM*9-"
    "6=&4)ljA#Cw=a}<^mu1X_Lt*5nTHBW9*)I0;i)LcYjP}g6d%-ixdYl<k@I<;^Vo#H-"
    "E$%5O*agUZ`wedsm2}RNf0q~CjI8SovrcDPBU6ZsBxBlp~5|B4_h98J`v{z81Sf2#7O30G2;W%IsTp@NRZy&Hq+-"
    "4$M>^t53=rNMTDc~Kpm|A3TQ>%_a?tM^50EZoP<j)3^)=g;yvf(BzfT}`Q77dH8>10pj)VjPdvT?d1NSg?831GjLtKl*j+?Ol%#ioc8pJ2nS@0J2Hda}"
    "QJZ}??j6U%d0G6eB3>uyaqGB%L-b9@$agN(5#C>;V5FaVbX!0L_G7_^k|lc4|8@$}%tq|<7a_2}K9i4DcjCCd<e}0}6Uy`wF<ol`{QzZMh;24V-"
    "lVMad{Me@rF;i8&I#s^a-R1w$B&fx-8<VE9tL?{j?oX_Qub9W&i25h0dl;6b!k@if5dG!lv3`qH6y;qEBDzdZFIruDsucVeQ|>s)M>=-=M`g-"
    "HN+0CpVc@(J~kI{o>xly&3Z#9)MH|*8m{E;Vbm@v^5&ROAAIhvMXf<1`cfLapw6#Kz2W{R0y#+rth*>;CUNgDuOW?fdvZ!DUZ<NN{1$PH^<#ZO`eumzJ"
    "s=Y<i%szO%y}n<4OaBb>j)dHR?B$^BrmvKWv>%wo>*1K;$Rbw>=RL`BFD)7n#+8YwGPKrg8{9`<K?K+mPp?Qvko`8#-"
    "c%YBNi+cvBi#9%RWk6%JVxmMg1z)c=1bwFZ-gOg6H*jJHViP$NigrWSO#FnJ;<&??yR3bBBE2yX1M$EGJx1{76~O)ewbWA|^Q@Qi+%0@xzqk-"
    "+4Ksp|US|FY;Hw|Mn$W?2IPLeFkT0$<c?DeGIERqFC8)QEis@<8xX|j-NCZ_1$dY9;n9MX!a?up)0R9a(x66yBguKm32;jnM{6pNPamu-xHIsXc5nT*-"
    "1Uu@i?)HKJ?-7AdImyU<LUipY?o-<LOFX@H$?Z`pyWij`S(y5o^|nJI`NrTRp7oX~y&6B3`l1Pm{-AjA8A4$-"
    "<{DW(?~uq7&yUg?;^o@mil6VD=s}TC)C^C$k?}C+GRQ?mnUT@!Ej3VdNe5+X3Rdakz9ZF&OJ68qjl%Sh64PZI1IE{YC%t?XcjW4T7qu;YExYr0BF=cN}"
    "1$)X4^7^Cv|=51ZkLhliB?${CMV?u(wD;0QZqAJQv~pHhzh*ufE{?<?~{t_%3zdayMeG32Ejf57^Aw@=<vaPzJ~7-"
    "eCHa@W+j#yVfe^Sg6icJFV2dy{Q&?kN9_v=1q9^(y~vSvV}NZ?nf_kJ50a9(%>ueCmN6TdLvDKofi?sPRtnAqVKksS|xWMj__`^Uzd{mF&kC(q4|_--u"
    "&jcs5FpAD(JN6TkkRl-{u=f2l%{U!+6!0}+SGH=*p8_0-Fr_bZ_3AuYlNh;U|q-e!HSq3`_BxeQvIvO@ZK0V;d`nR#Bensqg50=8eZL3wiO9R8-"
    "I(iaAr=W`Wh+>rL(Y^$udP`uR{*-3KzkaTZSsoSp)oKO@j$DN4*qZJ-sv&|8I6&$NcKd$4yNm)Z4vmJ>cYiu!r9P^#^lgB!*PTbhLpbK7^Y*3BfW@#Yn"
    "hy8wWgrp<tqj0kchsZDASkGy!!)Cnh8#QAQykCzH`nz({d$;sELnR)m6N-"
    "LSb$ISfKawr)lirPZU!x!b73b@)%T+`geaiSZz!~z$AB$x4@Gu~a`f9iIJs|NvSo*Hel_#5Sz~wq3?AUkt<O?6-|HLDys5FVbBZcQ7-_>TkH0Q^gI-"
    "$tjfC0WDzS6hU<au&xOZ*#!(gXBZ@m;_b#{FN&-"
    "#gc<8HL1u^f*C3MMy&f9$$%iKI&pR!rB;dq=|^K)B}N^*z=Oli0h29^UTC`5jRR(;P_zWR#uS>Rw(NWg;LvgQQn*S*V-8xCI8;6|A9(h^aysr-"
    "g0vMCHr=V@_c7P3pWgItn7b1lk=zSBfol<Gfsue@gu|m>6?>!tT{`&Ftq%~1{0pCOV)#v)_M46wZu(+NFHdw-kNH3U<&hCCs)aZ@2&lj)k}xm9b(D%q%"
    "GNd3&}~F{G!n$!ia|7MRer#w&69Lp(mN$qdFSRH(|9=jWEu6=e5dyjuUNAHAwE$GxWZfl>6JK&9=~oD*LXwO1_BsU|l)2N`s|^8By;=4C6db<L^!+{$x"
    "!I#^E#ra_5Lx?Mv)ne@)|f{uIZ;xvB{-uZw8GzP4xX{EMXfuVbNYVvFh3Shp(HAM5-"
    "q|E_kuZ5D|?9Y_#rY#}eaQ`YbDPjJGX?aF;l#;+^!!7ZJzRM}s(0p}`B@gsq`PB^_=dG5yF+pRpuY4y?>1&SUs?32e|;w2mL_|3|GyX?`P%J<zZMMs#F"
    "eKSY9FuxpQ2lC**KO188KI@WoGH1|_52j9@L>;mEXDHf<MwBiRF`Bx^g1XU(J}!A{08S0mV>k8xheT<;a&iv&x4nB9iP3;iX)QD9RsqLZm3`{+qzYC!m"
    "|#0eM5Oe5gmWfPTZrwO;cL7#o~#g&!+F_#i}_~_8TvKF<y33@S|Z|)w4XaQ=(Dk`tLz%+*WHYA6GU8(WnFS!;)c*8?Tti-bL96{>{If7d*aDpdgii^lC"
    "W@!8Mc2#ye9W<Q|>SAp^mLPSh+t*9_gy^NViK4$g!5=<(RhtO5QrXb;K&=xn~IDU6pwBZAWyxtL%$y<oG!Up`80ok5=Mh3wtEoq#mZeYQg?$M(?UFH5@"
    "sKc4$oQnZ-P8WxN4vbp5clcqiDRfZnD*^>aDKEvSLcbj`-1HP%?TL;e5U7pif_0T0~ezFSGWdd*rfOk%xJ9Cw>-M2A6Y<P!Ifa4bWJ1-"
    "G@~7)Jb=N1tM-"
    "&trN1k(|fnz9CpzSC1S1YCIuMHKpdrmDb~$7L5K6b!h&FdZ`(&iT$#Vdc3(^ApUXC;qn&h_VMg59$$m{tIw`7*sazeo%pt!{CJ&l<2w43jPVF8Z;Rm%)"
    "bL<zsKN{HmpZ`^DUZJe^I1j7r%MYb#3=XkE3o!^DtMK8ggovl$L*;HF7f!CoU?0tM&ehT9q#|9##Qo&?@^h*I(Hn3?~814@wOWG=ufM$P6iXVT<hfGag"
    "a3@Oy_tw?kv`4Fm;Y)S~?~@Fk%$*aFsq}D(f&RkM+_m8UttO(S3o4pVId#^lR?aDc%;LsF$jPWd-"
    "^R;`rLf)WhtLwGS$y&kr4v<3&u2W8Pm%_bh?B$;lW>pYoush{DQ1>?>)FZDRM=G<ZEXAgPat7mPbGzL4=zlhW|xy#XD%idew&STg>D{4(`x5*~~<phmU"
    "`ANr9uJpUx}#qk4in15c61wJCi^Y~GWPv!3(nOP0v7a8y=MSfn^nLLu-"
    "Mba=i=sSQqocQKQUFS&u(v7~g;;&{XsA9(4?IPOIFDJj3_T*$<m$t&%X^bBdVWc*ks`SxFlK0+H^1S@tKJ-wGGrHI-"
    "ev^G+VD4{`gLl=A!i-_|IH4(xHsrM^X+1sG#FTy%BgG~(T(5>d^=BJu0TX@4`q_aPU^Jpvk{Y$hzl(@VMHBh|{XuB-MUS8toEJBF9lRp?&@+RQFn5><L"
    "tWJ9!oEBGue=W8!M62b@!gC^71gk!c9^X^r%r2W3&&b=|E?l$eOG+0*wq%BQssCh)>XKKZ|q0sqzsICX2w&V|2v_s9`Ud)ebt^CVJPfmMDHsi_S4s_WK"
    "G8|m;6vVdX=$;na77SZ=IC9g&qI@-"
    "}mdnnhjOfNj^Wy0b?f1@fGa#pUS=i>B}7vvQCaaVtk(xzkA3L<u)lgR$wo(UxrB<?oV$EugZ=H+@gkwIh&$9uMgYn2#vxEqj;Q)(s!y84mdwk+5ejJmC"
    "0Ux${a;b7>>r@Y%!^g8tuGTXFN|g={uFc>hODR#$h`(HV}WVFc0q3tA5VmxLC%BL-ZY=SfBrLoR6vVH}(s`fd~W2Q~y_!=9hlzD{*^tuP9iJG-"
    "6$W2upw9%VmyT+HbCRDjMB0LBCE!6Vhi9`*9j|@{A6hQT~-RHXIR=L>$mCUWf6ux7y*<XKVCgeI`*i)+eT`$uEz-WMe{xHTrB8F`qbak8|%MtxIW_4)1"
    "kl__F?c*vsQ=wTd<B)*0RF+F}{~MQ`#&dF8#nTjYhj(Q<yIJwKN!y5i(K2h_Bc<4vrkbvLME$<M{L$Dv(;9WJd_!<Xj|V?33$^>J-"
    ")I6bpPgAHnQBL4T{-0WW~?Z=*vx0be;$~ms=Nj=7SdCmGc*Qfzv)Hb-jT#bLpYv(!lrCH}a>ZYU2OEWIdRijBH^Fx0zjXn0ZNdkIoF`?o(H5S>hJ~%Jl"
    "{9W&wbFpR;H4uIMM)E@Go%9|<IiCx|@GjYa%?;Jq+K6?<cnEReZSx@1tf@x__KS5B9>15^*O@qQDG*M7bQtKa#x%wgcwJ}67kX_F4xiQGoTx@R`JxkR>"
    "pAh~OSwSA=jx!nAz~JN$A02NI`u-"
    "`*=5kZx&{~N6OVBod$X=uGyg8F`(o8z8!VuolJ<9Q&g*)%nPW}tf%*1!=qHp$IP>;Y8V~XR#q#c$anly}f2%QyIPhEXUvpM<!qY5Uc)wF4hPvW7$GLzw"
    "(C$kQOk~gRqAvAiJ*TtI1DQ{YW-ZZ2Zv%UJ#g74i1NmY$`}$hlY_yzVMi*+Pan$ooFLG{s0&}Y-"
    "VjOkRXljM2nar=W&Ia+>woMea4Amo$x@83U^bq}TQ|UdvSz%Z=TZe#j>Kq>b;34acJbZUyWh`x`hg-ggmeez~pA*yRU*;@M#+|hWJR2?ID0!+e<0IOz{"
    "s*L^OR^Es^fmVEqo2>@@5IYRrQ>}aBZQeEro;e2j6b9w>^~s|U49v`a)?;6Z_6#l`*2<&swBc`ssX2(ipUF>c`9GJKTx(d9M>4pipT$EKc>G18g}LU?9"
    "0QsDJINcE#e6IqBi+r67%rOvORVMnelE9eLdq2<clY)pBC}?@ThObiNhl5u@^HGeYGja5%G$T{>xrmqCC&KrEx;&4>`Vw*L#bawA==A;e`O)Y;DB%c4}"
    "k=Q(6)~cP|0%3<^agfyzWp9~MNOBZjUe4s1(|K-gacEWU}DM2%^q_=4@(HgJiO&wr9y@~SePP_->e*O23VrM1JA=L-"
    "Wi+hR*2Io?4_9%Ma#;NO)A8j2imTbT0Hm`Q%gV%-)~=j5f-"
    "gFkU;e2g0B=}(`qPhZjV9R3}Sn2si_Dy4=O`?M$f%aU;q?+_%HG2uMp&nrm#S<+4}2DHXVw9uGw%T<jx>|bebt*;#G>n1s<eb@$Lso&D6#j+KiUDw$jm"
    "wL$KEIuQ@XI`>C;Uar%>o3Qh7$2kf?$}56=r&xAi#*SK<@o%o_Gqs>m#@X^D(3Ib=I?I%+X`M4ZLz<V8mEa%2gp-"
    ")%>U(IDcBlo4IBFIt*n!9=4TJ_XV#5)s3OdW<8?{;rzP{Vg?{Stk#PJIX2f&)i1cJ~2>J9eb;WRXC~kEyVCrBI!#TfOi2t2<d}Nz=boDaf-"
    "9ix+sQ>#*+)Ui=Q<Q;D4a}&+dYeFAxO5Xpq%Uh-vkL~Ye{n&?kpIKfRfkovz1@Xp=E&(FMY^TCL_q{WL{w0~#18E4?(XjH?(S~AcIU;`YxlQ?InVR`{<"
    "_b79ro<~##-"
    "y0GqX4Gb|CFbI3IOzS1e%c^T^AGXg9G=FY@{q%R8ZB6C31T5RpebxkOCwGnqNPp(%1cnsIBN2p9IVA)J4mh3tb;YhmhdGnV=)QI`5Ph3l=%+}fb(#{St"
    "B(;F+HmCi+#@`3!g>r)mAV{PF0uZSni&t}A>db6p6j-}w$XcP2|V^NHxUzt0c&&c=+ICs#9D>aokQBi(QQk#9y-"
    "Ue}qcws;abF?A*<^JTO6C;?bX_1Kfs7E{U)Gq43@yyNo)Ui8yg&^&P4z@u`44{s@NDj}X&L7<)0F|F>(auJRJLE@y;@f=AGcmjwMr4{1NS%<-"
    "m374WT1p(dS;rT`6b&NoF<$)pFwRGMj^c2_7mFuo@QMD6qmCRz94@X$zWT=xwwp9q_m5a|e)ewi*LU(s@IimPKdM2Kiy{K0eJhwh5yV9?wj4r_YtZzVh"
    "|^r>wVl!#s$5rfH#`eeBdfCrE%Tu+>$*Ja`rLym==9czs?(IHB%SZV`ifz_JlvjxNwdrte~x;d_RIA1M&fy2eQo%UHKW~WCDxEP_mO84*zZ5o=Av>1Gb"
    "SBSVkF}-p7R;UJjrvefm+K=xUxoxK-#M^j)9Uk-Ch$FN1C9XtHd(KEB-"
    "8fM4c0xmw^t|jp){lHN$>DIva0q5xM7V0;0Y2aQYx(RHl5sc3bk{ji%)haY~09)FH|o@*Mk!fwiT6C>lGRbtv6l#7M?hM_$m<&#v1NQ2T@qLkBZ|i05^"
    "mlP{XHUtFJtZMXI4H&?_4&Zi>n+uKO*xn*K=M?=Z;A8EhA2huxLZCTfGS@2qHz>bX~RJ2c&ev_kpky~~A-"
    "D^POl_C};$nQT`wPYV(E**c#FJDHBU`vfd<Q;GF!piAca9v`+YVuJFUhm0q9C>{MZ90zS8jwCr#D{QdEA}a4IG?g9>99U)z>R6t!?f3y`V{)dZAb<VJT"
    "{;=`Rf9ErYn~BzWlz}BiizO!}$sO3d{R-gF8B)>j~L@jQvPy#`hX~1Fd5`-"
    "p81bP5gIbo+L0{>8#&_IT4tyHo)|R{THvdVGmh~dG0f|685b%!cL{cB;uPl<M^8WMN`*YtPD3}Q7HA5^v=G8k6JCUMNUo2`$**c6wB{2R&jQi-"
    "N5pDBH^Zx`DviOIy9g*9D3Pc7kh>^<gNR>r(>d|HyzO}!Wz~e1o$rJ_0oQ@tnWCT6KsZAV{x&7UW+7sPfn{OX>unlI%17V{|ShrZg!d`zlUG&$PM8tEq"
    "-njq2u5Ca6TRQZ?nDyV<P$5{TKO~y5g&JJ_U8}3D({59eN!2#C%I-"
    "k4r3XJ572=GYAD^4T!g<E@Ho#L_G1G1$?^@0Y#w^U#aKoc$43`uAlRPPd=%L`eTBvH+3WR^BVd&L&CAGDKJq#`x=zcF}_oXhgPg(w_XYGA7aEC;>KN$-"
    "=8=monP0ZaSVnpGvE{TgY=$~GySh5UiI4*iB3TV%wS%XjRKOXxzmXM_irU&l*Wkr%#&T@$J2~~+eqfXfE1*jH)7X%5htk|UlF(85+A-"
    "UY>mri8+;_5hcmw6#Q%z1SLB%i@V;+@8l2Bcv%J6AB+g&mZZsBFutnR4)K}!GBI>{{()-"
    ">*G|I*rVKq*~Vrs^hT$3H^Y*5`Y_<FW9>sr7X;^YYWveN+G`-"
    "T@hKWMS(tcW$lk*SPXvp(cCe>c=_r@^10#6x0aZQ^jDbpG;*GB~%hG)g53&=FIwvR61vom&v)gpI|eajLd}tBkGdcIh`OVBQudEOoTTgH~+rn42GolN*"
    "@V`>K~k=PYY{=`G+7c_@=SQb7H6@K;%k8eol#(E=JV-"
    "%1gOlgZm3uQ+o|Yc!cH;Aai~KjXE9dcKu^7`jf<p|wI;qDyQU_hJ6zPTs^nRE5yf0`_rTne@*b>f)BY-"
    "C*5Jg`p1woMydMJ|Ul<^(oyQZc|lIeiBN4Gu%u2KVi(Hx9+gsuEJD>i0Q-"
    "GkMX{K?BgGqywEd5jnaW4c60uZiO(W=xOi?^%vjI$b`|hd;*qWL_Z^yCU#MfXNcbYc#5})3-"
    "|c7rc&>jmvN!AD!u}<xiu~St8Rk#v{qcwmH{b$!d2(gqw)C8NFvla9sv8=hrM)xlMbtURInL3-"
    "L|mw4z_ffN9<V=gJSo4Y_bj&p94i~pkafK|9eB@t{>%8+yPkmh8}*2HQzEM-uO<GkB|i7Fip7y_I+(AB2(2pr9^o&2e;<&D6JK;V%6|4!E^+uC;6mQ^="
    "okg1w+??=izV-$$4KwbNawu_h(XRU9j5gVQIEJ#>k)e?>iLlLWRzd4FWI~EE&cc(`Qi}q_R!QS&^9vQ?K%-(Bjxk^ZZN(*%+)a%Mhv5#dCGWPB<8-"
    "A_MeQdg`!;A4~lq^1sLxGE9y)8SHz*8pvT{O><jA4`&6j&iA$rxaJ-HVbIP+{D<oH6mNbIYH-un5`D=d#5wX+_W!cx)pdJXf3xr>99g1p;CH<o_wPu$F"
    "5<Ul_%1RyT7m7GYJaK1#beMhUu~l`DvA_hcNF^?Fp4*AVPP9MHt%do4CKwZxSjBvZW4#<=zjMBV8~j`~@NOXD5%F!@0eS!A;J!@Wy*ljODf3Y#`Mm{sy"
    "k_SxyeTpu@-6-"
    "EE}etRHBr|cZdw7m<4hP*R*ARNhjSTs2V&^Oxmoa^V=n0#q_dJfkn<PwzQ0(kcKcN5M_x8CZ%0aJCzm0X?J0xj$4cYt6$SEHcV5IdUygTeo)gyJDUF*`"
    "6xc<sTTVWbo?HB`P!{tP);PIHf%U|%V9xV|R4-;1h%c-"
    "$v!eh5@h+42oX$MTU+ap1$0}TSE8tCkxlb8O|If{GN6tt!D)@?c%zn&~TqK=u)<o%v>RB3mV1E}%o{lAUBrxuC9DUgTY0+tth@R{}j}d2D5U-"
    "+=%i#?3&5`;fn|Z#Q^*D$4cAD{h*H4e<+sWhXe;+abtymW`0wYoBx*qkZ&zCblpE2(C?E7jV5?jaWF^PG*fb%o+`uDuQQ@22@4bq`WoCq80|3kd~d}r$"
    "A&*9kBNsr!5MC^}|&!KqMmpy-a0;=s~pF2`SH2J8abY9YE_EVp#<Mk>N22;;`rQZ)xU%g<T{^xiLeE4OC-"
    "5IWzc(hb{zgl{KGcpDHz8f))I%F)z*hF4#VPqUyIw0nV62pfn(3vrOMn7cgc%Q;Dm=#wV9@Q0iPuzAU7EhzzFZ-el3g(tZ_45kcC6>QtEN@1zuZ?iV--"
    "_07?Io0)o%o%8lYSR^-`EMoYncze3LM}V=Xei0J?r8>@;>vyGh2b*jMX{j;SB@(-"
    "6>_UNn?$;E(&~RUis2*!==v|syd@iqBZ=OKbM(T%h<~WQg?XSyI@T_YYgA0Koibo4taAqd2j0#7fhOI4cpTSY$r!GVV^aYB~dlm74<u-"
    ";Idl4QtIFr^s|kF^b)WuE?%>S)h7iiFwYm#Z^vl&33Y?-"
    "zt)IR2v|wJyhXb^^R|6cH&l6VjV*cs4au(|tW!J2A?AP^p8T~&GdBUprpxETn8{xiGTc$$M}?IY1x(&UoTVQ~TNv}C40e7kjRi^t^2isRxUSpO?-"
    "iNPaidiDeMZ1b*4%DsKcDi_`!ilRQA3RvX(BrI1{RQ~re<)xdwsFeT7&Vd$7k$E?0Ef(TKv1V<shumpgr}*W8#Jn?X#%2lUoHq`%Hsu>ZmKk>yE7VLgL"
    "k!O2PQIofdW{MQ9oU11?D4rW4nb!jQFJi{87$l5;-lP(SP=&Q-h+hoO^o2xlJtB%UuJjw$%x=!_Vome!$MTh<Zr`Y&-cm-"
    "uu(Dg@m%T6A$BE)fSu-<S6TT)8C-;sh;@M~mo0{2waq55RfG-HyT4wmQtNE5eucxRUm(ysw*kBCc1~W5+-"
    "d%`3{^l}sQ%KA)9>Teb$|%@eVacz&Go36lD<J#k1Qo^K}J4xd5Y&z$Z@et6Z#9V=4QSUpXEo|t4$4cd=7Brd)T{wZgT!Os*p#yn|Dz0q0XldEo6<fVps"
    "l7LmTdl7&BiDlk)amCR(YD`%#z?1n<hxlBMy7$5uSH#UzBl4tx6C7X4p$O@Bh8>RZOR+}x@dBcW*&ms^y~qPYT+3qMUgrN^0iXHzD(aaf)LzA%0&ruC9"
    "%Y`2=)wMaH}j_par@bbc-WXsc;>8xJ@dZoN%_o)gN?{*);35ZUr*$?3&{Ij*$=K=5Q_@sOqlM(zK{4~NA02@?iHuFqT_B2N-"
    "q;}jpHt5Uinimy`~r$nP!dNRs#N|-`cW{s&l-"
    "Bot<%d0q4*BcONUC|67eblT^bMb)!`fYYJGxaT{}<3z(lP;DLIz)UdrGU`%`FAJ=@Hak0zvLQG8!uKI}B!+dxq{r=9rY<9LEW{%XNX&Vtm#Ls7pS1scD"
    "rH19;P=z{es)*L)gZT%gy_BeXrvxLTBK@;l#AWJ%k@SP~y>Ca4Xx!*-"
    "z=}&^N&nlE_%M+3^)Xb!i~ZEk4|v~T)&%P`n{^gGISF+$jJU8*eqZ<Y4QgEasdV{73|VTxz{w)i%(wE_f%;vT*O#Kv@u(gu>XyJLd0%>w>s~!R5es)2a"
    "EE&SIqNx*^)i5Z?9jV3MBg@|=?W2hsq^Mh=Tsff+-+BgfQe@GJS`&9g?Y@p9Z8&zsM-"
    "(%YMD`$ywis``(46+j{jtK5^g6M@x7~vbaHGpVuzmn!LPoKD0o;JXOaZ8A%3=&_S5rYuB17WkE~%174Vik@tHU;@z^HzQ}<t~aC|rOmiFGv5wU=MQh9e"
    "YSgpp0KLUb?w~@r>6wcRQ<${Fi*08&zfFpA@i}{d9d{eA(LHK@a%zmQ49CFJ_=6YkU<8fP8tS5fn|EfS4)<rq`zae#dpUQ4nZ&aZ`EuaVSxe{@>nD$~T"
    "ciatAVYegeh4U#zKi_l%wq0<?pj;I^0tDP7U!JG^7yFg^$9(Ydn;MM`tc!8f;q=eHtS`e051byL!q+qbof+?P^uN*rIGN@N$I&WuV1AaOeJt%Zw7(kci"
    "9QolaH=Pu?ikiI$LY%aud>$@4;N79v=eZT*Dqu~c(MKtxAjCY?YbU9$vex3>4)CrQ;)4>5jxHq&DSbWUHT4|cz!&Y`5f;9-"
    "`i^ROkv!a|K)i7KK37D(gKjYMuR!i8As}>vZu%u)S23xAWRC>q7C!8GIdNJ+MiH|P6!D>(s(V_vcIb%?ZeJ-"
    "{z%_#Tn<J5TUtE3BcdvCb0hD&inu<qXB5Vb(xK`r5xJSb_<v;Gn3J4<Q-"
    "OLkrQJ3|+BccKFYlxJClTAd^!R>U#31$;R>WI3=EH!$@u)dY$C?r`mwM?N?K<+nf(4OS|5S@w)kN&zdTJ1d=kWgJ<708Umkt$LQEyP^l%w8$!}@lpSpo"
    "Gk^++5hq6*_TSK3FK^SraHG9K9*FoF1EVBaxFI(JXnZ=+%|KC!>pKwcZb`z_-2L&-"
    "1A9pjL<PluC9BJA0>lww|uik12#ALP4huy`@^nYyBg8pW1)^=^wZhSpMHM}&aBtlRJGql2i|mNYGg{{L!FcvZl-BKg~r&g`E~R4s?CZM1m#UO)r-"
    "Yai{Sh&MZ{1;A&W7R}hl&EkE}6Q2{wUmZGo;XKD_UR{J8>$5U(DTDqSf7=%a9JCnMScH|NONgtl*Hfp9S?~?9!KWHZG@)+TN?snnQi@|5(np)|oO=5%^"
    "Up~8cGl#OsPfo1(gb}OB|cJ*jwJW^lFwBIt{CX8LD|kCJXl9|9KUUd<WFZbdP@DAA|PuNZOoq+(Tv$lS8VE_LU>d5tv%U~5hqi~tE+o>U}9@E=Is)2t*"
    "!ii%Ngcj%>!PjmZ!lnt%%nJ())IdO>6ep9{y;yT#Jc0B9<^N2I9jz)}>}fAddCXp<jOyHRwlAULV$1`W+_(5A*c6vQk6`;_4A%RwwHE>a*hT^Nj()*Vr"
    "G0k^?2ZLELN6BN2BR7_sCQbu97g0P$ff>#yO9RGhalVc>lc->HWWN&8E4eYYH|BA5M9<~b3!iRX7&zpteIs<$M=5M{*tg(Bt?KdZCerG2En*Q<c>Ck-e"
    "jK2IP%Ut<2JurFS&tb)I1jc{TfO8Y&Pm%alc9;E%3hX#>mn266ciATlKJ}|uRuZ4LSQpSwK?6VpYABQpj57J&Zvng&5F(YA%h!t_n9nPl<`O7>z0ekis"
    "aFqQ>9p-"
    "@zIW?aB_w;UMv3ZONOJ@m~HG&+qpW29ZFzb;A2DDK_wMxJ!*5_*ats~>GYfb=;G}ofES;XQVtO;UJKkBTwZ{BDds)f!|L>v0+F|V&fzfSAqfw9+B5b6^"
    "5h?C8U59R5nk8M1Va7u;01p>~FlKHWb#D^sRX-`x*-"
    "$I}O2lmtdPlBa=jJ@C~s*&AIKu`K>wRAo#^KFaH8(uzYn1>1oW?!`amfXk9?BD}kp&I4p2qovmjv=o}d_J#FIKE%hVk!A+DX$+v|M%oL6*l=Gmiks$B4"
    "91;P480Q^1f&9`oOcA8i$q(7(%|UPy12w-TTu%xJvz&PCfsc@y_Bnt!Y<|_Cd#ZHP&np@Q%Ful=C^yI3HT+jUxs%>MRoQYCds;{nu93U-"
    "=s@xZtZoV|xMC)RE)q9}RJA>(T%`bk>%fC+<rB4wasJ(Ep=Khaj!B7Q=|o>BQdv`Z<)kw7xnF)sAX0gZOMkebb5duqMFeo8c%ftHYT`B1Tf*w&nO6T1("
    "G3VzBP64i|oj7*Bot-"
    "(_+L`_uL7DxlIOJ$jmz2qaFek@mCdKunpKjK+WUc&1PylX)|Te%Qyp@@RfCQfBEP{v%>7@jQU|e3p1yt9%7?|Et54In)ot=g!Q}?OfNXkXZa-"
    "o%U%bqBe0Yh<5k-(s%j^XlbKIGuFjG>YNu(n4iSw##t#CDH!0)`qj{${+xP=`Iz)21;0cCa%PGc%zW+oFR+gH-"
    "L*9dr<dqarJIOM#yw5KbFTZ_$rv;qufx(H<|p}aEpc@m>*Lm5KiqAj!Dr(3$(izdAkz5;u?yWXB2bOoJONJR1HXry3-"
    "M}RyD;S4*24K6^$h*Kfc0oAy^pjz7}eA|<beHAJ^35ub?gJCg@xj3whn*wV#$8Hsl0C>d8~4~5PV;vL%r_gy9V;PSLGPr2Fh^ejt&P#iAZEWaGLlwmpZ"
    "84iWGSDH6t@jiOGzYE%nt8>bGN0{ZT|eSTnvWh^uRf<u%D0u6x~(pR2*o+RT$7^6!}QsRuTtyWziQDx8=h;BHsyAL8>+&VP6*PfT2^E;%1+8}Yq|v~LX"
    "YHS@j~7HBkReNlj>F|mz#xQ6S9dgzP8pEQUg4!q=gc1ro#T-vL)9Qv-"
    "+!kYY;On$1*aRlboXRlz4U!=ni@<mYspeEi14Ul*?3TfpGsJd1}KKtZltaoXj)Y6Sp@bIn?y@<mNiF@mrhcl?lzHQ6KuIDCPej{SY2d>ldU2SAVTNGRN"
    "-w>&jq`ehvMoZ@-"
    "RYOKw6Dpn+F_k)8C!JSL{O~$n8N=roafEr*ob_3Xc$+YY7_l=A9S#{$pSXRO`fMBRM)HOCj~Z~^W5TXIj0<&MoYddZUUzLSJU*MCVSoCadh|H+|1s@zu"
    "Gh!N0y7Q}hfDd&`{+u%@M&5W)HO}mI7UP?>*y-"
    ";VHS1Sfu>RT^+S(I$s*jjel0P&E9>S`3wJ2|G>AROxDW?cv3@HP$Gn0AaIUr%E#6Yc)@2^kZ?*W}es_b>@sXAwLL6YcCQuKUm=6(lfw)&&hkNXI+&B+g"
    ";!k__D`S$pv2dmuO~$a#BJO=74j*P+{JYK@f%DaP#QwA=?PWQ>fjZ~QZ*SC|rbaUPq6zs-"
    "$+%E*!SSmPfklnm%gFo8!(j5GAMN+n_~PRMHEv1gT6X8WrE^bWsOxh45O7)zE9!q~zf0+xlYdwT8LeWmXsiwn6GVJwe!eHp?4^EiN%Du`k{Vyw&+5o$6"
    "Brkf_Ju9{VRKH6^XCP`aQu3-"
    "o7fN4N$^Jj?|baB02|hOI>)I(9zHYH4=Wd_p=Ca=;kwRoJ^|#ZD;<1c*H(>>^z(A|G}EZlT9N0s6}Thgq6+uA2$)ZQu4kR*5y$Gy3qtc;EnZAvo)C|Mn"
    "a?xpNc*>kV0S$&49u(JT+eCR&8(L*Zs9n2UJKO`5mD5UL+Sq)&7|{{qVZ}Hb<1-"
    "RN9c!N9AA1~vbjtmJcsHLs8k|@b?YJR=TDt;wpJw^_cfrJNr@cd^HlnIFRxeMNJii9dgOi;QH%Qa4RNU{b!FtFG~9MGU@-"
    "elA8DUo=AjdLqoZF%lnU3Q6Z8BiajM}X`TML#6XLMwk`7&|quvl_l#I)I;@|At3aB$ikA=kZUX^5AeL{O(F&Vdy>5(*1#6<RWr-"
    ";K<CH?$QCG_}LkDb)p4#ej&FUd*ur2Wa0VO+-"
    "Ro3p<lZn$t92`9!TN5QqC4n`w!fOzqeb=sIX(L5^v{Z48SwL*ZJdd{7G=ousJMec_xbv2kiR=^j=C64p#!g|{NAQTU`YEkbW0mB)WZLHH)T*vVCLHM;x"
    "3(Z{t1&qrluD3S%%CId2R-QU2{u59+Up}A0#5mqig<<PO9a@s7oH<V?`nibyNisxXwW!BC_9LsQv#obhe-clcKk>y_2R&ME;(D34O}5CtJ2W}%j>3Ax&"
    "j{*v=4~AD{36%eXQ>A+Y-Qdu-@ennlyMgr=Mfp6c-c~omVE?brTo3>H~O_6b^hbQYAAcN9?3i3ugd#)w~zKm4E2A_V&;DibDVZ>-nT~mK-itu;<h*YQ{"
    "HdUW%+ku-N9hI>!L$awum3)nV-D=9(CL|;&9XrJ@nG=%hWw%*>6hwwAHR(5i#LLJm$J)u^w(v&xBL24ZEC<>p>>eqAq<-"
    "oK!Mi+laR|)w19;)&vLY5F>GK2JPR6kr#M<FDDZUcFX&i<+4sE4Q1ZCRwXW*aC|+-A#NXH-hQY3bX*pG6q#_4ywHQ!-"
    "((zfIG+urs^M}K6Lu~Tae};XgK<}I{9f*KMzIOgmy2*BkBnw~f3v<%1y;ooI}`p65iy*3^NBd%$M~F9CSu=R18(Ju=)!)X6xSO>|4e)wfzaK0T=f_6jC"
    "ol85H&G}elgw$N2{?fyC5KmaoKf_Sezm4i|mVD{WN$+9oUX_b%%IZK;7ZjEeMx-Yw>{j_LP0!9_nfZ^Jws9U$pwD!Q7j~bM|pojEl5Ckj9HTcBmR}+X$"
    "FIyZMT=K?-"
    "$gfj1@&puH>gGxcXn=0hy)k&ZriP8^ON!alLTtSdm=YvAC6?PJxrJ5qp(_0pT`a;3f1RUgbH?)4#_4<&BbWgL&OA8Opf7Y|kvSLZXYs>|;`II|8OO^Je"
    "k2>GI;h-"
    "=J`cJ!O{p4!6!ez4u6#<R@=j?*5+IIbkF7VP)K`t522vX86ROM0%&Tnq=A4e~?zMQW7YBT)Adx0w&N()mTteKBDKd0{W{miFbe=h5!x<A>1`)cDQ1Xu>"
    "|CjP$-J^XPj`4@_04(U5)+8K*Z~*9P|Yu2TZgZmtG@)7aM#2WHd%TgjWn6~mBKU5h~y{z&iJbDX}!;ogg);Nqvlbn^JSTxmZS_I-"
    "@o=@rp%YoWvEW7K2B!}HSlXwvVA^W*VFsmJs?BGQ;Axx@i$>gQHRsFO$OG2@wtf+XqXYjPL+*Zh#m*lTA%?kllmKMN&wqcz95^tci}t<uB(FzbSO(@Hv"
    "DjCK6uMg<It(4#(eY|BhJZ#NSsXImv;DfRGk&i@qW)1PsvpD%r17>A>8bVzPbyd|E`p&sr<Tq-P^h_RLQ7}7$-2;!g@@%9_-"
    "<^7V7I6{v~g(7w^zJrQ^Hq^tdj>n<bSsg-yMJ(j~>Jo=naealIBcS`P#fA4m$@wXFdEY|viTC|r^h?v?FnN3s^+P26|Az7UGr<>M+%yQRC%}(7CX4G0p"
    "{|HN8;IB6HMl%pK*J{Tc}a)pCq;fRTCUKd$94hs<k6F_*h5s2_Sua<nGQOn-V-pBy7NBo8_2pl(=`e$qxHDXbvbkXjX1t9b<TK0Jf`;7V;$={f_Z*vAG"
    "J9DZtWp=6g5*L<F<em)V<%N^8%#(`Hs0`N~Q|4o+vPc`lQ!G=}Z-"
    "3ZnzhFF%{B2D6oz^>dN~{=NYc~*9U#s|K1NEu2MhMVLsb2o|QyDeE+1zU*_Q!=1&{OeHim=+s8n>YpR9QLjijjm$J;~v*h=Da~M3k>oCeyga`4q673a<"
    "EBzKkW63%_9Em@hrS}`iI~}QCvirtiO1J@=x`+^1=Yu$Z7uHktEAf~#(185|*dNjFj=Ww)d~?aEfU%{G?9b>QU->-KZCv-"
    ">ed!qJY(nHR5oSN;KJ8}Ozoew&t)~eY#Lxf8H@%tv(Zn-5O%(1^fBF}Qm`#3nWgWd>-"
    "?uhC3J>1tF+Y?26!TUfU*s^)7i5LQW1=2U<Cvece<of%q25?uHyr)v>9ILPL?H7vjrK6s%ktzfgtpUTE&Wi5b@-"
    "G0LUYz@fnOj7jnpCgyMXisT<<1oS^D3ry(^yXS0O2!ead8V=`r?RTz9=y?ijySh1KkrHxM_fP+zfS#o+!vm~dH*VQmEzl0P;Ox2G~*t6O*@Vh(j_k$|("
    "~$ZOQMnbi9;0z7eew+dTC0m208cG|DgPeFe@pkJfHOf_+p{Z%CAZ=(HKlqW8%Qz6VOAdTakpdY5w-*bn1B6Pb7Pn?L`#H$SA#u@Te+$m2~-"
    "KD~NcLB9{{chT4cuVKnd*a7N6~=lAaHAg|Fpf(Zhr_e|5!qRTk@Q3Hc$r7;kyoQTcw+r*6#~8GbIo?M?lw!`0T1=WjL9nOpuH^ZyNF8{$cNLmd*b~#6%"
    "P0csK)y`bNqN||A`DQq|Z~~dzgR)Bbc+yhXJ1KfiHMt(LX9&OcoGL9<9f^FiG#fXL+Hs6LH>8Ksf7V1$p@l{a<;pA9_%S%q5=pBo1eA{Kxcj#>`;orfY"
    "DBI;SJ+{VV5_!1xz73d7PkEe13d;YysDN&73-"
    "P1}S>+`6L0cj~B^T=|@@6#n~_wlU~R9PYMAL>1Qid5%Ax_<V4AJggq+(0ikZpe*@&=T*edXI&Do>7x!m$@@LXyE{0~ZmdULVH{Kg*kAV&;YeMSOZ^Z{J"
    "Q-a&4iEb4@VymrnD^_<^}2Js2LmHe_)3c!;bO^oHg$;uJFD`(1ra#*Sc_>+BJv94=f)dp-?1?g{-"
    "O?h*=Jc(PyX|Sd`cWk(?sCyHR{hV0<LnL%f!83<f%q`B9K?C#X0J<0nGbZPo>{H$$8zvaBij+>FjHINYAzCpTR+_y*j?AdRvVm=Kl!B_a*Z%lzH`Sq!*"
    "f`s<G@9@oJ!~SEc6_A<lkq{-(xR=EGp>n@FzrHTil-JAXV~qCwdL=6M%6?qyhCwNCn@)pf?JgMdKV8!`VKsFN2T_d{fk76U~Q)5z~v*uN*FN_&F&V`@3"
    "}l{O+2oaZ6x{Ch6Mr9~b{u~DIQrUD&^Q^7CTmvNrmIL{YTRfrg_z)9juck)ql;_$0hK6w3Bg}$2=Xir>?=Quy4?@>1RVag6Qt~;|YV_dsR&)a$5F9*sI"
    "_cW-|k$ge_4`#g7#CfOmK%BMHV#!JYv-tl%xh@_3ebO!j`+sTClYX`)4)0-"
    "}+?Re|@Ce7}Ejr{minv0&=pcQkOuXOPGa9D3dfZGA@it1<EnBF^S8rnTF~)!k<elMRvR*yK^=Us<KrII&8g>&gp15Zt=`Pm6-"
    "wk0%OVgueD94w+XWK)J;Jn^BdE!uQ4gMSu;KaJS$9`cA|IX)*JHlS7QLsS3&q2(6@>etF)fcTN4D9!t%n=YtUK@FUInMmd&i6vI5wtHBP=h>C_Xsr+$D"
    "er68v%DU7_(e}&th`UX6kmvef@bC+?t}oAmUZEt?W0p0k>V*J6StpRS#>tYOX-"
    "%xj?nk9G^IIAm0sTZB<x4Nr7L)TN~o)WJ!0~dSKN6;>l43ikOGr*uPYyuA8vj6Wh+HU`5_s!Er9}`T%F@&&6(tE3Lwh0SdSh&wXfb=)}G((+$7hTH|9s"
    "j<b-q3(~W5)`7DdRz0#t+dc}c=Jhw(7rWA4DcKF5pIf6j$FZS3fI7LU1N-=fZm9Rl8q)_WP-"
    "8x@gmtmWo_ROH4c_mq(QcRmc~gm#9A`57iHnVBpQb|P2mwdu%J@9pPI|8H2Hzjn(2P@H-)#DaxY~wxt1j-"
    "i&iUV(qCn;h=COotv|r8ezyuE!HZN4*IsMa~`B0tnS=7iA12R;|*`~nYVRF4o=PCR?>w}ErDpYx>KqvBUKJ)w*?PJ2cF+#1z%WDeQlD}-F^W~-"
    "frmG*~%26j*74Vn&*MoQ>y$4+|DFEs<YS>Yq_a`4;r#`Gsy>P8x5O(@#5P40&Tk`G{>0J9n)<xf7*!R~!|3g4i>dr32!}HXGD|UurgB5X!{a{2R`FxyY"
    ">dh#}2wWMY#evEq#!^@L(Eg5jmG~_Z%U@`5vkrA5by_>#cK~%`Jw+_sqjk7kgX3`g6RgKo#IZxv+eUvazI+tWnYhuMI_Dhg$?uv!gz_5ra{jy6cl4&cB"
    "lD+@!XGX+8hq#{;3f6Yb&m6p*N?vxfEKLx+k*s5<a!OnVJoip$<jc0tkK}-VCqTo<}3QAKzbf^ItT}u|LxfaC6M=Tkw?an-"
    ")~s?qW?}c^wbHz=SuI304?bM?l0W&X0;0H2?``mmFu^btF*tbCw_fVA#17vJI7ErbDZ_$!6lcxVc&>-%6<h}jO0Cu&rewwZ-070wMvbvPZfwABA-"
    "k3K0x|S+6Q}ns8NHu;_(V{=`M~#{V>JK6%+5PFun%sca8iV%|4DZeW)w8Q$HB%3Wy_49;Y5&%K0~L?t|iD*6(5kOdXkj#MOQ5%k!LkshQN+c#!!|eL0f"
    ")|1{%%>5M;q8#I_4B48tV?ke%qg}7lC8H6^@TC}GQe^HIROMhk&M-"
    "%FXVk!A+8~a)5xk*3fkF?Kv=T;G@`d$Y&t%!9o^7E4<#wWO8JdSx8keVgp68q~mj8{wI+&0%#q^&oiEOmI7AIz(9miNh>Y*DmZejeL{{FK7HDkh!{s8k"
    "8BXByG6ANzOCZwu?!pFGeaC=?^@_2}y*mb?q_do@bVCEFV3f{o`@xY=1i?iAwA7IHH4EhoSYHS*OM+*|&gWgPKSdcJk*sRztP4N8v@@S40<R(fxi`CKo"
    "{2f42`xW7ceR@Uiu_T#bSm+=%k8PK4poq+nprTaU$F5-"
    "QoR9D=iz8d$R0y9=f&(q1L)Nea4x}bpg@RPVvV<!3FJWBRIJ{;?gGv!riFjj#_6PcI9pVN$Q(P9rI&QhT(<Mo&K-"
    "A3K}ntrW4+8Zl#)VNk!K<q5?Ec0^)^;@3`?s%K5!hGV)XyQ^j?U$IJ$L-y*J%&1PjRF&R{S=Nri1zjk-"
    "61BaaA~^&!NjYR#L0EEhZMUbqqYj6#}s(XJbzEUdXe_Z9X-&hwF*NnD=?n^Y;jF~pLyg?53KK|LdsnQ;_1&1wD%zXy!q^j)B-iW4HU4Cx-"
    "^z}xSrRaweo@$*H!yH>xlJ|Nj!Ao{2vP5@TNZZ`bB&r-;H9PD5yW@Mfo7>I(bAT;1%`5DAsQ<d1qKZU;O-"
    "~LTNhzlZc1yndeLCPgml`#)@i8a}b~!Cw<q=xsVqYSM$Wx9V%Q{rGO7{#)kOWn)PCL!XK+zu`U`@Kd_F<G0)FOP=~(`#Isjw6z>r5b(;J>z(5^fbo0W#"
    "FV?v7trT`rACzN$9->aGygdxHSuZ^T#ghG`vZcK(n5Us_Bj7bfi-"
    "p9!*TlnP#IX;ovnB~KsG6_C%f2G2@w!pG{yhJ#!^#A#bkxIYo`~og?320P_SCti774gIM~92V)z^irL(WG!*Q;ae2yASoMgKPf9HsBIq~{pH(tho}DBP"
    "&VoHhb(O7FW%dydndY~_a?|ETd#BgTdJu#M{ti;~WI48-*@8cgj%T|s>_p7(WRzSZd+go48wbSB<DllC{4ey1Z2&s!IO!96wDohM-TQ2D$2eBzZ5>x<p"
    "Qw^Bb97|Hw@!#;5$<39beCtQE1cpnA)N3(vp-V==P^bS7wTt^M}MG72X+`lqUUa?-T-"
    "SfflnQHi*RiHX`&QxB1nDN@a!3U52s_`%DE2*1&j?#lr=IBO(%L3{>R{<f+&!f~in>f!+<vsB7yc&%s2^c<!{UPn7DENQzM3tv%%vev}WS`lKcqo0pZW"
    "Zi@^YokPs{;1y2a4(ECB(lC(Lw0us73Q$0urgWCP?QpNar=@gyCe64!6$>c*s88f%93-@miQ-klbI7igqI2bKZYw-"
    "$EYy>YIYf^^I_CDWZY2?+WvLFYCM5Aq$Nrn6RJy^)l&<<%u#LZtH1_6(?mpoK4-"
    "_pSU`P{wtc7gR^;N9NH@49`%YrIv;Sbv`1SeipQ9+hkd=I7x&Wt<*75r-3&v=E_(FvV?RQiEF@08B2Fr5y5Rj|@)vb-_Hdvo^{0w>c(Jtybq@1qtbnfM"
    "(<P@k2J`V|sxRI;X)$cAfSN5CZ{k=M^Rw#!e=M1;MT>0$idc_3rF}d&PGYP#_7BitPfr2&rE?x>A4xye&htd^RyC#u3D8oH?V(=1O&%PS;D(H0DpcH|K"
    "n&x!i{nfC6z;M1K=OFbbE5)Rxz0JnnFz+cxQ#caWUEo<hoa<n4rk(6b>j2Kv;Jr{R)dEb0$LJp8?hg`OFvz!?vA()D$Ka3!2X%iIiS?f%s0Dg?l{_&ea"
    "F8FT%OKa<n;>bs|S1C;XYc0ZOrF0%%5Mx)!Up;rSl%>y-"
    "$Tx)CtXrZ{07;=LNT0<cW3nREP^;e=&;sgZ}qs{Kq`<!a5aody;@v?E7{we^S`@{c0PENl{u%QHglMzU(mT=okHPq?<3+)mEcXp@6C6zuAm?w?Jb0Z9f"
    "d@p++Uvbt>~|8uRKDufKQKA9aRs{MG{ho%u#R=}!BhU*+&&sv0$$3K-"
    "2k<S%j3k?VbVq#S<Nr~iEzU;1GR>u#>Cbas?G77+g{bXVXU^<f6<Vq7$}eqIm?-"
    "8DGAPC!5lu9LW@CjYmL3&X~H)RXLw8W0bUah$=dkBDz!IFX=5vt$t(){%kuGa_Gle>fU*s^}0nf;xfuIe@yTJI8ZgnTSzC^)Me3OWs3`<Mq<>a^L!CSb"
    "4&L3ZHm=GW!JLz-;F8=-d>%D{nyJdHFrtRh;J)>XX2&NpMQi<0SiZ>AQ%E|H|h!ZeA3HbIY_i{Zha^=FbY^q%(P{PLeNb>{G+3m4Gw7WPRR-"
    "Jl^=6KRR>0%Zmgoq+V@NO#j65-<O8sjJFmK#|ikC_GKL3mMve0o}rk_IM$jh;4X3XBIEm+JTmimAU-VBU~WAD<$B2HlisGT$#)6B^E3_CxeJIHFKvLt`"
    "+7)w_<Ca+`}9Kg1>KlGgSp<dw13;=gEMW^c(Po9h5e|{i5nA{SL6HnBf>!g^Bd{};<i2KIfl4b<w^iXztq6mSHRfD%)>W~NgR9Zhavd3Sc`AelWVDmZR"
    "qDG)J3DT0k}Uyi?`&FY}R!h+J7-W{hEZp@tYRm_9DI!2iq|26`8+BQ$w(WdcMMC@($}Tf;gGim3fs~0k!8D@Q{7+xxeJ25i*W-Y-xjxNivRYCceG1;9E"
    "i~TjU>-"
    "?blwDTV~7l6}{|m_>*jZN^QH(vTyCE6LuJ5`MtjAEf6!yvd`W+J9xa3?Wey|$4`*$J+9cIy=8nC@>~q(U!A<s@JD&<*BhZEp9Z%lzLOV@GfyKv`9N8y#"
    "mbukhB3b8v+{Q|i%0pR^+7F`uy5K-"
    "9ybzy9?%b$76zl<8Xf*)UQL$H8D+iu(_Sz(7{6BQpn4=gN51Y)y=u>YFZdh)gHDI%dj$kBUQemVCb6HIQS67%2rZVfe@P&|_aIKbW4vlN^~97#tZV9-"
    "vBZHq>am05sU!V-ka}8;oz(5I?c{TW+c1t}+Ll8V=36*<N=v-{Ks;H>I!O1AgyRPt+8ky-"
    "*pqsl^Vz~Wt90BK1KAG_8!4a#<M*35R#7^4<(VG}J8BTTRlr&L=MnGwNz%RR%3=6H4Gz-"
    "}pQ*pku#O(H{<a?pM37dCaUTSHXeyu2BE28tOVP2gh8B<1V#)bQA)HTNY2UUJp@<r$#d|jqCa%+p^*e#Q87NeN*2w^eZ6ZE3lJ$QP@x<d{7``;tVm13v"
    "Z}uCB#KUCH$C>@<MYR^MO(Omyj{HL&yhC2A;1`OBqZ(v=6W~s~*vkAo&-<*&2|>qR8r--ll<bc%n)Py!y2#ip2u0L;|Iz=YrT4=*eslJHgWq^!Df=(C="
    "L)=%-"
    "UsFQ3yE8G^TTnaq88r+MBHUuhtdzp^%w*4zPdn%Pg6w{SLfX6w`|VKG$S5=BlOsETEtZP{Vo0P$~bssrsC>f1M1r<OU|85pk2c_cfU{tr@9)^#8-"
    "*0#QobG{~~qP43|7Kd2GU`6eZrq$awp<n{>=hCib>Aq7IZOpkMAtx{>}fcddk_8}vvSDMD3?I7PfF$G+y%%XlOX)j>x+VAnxDBW*TuEbC7o2A<Ji>@N1"
    "*#D~>fZxZL#=4lYlK4jm$Qos$?!ywjV266T1+bE<8Iz*iju!8>CM1IL6PQSYv0f!S>EN7nY=lYe>b3bVxpemv4k+gV1{Crj<&nKI8>9aQ&Ti<E$tUmdO"
    "`XNPn?}hy3rY?tl<utHU3veReRg%u|pgm_xIfO)LQ1uJ>n)#q4cATa^bR)vBtc@0SwFUggI(7I+3}!w!%#6gDOdV<s6;O)#Wd258qi%1pKOQa0=#jBTD"
    "ETgX^n1pvP&%u-3ZDEn;KFzjjiv8fUrQTN5)bpUF@1^=o22j5IL5~Dmi<4U+F-rq-"
    "0d60rPDJl=j@p65PVa%|K!}4Th2?pIKUnXo%}uWX!_p4a?a_2_73p3?01((ZqKyrr@BDn0DC9dp3R)iSs>eA46?^-"
    "$aeK7&SQ#Y{2F!$y&&KBF7=8F@4b?|QM54w<FZV6+D*jY9M;<<Vh#N_y<ITU-|1k~iP+2hXi0pN_8|}^hT^%a9%D+&-"
    ">pn|AioFLMLN_f(tr`J)RRfXBict14-`vd@iW+fFRVxDJhoioz+C3xx8-4&=cR{|IGIMCK5&CvNW9IO5e(;^I^^sZaD{moL_A!QA-"
    "$jChkwelZ|WeFyl3vl{7+!qdIbf*mG-H#gc9A^N80C~^>;-Tj!O-7c>RL?OL=*3{gbVw{yh<0hH#z=5l*~rKI`QL?|b-m7+hGVJIHHJyk8-"
    "4#*aMdSSc38sd_}#BrZwsTa#zskOyjhDUa!A^eEL;#4e6Mg0X4B@t+J$#@cWL3MR<=w@Cengkv*%Rzh>ym(LZUW4)iJ{RG!>`bQ-cIU8_pDf>S5KN{)1"
    "7wW!wK}A@$$b_{HN+c6c>+yOe`@vne%Hx1ZkLIJrlHVJaGLH8+|IwS`(R`K;g^c4O`k@~2Z3X>3J~jrmzH8x|BbI!x(u25Kn|c1HdL-"
    "5nKgW5BCHk$xOZt-`tU5Fl<)><3JT2e=`$vK6eL$W3^0pVA*HptckvgA!<{j$E9*kq33l-ta`B=>tafo<vle(oed1Jh7I<nUrU}L2$IrpK#bs5Jh-"
    ">rtHosGCf9Fu-0TS0#GV&C7}wjp$t*}pVb;so(6n%A#jAJiQAXtd1)n;J?S;`QSguR6n|eL@-"
    "`NM%MF>3iOCl8#{<na_r>2B<d5gpck@__I!Zi9b2S`!$<tqUkFGPVEx$mT`2U{+UkPI`pJ6cK6Z4m@guhc=hhN^lb-"
    "m)FT3o*J}~>O2BI7XJg{qOs@O4dlUv-"
    ")uQVIq2%1tCBzdgas7Q#5>{*U@V63iiafXYKWbLy=YkQ5c(F%^F4R#kYsvdd`*Hk&t8u82ti!3}?7JC1J6?ZDI@kAL41T`QqIiygcC5o-"
    "Vn=EE&zu;J4dGgRA&xCz{#ns(#r55F4@WmoE%sLu@Rak2r<NMRc@A+)M6(6tokao`aoknE$x+m6uAM65+B!WJT@m0xp1WSk3afelUQ6?E)L_I6*1Pom`"
    "Ejmm2>W|Yw~iPbXU0xysZ~lVd>d%lM={q9-^*Ffr(kZDw%}5eY4&KTmF?0y%lj<*Yd!zv0CkjQUoz%Mq-EcpS5+O+EmgM1u|`{4&Z8-"
    ")>wsg<vb`x|Qqi*ie8Dq&3@9zzo70zxmiIP~?YF~n3$8k`pJ~Fn@a0@1{#3(-EhbDHD8hmIHHCSVPM)c-I08ZU^e7}>mm)v?qz+j|`^g3|h>c-"
    "?5hUUg{kxlMbD(~IqpXObMk7oWL@bDu&waj2e7e#)36YWP-=jp#l+Fd_^}AT_(G?QVYrFx^i6?`Z59!3Eo2-"
    "|Y`yvp{d^0>1@Q!)4jlTFzUK;2hg5*Iu{AC~glJ%R(Ja5N;Pm2kIVVDk${uR)M_4$|gjpIDGZ%M-"
    "EMFu?06tS80a)b3)raSfC<_ruEHKA96hy?O$Abnb1`u=`m5*!y8(72z7LSC=rIEmC7H``ao-"
    "j7DyU|qPezI<qJz&ia{sW!@WGU3~G5v|A<=Oq2mPuib25B*1)aQ|Ns{r!o<v<tj`Aj#FInh8N~>1Xy)9q5O$J*0CZvoU_95qqAA@N%J!;XJKbzniaiM%"
    "h?1+J-4nf%!R~^E^&Gyxl1Ud&~yx+%F=P@p7ZRLZS4WJ`qil^_Z|o#Iox0InH6M-"
    "=k6SxHnS=r@_QA{{KYI=U?V|g|AVVe46^anuzvI$bZDu=Dd%(Y&h1v)xhS9fbP@}4LHsP`s3XoUo<(dMh10nA^rS<^Dk&8JrB=7k5mIn-"
    "51e4T;A&{h<SA}qb34A88J0Ri5=7*Kd95*vp>sAsEc}5X1s2ygrg7paq7L#)C7+<HN}xLW<2h$gpc%{q_htbb$#8`Ch#3%#{6bVtmpMAY5%d|<OP>zXj"
    "b2h4_Wd#KGqNAefDa(HN}EICcF;f_3TTtU&!aZ_83tIssbZMoRGiot3`ZEqt4IomyUW5^cdAz#B1_hEVV*E^4kKNXdHFdA;MZLIdAM0@lBB4HyIOybL`"
    "{R{uD~izf=>Ky49qA!jdthh8~mbL|i0(jUzs!v7Z>PtpwK`Jq{T~RN}ZVY40wbqcyl95`OBSdC&fu*QsgW&A<Dbnt-"
    "O=b$E7={X$*VHuY6i^5L$?XtcViMcENT$-Y8atZh%m>&$={)cLAK=0E{MSf@?D6H~~0|8z{l)Qx)lxhs@>r_mBt=*IPRJy{cd&l!-"
    "d5mB4GaNWuZJ?MwV%NyXsL?hD6iHNghj&S_JgQ!Dv+tA#>7Vqd!(;w!~D9bq@4eiOJGR|~htRgJ?$-G+X2$vMup26BJu;AFSo{or0lkL(s#D7-"
    "E_Kdv_IO%CQ|LCc-AB}7ulk0%HmU9(d-mzb>=)*!UdrWyE<5(E`{9NX<%Mj^z?HXthU`840ie`-"
    "OB;I>HamhX;1{bFq;Fn6AiIRCxI#>5~l?wQrY=lD%5p9D2J?qY&`0)5uDh|Fk;!r)gt|OU0O6i=bIa%m8*aQW6(23Xo$2^(dnS8Xc3fjFhVh(YyG5e(!"
    "9Oo|a&3;HCMiv`jWv0$y+|#(`v&6Sr)1r{SM~_X6%fIB=($rB~sFR2Hh{ZHV;#G)<s?zse()q&F2@R~PV9OyRHun{AnErn-Quag3&UR=uTK2;LS87ahK"
    ")1n+ecM88pJ;~n9pVEyt-)x^e!rh=Q7p**^k$#dl;cakldsKbjoAtt_^OrAar~*4_(q(z#V3n?t|gtD!+Iaj`fBS?51+r8FwI(tSJHDv+J6%F9v0-"
    "|GWA$L+Ut{J(<WK=vyHUF$Y%05Q{&`wK%THaF5{Yp@B;>1BW_G#-"
    "uIv%W|6mDS5!dK0X;4p6fu(eYzTQ_W<BY>l~^>sti$fvA~MOxZRt<x{N%g?QE2R@Lw0)+0&#F5^WitgYq&Zbm-DrF$^Lh-bS@^xv1a@qy!XR{Bn<*D2*"
    "@Tr58*hM2T0#b)rV_M6Lzqd2q(_}L)|r(x~oz1hA1`MjHYdsXhGiZPW#Epj7_U{*fYxprA8|8*$ucQ?b9xuL+;oS{b$=CXNVH#sS6)64?i*n<L9+xZ*N"
    "8m>bD1;)Ejr{1IBWCn>MI>&I|)<DV*0Iqkq;ipI2^egMJgt_!h5(2lHVl^X(V&dHurz{5WSs3b}k*qTE-GV_rox&A_l|13oiP&JwR0Gaq^o2P$rkMkV$"
    "Oc^+bk-W|p3yU?C>I1YikbqIG6(U5v5nmD5;&yHG}jB~5mzXXbiCciYJ-I{*h->D*MHq^t`LBvtk*V50_!#UFXa;bQKTaS}vMBJj@)6l-Jn)KaJ5-"
    "QZxW8+%^!-#KFh?8T8m+kx$khP9|6Z>3&b_eO)EB@W|n+d3NOov_D$>R~!>_4f|rE>*`<f4Phh;HngPBLDniDO3U{Fc5AVYkhQma!rll(j-"
    "?TRGpHkJ?~vPq{9(vEO`c={FaBu}4sAIS&gs$BfCAa~Kynz{;{eb_!$F$bwhpCp+RpW!YZ-zPz_?$$nX19B|2^>ncg_Y+7{PhSd%*I$HWo`Z#B<Wk2?H"
    "_PF&yo=;QO=@jPYI`aF8rM01-YR0RXB3?ww=Sfv2{<Qy?i0rOLye0m0;Cg?vPNT@f-B)DbRbvzCv=gy|`8k&HE$qeGtDK1s-A!mi{`$f^Oy<3d`by7-"
    "^02<58An@+Xu<pLBHs<|!#Fmrk9keanAVs&oa4WIC4bj>BEA|LRyV;pUW5z#`sLE|2jZByaxxnE8`)3GIC+-"
    "yFQRVwTPYpE(!K=rbqH~ziv>60h%;lZ$m?B6|9`jmzqY?U=C73Pjc8A?Ja4={!VZE3=j(nYURa)Uww`T=Gaj=21h2ns8OOD`9gg_RcE?}L*D<nv-"
    "A`Ls`&rJ#Antv%%%}AkTht1a?c<48JE><H4v>DkD!|kIrjk9-M=>89E%{IsZif!-"
    "<m<DDSDC~I4eRd5tTc4EY(R9eh}Eo@^Nd#@al6H!ifDO9kNmSD^sK8Iw5KpXyLE|&x3wNC)`)n;Jg?5WZY}MD>KBI)>amey#ggYjf9cON)C1QP(P(r_i"
    "_+O*$+?vhU-W04wVM`*IxjW2LY;h_`sWkJzfOD@e5*0$E;QjH_0J*V$*8N+J1oF~b1h-"
    "6WP^cYl$b%@R9krSVUj&=%(0vo#+rLLRklBcJ@lhxdlTZ%8ulbJh*#@E+92n#nfao`MULN+c-"
    "5A875b(va;}*XSx1SPUh;E{^r6Jzl|{%Q@3gI`M5y$;K6%iFdFA;z4~MoGVNK57L%mXle0qU6os^$}U4aJt8NvFcUOhv6>r0%8x)G0Fw{^G~BjN#h^VM"
    "svm-hGriKy(Z$36BDy@`8msZaJ1AGW_p!&pxPqR8))Iet&BZC0MNhhQ3<)CP>@IIiqF9(`iZDSZbcWZ-_h0oOxC6my(f<f>NExv!<uk*GIdKJl#q@yN`"
    "ah>*T3yqt_BZHRj>1Y9J3-K0+Y&iNj0SqX{L^r%GpP1Z&B5B9}{(k9yV5wpXHUsc#I6PMapaH-"
    "`&8&vBj=l@0rD}3l++1J`^i|*rO`yLH7GX3y{xmY)_p_TVodvxrt1k($xnI{dWksI{iko(Xc!zL;b#$4!Rc@B1`x+5-"
    "G>f#%*Vbwy*KB%4!D6f$1otY1zmV9WDX^$)S<ooWF&YNJor1z@oh8CdtQ8R{35#dgJD1IWJH{|>>9Ww`*P@^sJfpwI`d!*7H(We?lt})>%`S&b!LRHRx"
    "9Pz4#TQ-KBHQ`np5hnJ~6Nkz7t)sBR{9*EavlLdSJyf<273`oHDBJ(C4@%%%TXUSY>l?sjlo`FLyPC0&=tsOd+==<#qZ%6Lns6wXIOZkwz3j32v#)tq6"
    "U*Gpu$j)W7~4e_9pWk2L$v%35<r~kVZoVF{pm}~b4YjM%peOcomy)LFOvo5si_Ms_|tu$9S)U|?Ry=8P4rJRF{oLCE-"
    "1Ib2K`)=xK3O;W6@nzRd%>+(YGgAM_<SpPpPZZbqx`pVn#<|-%;xDTaV;E(Z??r)!R{LQmaj1&YCUHC--!*Lw-lge(4G956MZH)KTk)RmRly24p;>UaB"
    "m=A9Aq;>*7uYe7mbhlhYz@R+Yc6v}Jy(_axw8Wj&g077<20FkL$TPC8e|CLWzH>F{}mSn_?-NzQX=E%L^$SQIzXVN_=kAE`6S@V?VH-"
    "W78weyr4D592bOxRlRvwo!lHZrTK!cbHHiRS5&_Z`m6xrGLu)Y=Yd&W^C`EgjV|fm^d?tK6)^*1?Ju{Bdw(p8C-"
    "9HWxcIS+r!zy%hI#w36{82Ut@>q7JPog8r^D%ub%Ps?jqZ}aQrg#ryb{We@+{0>taUVNG1MAx`KIooc&;#XSt}+#E8?x?PB^<!MxhunEn`64a?dYP|O@"
    "{M;ve_{y4I4I-Zz>F`e|7l_BB^b^A2tX9DdX4Jim%r^lLf;(s&{LwsmJKV`;N$I%-"
    "GjH)X_K|Z}j`>_V(#QZF5InH?15;21O;=(omW}kP;J`)?+r&LZ7OTNS0^+i6PecyVHBkeCrUQSOUSM&O|j86;qG+Y!72+)dX#BpBJ7t(vv)uvTL<W2)("
    "sh51H6BaRd?=l~v_cTX9ZxceO2VBZnVQ72VZ|X-j_-d)gi;h;fY_Vr5Y*Bi+MMtq-UR(U1<L!VOMY5ki@t*4}{@=UQ0q1(l_Q5X!w|SQH-"
    "7eYV?t6Kh1-"
    "!nLg=ZJ7vqQIovc0+!^Mkk%M~n#i(Gb&NgUw?^RFTd{r`?r&z0f5Wx2u}5x2st4{K}X1#>Ai8+Syq4*o20}nW4mugVK5H#DRCk^^iKpjGNs=6i}aJSoD"
    "DRbUU=Q{I1&SFL}k%PaUP+u(zBSTu-z@mBE(Z1(bFe(ADz2GxgPA#=R_c*TF}PaDR#!9$6yxQ>#9)=#V;N>=0#n|Kxx5%z~eG==4su|L^-"
    "M<+j?Rx&?oZQ@7l)%riOI4#U67_7j}vZOiY%^IqE`>X&8y9`Zgay9Y}9sy4vdmS&8h)@)3Co<iNYo_OLnwGI}aGhyBv5%tJZ-oz`9l5;My(b3z4kIzLo"
    "P=7XKz4)>(4kRzH?oRzlZg|Oc?XaxtF?G~*%RaHwnExi?z)$L*3-u}@y{rK)$3)C$zhHk~-q*d(_wtyqfPB4OgfI2YzqC)KF48VYz}XZ%E>R2Gvrkw-U"
    "G$WBzU@>3>N)B0c)p0K+46VFuj)veI3DK?>kvtuP(%8DfjASuK4r<DNZcQ#Lm%oWX&=`~ynf$6={uw5XmZMgnQ=<2VE!+;O-"
    "x~4l^W6zCu^GF!#?W(Yv{2h4{zDpLt#1pE}L=5vE=i(-|Vlu%l5gfA<=^WlQ!64Y;)Oul=xPW^Iu3jcTR7C1<lM*#wd{-B0p!l)J-"
    "}MHy=rUCcOF0el485L;W+8nqZ}M9*z$%qWTf)9O6#_ai%r#`NXqoi0){>`92~dS?{~wO5aA4ANN(p=}&qT*A@{-ol}+iCy4fB-"
    "zuo?V1N_xB$c}BKjKVLQ|i0wxu{Hh>&m(~N_zvYc?jdOX=x6&)i)xqg@`tcdlK!^@61hl)qtz75oPm4)Z+E?=!-3!&yp3n@NI9z5Z-qeajYxt-HBHpw`"
    "D;`d>cz0HJf>Ln7Lks_r0~TE(-"
    "rMq7rrNBI<`##O?y(k6PUXzgwF)H}(<ER){XLJTIDQgH%g><<ef=Vt3qXgFank`x+y)+Xwk~4ePacQ1`tpMkXq;#UA)9eSb%L{Dp<cd1Q;P=}Np~-"
    "fp(!?Pn)D44H4iG1k#*OC6=WvW5F7*&fLL=m*E?OWc?nk&g$x%vi^MDnPJ8k0J8)UfXO@V}N}93o9$^wCI`TE_U#=Jcr3<{pMJBf88zi5ff$m3f}jJ<-"
    "T1u+QG(h-!$r<`WF3Ds-"
    "+z~ddc=h)D^B=@8f>d^P4;4M;9ApHxzM%ICkHHSF7{w&~dwbeFA&pmLvXuy$6rU_J?JF`O?1DoNH=+6V$nDgV^50E9xi@_IF36_uuPhL78Gglcm%Tq3l"
    "12C+n!&OP9^Sh_gn-ETi6zA~!OhyHK;vZJ&xKwT&p|^<K=|!^Dk!jM=vp$+&*ifG>+gY)ob@vrb17A4Z=|MB8)&I?k2fJL*IoqGFy@pB#@d2lV)ND*IO"
    "Ye=q(2j_bYutq2WuW(;O8{e^wg74{uBnQvE9W03h-htQ!Sbke!-"
    "97p<HsL$(Y^xv&RU25xzl8z_NG$%f^KN5}J6Lk>TpQ`x(`#GQ6{JZDc7&!IOVK8$}$?LN@|7y}c<zg&u57r^A6W1&4i+G1Vrv5+hAq1M&T4<V3|FA9|J"
    "(u@sx%H+7Hia0WcrC()`Jce?bBVVVf9B!RWD~Ne#cR>eCy5WEXwTkM583-"
    "naA1#`7a*M%$ex(|m9?=xCT`{RZqz4%yx(K_yny#7>ms7R2@9z~SMa`NiRTX(mm~JIaq^iFng5AMqn>HRIR2)dNkJ}J&oDywkBDyZfPp;dLO*-"
    "I&cf2w26$4>?~~rQB%Yk74!K^g8m1Q*uz`7ghIsOuympItGJ0q#yj%?^!}^V;UB^6ZM0?EiG&qJDu%nuYvBa@L;>-@>$<>;*@UE^AUs%_UwEv-"
    "fH0`T~<RH7qh-}*TFt5%NXUcFsMKx=q%5)<x(VrLTw|B&o$Fx70UjX-eMm#8FUU9u$i6?5}&#&0p$QZ+XqrMu!^`=roTxWb5{;G*E;%8ohh@xQPDX*W%"
    "`#u_1fIDA}SkHMLBrhyv&7?FVS5|I}KE$6XxgtDie<wZ5rd?mQ8QP69;S=put*wx6IZtp?Cp$c}mvO#kX)DyU{H|wx(hheldVV@_Kr>vnZ+&138%zDpV"
    "?W}}e11rs(5-fLTz_NwKbFoqK8ov$;s<tS$!Y|L08t=BAP_WI2<}dS;!cs`4#nNQ!L_(cDOQRVDK4c*fnq<3yZ^q!_Afp>hPQ9-"
    "J?GqWcW2(yYU23~{$V9LzlgjSDpy~*TL-D&x2B$E_MNx*ZjiOVLMyFHpH}HRp{wXdn*aD7OWnyo)HnYhjGz6F@3!OV6*<f|-i`I-{ck;qd(-"
    "RT0N?m&>Qo}1I|IHCNU5b$!9iM4(3BHB-veL%>}EZmTtc;11*oF6A2NKu0zTW(XBA6TT5}f$C{H_6tBC(*cvr4BzN}j*b?y?VOvOzt>p*=!jNgWjGdAV"
    "c>mZMQ%xvmqS9+&|QYZM8yJjx6*yPq<8O*Qep-"
    "%plf1}P{d=#l272L81ntDvX7kB`z$1lv87^$)*<M`Xf#Oo*KX@Gu|J{+kBS6p(XF;$uP*8ZK^j87kZET`_Db;&eM`g?j-"
    ";vWUBvYjiVrv?2rYk{en+|Mu8`=n*)6Q4z?!Yh|fGtVSEZ^$KTC%z`RP_&W_x9&SljiXNYp-)-"
    "JcwDI%y$EpYrq|RL?stdvTm<7;vc&3h4!8bJ$MxVT9XgP@EPc<FyeiVgt&Z7Dy~7{aPT>1k*OyiiDp$s(H7^Z$xUL=ijctrBwJWK`ZM^D^2QA6?dGa!e"
    "`P{2YsbGM=_Uys`^Lb}2J!#GS<;!SBMSu0&Z)!~`-*=H)8UG(2KD(X29_%%>AHJ_dH;SRdr|wqJF#3X1T;H{r?>WyGT>o%YC5=1iuh=cr_u|Y!y-"
    "0vZb<qDGNBS#i4t=+^Z~VW$ePd1)sigYV{WTB$EYJ0u(T$(*Z~nV0>r^eqzcKZY^~`5HQv-"
    "U0?PXNi>95ZvO@%Vgw5Rw(YkxCuImOlV*YEK90{YMzeP~C%T6M3j@Tuqny738pNP+JT)>$>Of!1CPQ2vhG7y8)--56-?<Mmr(?Fb9hJNV*dysG6x81K-"
    "niPk3uY7e;Gz&hVPhhN}x*vay`IM!dUvXU42<*tw596odNq|yp0;IDypQ&)(yjrAX}`o7j>G_sMurg-pgMSMMH2<v~JrmUjyKVjhffAHuS>#_FF>sPS6"
    "eli0TA55Qup6`F*dyYlR>S|EEK#gB#>L|WzKl+@B@%2?|Dz9Il>|0G0WBh0GyAB^-"
    "FFa8nPX#C?p1k1a+u<9B!ka@WmDJ{Npc)gmdh|zi$je)J<DFesZ~6r(n10hl2U7mif!RyaE8Ty(v<2OG^q+1NeVASi{`<bAS5|P&{cfdBx!*L0(`gl38"
    "=uKMvGhR|4fJ?ddi7srYQVnKs)oNhjxLR{(YMl1eR*waSnt%j$-TzlXJ}(J)0sgnN7=P$mQ4|SX4FE~3a(m?57VdpZq<HhXgb#iji--"
    "D!~MNYuLahbCs$G{1TEZ#-#+H#-kN66k;`_C54Pz9*j$4@kMf`?hcYNUQyM+0WK#fIb#cbmWA!w#RYo=Fl13YUuzh{kaQKhX(scNRBAHa-Od8eMY|}4r"
    "G4EpP7e1>}$;|M?Pa!vK`kgp>__t&Dmm~i~=-"
    "q0Us(v&yV+&Z=OixU_zbhH6yul7_&SfZY6*{$xT5i4HeKDJCy&O7U+EDQn>Gf|?7y6Iq)kC!7i9_4k7<xGkAC6wt0H+g!!jz|kQvrhveH!6Ahx#%+_@-"
    "l|${crU>_S67vi_{Z?IH1;{w$-u{4b5tFR&@m(gkqR5d5Z{9jB%>-"
    "I|)#R2)31&Hu?!#X7GnuWBE5D`O?_iN5@Zp9!vEov&0tXOcY%Y>a<^4|UKPKlFKc;X-mR@u(eqxx#hxh-n`1-"
    "|<FKl@0Lfb2n2TE0P!d;bVAhbS|lCqrAF_zgojOK0Wa5gHd2}8FjzrRc^fdD&lD6()ymeL!a^*EPqv|zWh|$_uODt=I?c)ygt#_j6okNu<i)>m_S{q@v"
    "@kTBzpCb`#OWZPi4F$e(jesB{gA>SMBqdDvh50gFhL_{Ur^msJ`gf%<||4{$V?M@~AnnI9F45CI%>wI%`8uzJ;er#Oaym8+8u~)YZ1Evmoo^`bpNlShu"
    "Qb0=~|kNF8k9t9MhVI|;w$Q~z&0ns?mLH||vWg<3Mvf&YJ$PJ`m?iYjQ+aeQ#=|MnM6ZWOD(Js$nJ&ro;#%5vrzOI;gPHBOhddDP*&@%4AiM0|BtYd@G"
    "j<5e}rs}g@3DnVZm*^B%2g2Tb-_5D`6GT(u0-"
    "0Kv^1HobTY8lmbWg4CBYp5>x^HT3hQYTXat7&TeAQdOS4KspSc>QUJwJ&T3h0aW?WOV!aM|9`EKIL$+40`$B_a1HGd$Ip?Bj_T${!ce5;?D~+PbmD@l&"
    "g{Mz7A49;`{|2IE-"
    "h%0w29qGSL5+>e1h(LTK9!`lj*ltL)nhI(6HwU4PiLdJpy3+RKyt&)Sv+P&!l>oG+t}XI+hV1J}PA;J>d&XI~g<3C2Id_oVb(x+tT@_fDgCQ*4@b87y0"
    "805X5kN|`hTUg!A5rjDoK4}8CYo;^;JS+jEcX~<tTeRBw(3g3I-"
    "zs}^$qQ7VQY29a=I{l2+T4x&&zn{lvRsPftB?TM0u}<0m=L7LgS2AZ)^R^DvDrjgs`dfp~NU`dzUryEe&83@944s%quYjM4rr&-"
    "3nGbp4P)b8X`)0#yK09NSl+PK87jSB5J3|d-"
    "_|Bq^uyo*Ls7|2+A9|rPXyIq{#~WvzzYwm(6sNMyFcgUfPdW>C@n1EX<kaf`m!54l^p3fAtk75gnC5I|rCaHzf*#{*{Vs3q-"
    "vVzs)z7Q#lifN_JsN<YeuQr+LSJ+A=Xm|&58v_0m%j2GPhX3CRw$%FYdlKV+Efd0X@Ji}a9YTbs5VJnMfEqe7knG^LD%WaQihe%gneEa^t8W{*I&?q2z"
    "WH7Sy>&5_1C%a@CzUE5ByEr1n=6Sk{XX@e59#q)^nlgz!?0)97h!`?TtU_V5*L_4-"
    "WM@3f(BXy^`kAKdxt;Bhi_kxV{H^b?$a4{c+x_y>X^8So>AN4QpSqAHO7PQ4{=3SyR8D1L@%V8ScA6j#_#*K0vMMJ8lv8(fEWy=)k1%jkGvdpthn{cjz"
    "O^S-"
    "%Yszl`Da6?8j51?jtgL1$i5tE}hIGRIcbscZiF1N}LS&v;KfZxZKSi{rGk5dG$FhRWJgX>(WK_ZH7PQGch|)qI*w?Hl>dhHZil<PJ&Dw3{B?IBV!Gako"
    "?3wp#mREKX3ZYF@RyVW=j5{|jFCLXWF-E3ACmyz)LWbPvB;<|H*6y&PE~N>yrjlqS^FeEQP$;HO>{>s;CxbuR1C+bC1{zWDay{d$h=$@IGZ-"
    "}e%~e?pu8`@V5?p$s}a-Z%a;I&km5?>F|}Nv{$A{hr<&-"
    "6@Nnf22RHxVfGdUkcKi(x#3O;|y@)066^RS_TzwWS1S@9bk<W!0znAU~5!nO<U)u$PR`!;t?8bwf2C=C!P$`RF_-"
    "#emC?KKa{W?J#UW>9GY8Ey}WvEGxZm}#T|>+^lfj0vM6w=pRRtf>Em8<vWS|5KXN_DqBYO`^faBJL2!3FwYjs!yGp@&+{~eyafY6+<nR33VC(tWyV<nn"
    "phFYO8v5USaQTbAo=@0zLJkFHa;j>5<LmkFJLyNBf(u7bs4_NmY72GzIeK0jel?*kOnMTlgBRU8M_+IN{_a1H#!<&Fbqv>@wN6D1F_eVX{zEJVFy13LL"
    "QM}i)pi_w#}_R=g{J_gGp<GG?lq?(<`{ZJ%)Ib@Iec#&kyG8$y3}Niq4Cx}`1CJ1s8<=9=TgV2F4fs>=)0Y)2hF-"
    "|?LX5etL_(dC@{&;sCw3U<n$ciVZsjydQsJ*s`P8u;ZFh9KZSaleo|pw`=3V(zoXxT@1yA>et_>smlfmxc{LioccboKppR$--"
    "+#21)bH!P8b8{U8=pK1+<M{lpDD?@Si)Z&XP`6i<p934CG|HaeZlGT{^~W))C>B`&(x6(@T+6CnyQmOKuzYF>Wr^$1b_cTH{PVJuH-ZR3hHI*Z{|6P4!"
    "q!VS3j?+TdVvPL0wBiAI39&o^{TeSynOhJL9NFi<mzZ{`?E`Y#Ucqo`L~t*c?3}f4SjxB>hEK|N07i6rhaXo5}<}=DZ`n)E$*?qB#u%l@=YC7UTQfq$}"
    "$`cC!I}b)fFk*Bqg5`Wv6{me1|-u!7DVr>}%}cj>PdqXU`I0W&U6=L&mNYOkRh@M{n{a2{SPj?ADM+0$s!Rhw4Ug}dZs82o-"
    "Pwvg_Z_iEEs@{6wiLVOz5w4Ogn)cQAG?S5+L61uk)-AixfZEpz$7WUWLcgENEZLRm(GI4#z8~IhVxmN}9(brr1{Nnd2Fuo&Gf+nKd&x)Bc-"
    "s9{3+c#kDl?*!nAK&){f?MJr3NQOk9-&fO(kT<ZD3|pfKKgS2zL))@n!ec<q!|TGbtDF%)a-"
    "PWaC@zcx&Yq?U$$w^2I>%4PX({fE(uom(N2BbW@sy(Vl@BO7M)A!6Q=&T-"
    "I{aM&<ZeH2LG^=d~I2qO|P@K6!)j0HF)Y#c<8Ry{&h97s%0yOLJ|!<UJGWacVYBNai4-!@Qy>R>lj*sM}E7>_Z)b>r`gr@8>h-"
    "rXKPIdKf8%D{aM4XP}SS#)IR*ujB(PEgJ2L{D*7pm{>!C8qYeEs+_zuW{}^w&B|_s$x-"
    "@LMp?@o(!{GT(bU5aIVVz6mRbaHKsjNL8JiX5Kd!FUgq;4)%Ux6RMhh;b~?PdJts$8lv(512K4W;0p!tsZ<7++i@w}y{&Y1|G&8Hwc`#+ShZ*X`U|Jj<"
    "nD4;Wv6rzv~U+G`)2HjUDyr7m?xC%66t7SOTF)_E5_vZ)t-"
    "<I+Gwzme|<>hjO5^X9WcTD*}uP92OS?#IE&MsPP}S`mFr?^Qv3{vPVkW9s{B>gJTm#kG30SC!CuYyY*s@4-"
    "3pa(h!5U5fKp!*RZIH_qbiKEVrTrpjvgm%ld6F_l6-=290rQBV7MYpF$-"
    "0CieyDsw#b3EZZpPKHgYtI2BuWS|3=;As$e+rd2LLThVb*#PAnWNI<_6TJR{zE+u4OP}Hcw3L27BiC<w;d?$jSM6#lTR%YisN=uFQ<K_r1^)h9w!SK44"
    "AjrPO--inY60IjTIUGoYNj981ZpBWX1#}8>MdAgywRoRx^gs7L-"
    "50?!C4^lOl)qQYgt>f?gZ!#eq$~^uRVNRNPP0YD5Iwv{1r^U|EU{T{zEDOU;pl$NBv&7wHzHDh99>2f>!V%Y+42lNSj87|F)?C^Cny8Q^5CMSLaiOTOJ"
    "jr?pynJH>EG=!+2|3UPXYbjTa3SMz^n_!xP}q&VT~?y{%WtHx1>3{|5e{6S}?sN`m_D^(wl6sUoazG<?6wc*xy?s<^_dCZ$Yu!vC)({^JKzZ!@(~uMcV"
    "EscEVeSf-8AM|2``b4F#T>!-"
    "I53^~c69S<=DTs?c8QAP3X)97<QfX!1sP}8ViKa|R>njwBVA4}bYzdL4GKc<ia|7;2`>{Qbq47uR%4!B`G*WBc4um(PN=-"
    "^WDj7M%{ookTCI?pzXY&M5RxeQIC<~Ll5W|Pmuj$p0s<WR3jLwV@AFH^G@SkKq|7_5F%9eR)#eONB7h3}PF&!2U(Ypdy$t&*W<c!WNDb`fxwKU0W~wQ*"
    "`&6NCR}{T_fvvwr`76sjpZoZ8uy`+^T9I9~)0?x)YM9^E`z>Y=Yeue#%-;=o-nI<P*2OR?h(*{t7csMqruKR72+Kj(Gn{Ve0_bJ9=f3-"
    "W^BKHYOFLphiHxZfVc-GM$t6tebK$ffa(U5ea@ufQvff!Cq%soTIPE!pH!3VfeR9}#Zte}eyi*DQyw_HwE^pBrlVM)0r<9eO!BzqUkpltSIf0e&LDPZR"
    "3){fPy1zLiIll1yzy2X=FPUhvSVT2T!S@#+Ts?-"
    "=m9hx=`8?SH<hl;SUYm702&P}27tsr9~Q?_%`nL;clvn5hluR9i4H4*a?oR#j5k0JWu8wDu3m2d<`2$2)zhp}GwMl#W^w$Mtg={}Z0hX;VY@f&&!X(^r"
    "=#!|QJN@FVkU>uSRQ-"
    "N0{LxAw)LmgHo9Z$eECoE4yz?dj9W({SpzwIBRFXH8{I3D9YLcmi?R3s2Ycx!JSS)AtHg9me}`{r6n|GoM?3Tn#N=&*xS$)sFaXdhY8l_Iy`S*K!5OKh"
    "{(<er7!P`v5-e%uqr}q4*8qGkZ|i@o5dH<MSRysoev&GE=|0yt2*}rpJTVhw^6Bh}voN-"
    "7%YPfvMRO@fPHF;2&Al`<6pb+8b&(2fc+GgXo8{Izu#holE8C`u6>}foFA;l`0t#`qbU6Z>{$z@d!((JC@&fIij?ulSi%nO%4AJKg@a_kl*KhV|8JfN1"
    "Z}UJwmfrRq?%NJ>t7eI=$6T!%rAG!#%CU=jTq#{dUcyh9m6o!=}$@L*7Z2_vCZFaDDeQ3Yue6L3+@!)6fn4aOovk6gl5d?Nb{na)i1_UyzS{o^%GQ=?F"
    "jN%x7rBaqGtov?04x$D2icoB65CYMYX{K8YHf9sl{Ne-"
    "<^F=BFn6Z0ffkkFboskN$d4d{(9V%a7jOrcnCsG5GU0)^l6TUp~^ICQofT38y3R3G3;*%{ReXfeuu*8(Kt<exH6J6T1G}r)=80&Y}I84TbcCE7aK;_`w"
    "!ma%$H{>M#A*bk^ceoImBZ_EE^8Xpd8wiy3N-K9sfgGr{+@xE`XdMV$Hz-5837)u5L?g)Vvbhbju6U+o+E>nZpUVr0Ds7W^tqpQk(ZxU-"
    ">V=z|BoCxG9yyCap}O&!NS1mRBxqh$g3rq<tNQ<h&HDj3B)?S1>^t)-"
    "q^e;T8*Rp=M~G1QTI`H9>Vg_n`N^Qqkxx0a?e)gOM|wcaBJ4}J>@>iJBM%H%h-"
    "gF0@*H{L~8CxsW)V~<yP%b3ayo)hW2OIYV@zACBj_IvR_rh0(Ck!bx-"
    "_~(5a%BoURfBgf$(iaCm@cJhDUw?B&V&$(5o$<HiyAyS}3x59Esp@*1AwUf}_<r}fK>ZDar%CziY5Vj5b?#)UCHJ!rz7Mg!A6T5MKl=J>cOg^d=?hXA"
    "--o_t2`{4-"
    "E&R1Gr>R1$XFubsS!cX|vRV!E*Zdr&j=_gW)?@ASI`I2aT3p^=c>_!(bN_R>FU!|VTUboDu6kAOy@7s!GyKeHI52H(0SycF>M_3Y5p(atd%cCHA4g<V^"
    "Qme0K$|8qz8{UQogTa&%&5Li)97Oln`Thgy5a>F;0vmi%cAdg`|0y%`pP4g9|F61t^MS(sLlw7LK6)Agzx$ZJlC+!BdwQRlkYqAO-KA5H6ew5xd?HZ=g"
    "OjkZ5%SD845<rZ{R_f6W4_$vneRnsW)5c>*=MkGG53ZUHvw*hOM@1*`3rnehECy#UrrJpN?nNhc;<s6t?MiYSBpQW^H(PVMP`V^RmtfHYI}D+hBJXxHZ"
    "oO%aIp;owuo*wfCxZc3cMP=dH}@{3(r|RJ3U{>r6+#Y~Tw9{hL__1N`)$o=u(cGI5KUgYo9hEJ_pZr=*rPrRQ@~QJ2Rt{-i_}4N3A-"
    "Vt1QP{SJ<a!wbfzbffQX;-^hN*ff;5-Nv66&xFpb$)fl^e%dh6rpDCd3@d&6YF2!iMe`~-)FO|e+`o}0xIK{T>*dO-sKtKjyws)$>pWt(G>!2m-"
    "(}V4&3@{-&Zf!xz|DF*7vlq$WL5S9e)_P}_VsM}`Ww+V@;)Yeuu4Dh)83QZFKdivuKVaxvjaJFWUE83KiO1<^`yaDG~oIU7qaSBR)=2gvZ?AM-"
    "+3I@@f#&W!c}vUQ}xjE(e&M&sN?I<`Mu|JYfpcda&I=&8L!miPis#;`oz|;8eiS57dH&eqi&T2pZno;TBCp#<@2bE!&F-"
    "EeH{KCK+ns(EutPK^_M=vMO`=l=j+GoeJQC<zk0O_-)`*-"
    "@b;1K_xD?~%c)8?e>JObs#>D2AGG$9PU&1lmoEEjD&wb^=K(d~eogE9jZ*5j$E%)Mefz5yu+F!#zIQznt@J5w_1bSJk$&VLxnIosdq+p9ejT^^ZZN)nz"
    "w?RlmGFDnyeM`0-mOh*4SnkE^FP);x+`8sY0LyS{SCg6d2WK0ZH!m>K1#`>+!~GlImKEw3_&-t!m-"
    "quRBBHeb=zZ8PuAXk6ubui&Hm1)^(CoC*==f1@A7dx_(>;SCCNR1N~0IaHZ{BsKFLjH@|AQXvyOuQkE3lG312oa-pM-"
    "W_7UTQ{nQG+U!`WYw|;BMX+0m5MN7Z<spmtk#}jqsb7?g-=6WVw%<8AVWT-dV{E)u2l=amuyeM1VPscJE3Ioei;MgNR*Drq#C2sK3(T+CdppNY2Gk$}Q"
    "O-"
    "F{RaZZN{EVStiT)9M@dP}`neJ6(wm2fEhwN1nEt3ANgYIt353^)Y0N1NMp0X^vsml|@vtz9`}U+brsHa3k|gMX#&{4WSi+?ZWAXZxu~Ynwj6l|Zha!T8"
    "}@+4W?BpMLIQQ!TDfU_6-dNsqGY$U;Az>}}Hv#?vzXBmI5AKiL(*_00#tQ#?y7{_`1iIr?arx;%8~V?9G&Jj*^l_gsMW+-"
    "`Px=lLnyIO56pM7UjudDd^vu5yd~^a=k^0gkq3{28BH{&{xIL_e!-vuWKD;%sRa`rzIkqNmv%in?Rd;90(NSZWZ@GT(*k@6iq&{nw^D#9=YlXJ-Dv%pd"
    "Z~PorDeWJgmoT=AXLy~UAJv+6iCAK(5Iyj>*sv_SgxRvwKAb}23WRB&fv03X|f!*(y@l<~Do88Vu>(2}~y_!`US|Ck@_yIF+!_4lWV%)gBKRjEZ0H9Y6"
    "h_A#cO!`ty-BAowsbXsY-g8a4TXH)Bnp$+glJ-+PxyGm+aBtU^zP5sY0_kntyfx7fhQYjS}=GEWyv-"
    "j|$zkuD()aTP(3MwMKM}OhlGtY!Zc%z;4o!1A2Y4&8Nrr-}J&p>Of{k~$Yb4|n4<_D)%_cPRu`>n|R-lM;$ZRF91SuWkAz6`hCV`TmD)T4oMF{&rGN-"
    "Q?C5iAG5-@oz8^NvL-"
    "=S!!G;AeW{uTE?Ov*5g4p6uF0o%)O)8wkb+fR$C?^TN!m$~?zUm%!m&yiCFl)`Q+gpU9@SpZs*0y4e664%&k^1pgVX=FmC%vJL3)3F{oY-"
    ">heZ@%xoSbRxS`(GP5zhnHN8ezuIZzNd-M?RQR<&2H!jYq<f2-s1yjRL`c%wH#Vazt#c{Ed+CmnCHx0a8=Erpgx9{abJIdpHO_@u(J^waK)kT-"
    "q<t_Jm<jgy&*r7riN=z2d7?Fq>rY?ZUW~!@x6ERh3jTVhbA4gY1wT0v~%DST{_Y-Ob07F^y#!si{bV9Gg2+^`73>5`2=|Vr%f9fpLoXC*GGE86jIEgtN"
    "*j95H&d;<HZ<nlOaqe7*Bp@`&z5-P=gCI{xl*?chQ0EX~+xp!Vg?o&zm2w60O_4T-u-"
    "3R72*k!}Tv&&+G|d`te(bp2itUqK~M|czWWz!4|H1KR6Ut!B8AHdBo?AME8b04Oh%IhY}kadOHFC!~H&n*TKI=YI!QB<~27|5ic_xoE#<&<+?;_#3_d!"
    "2O6q?zGXtsli*#0h#YDi<WSviwy*m#7q{L^EkMoQkVkz(Txv#tv9c4m0*4RESZ7q^Rh=|${es^sNFDm;GB}}rHK<-"
    "l<=c34d7P=`)_Z@}K3(vjPWeO~Y(l-;2Vdx~#=*y!c9Q*7X;rK0uljdPjV|mv-yt)7SW1JcYI`_971G-^-g-"
    "Wk_3VM)4N6p2=_UbM^3GHP>wE_^{T4oU*`p+7IpI~#dGv$Ledi-w<8wkH3To<Ix9Y;XeZ#1|+;0wa<52os>R8yNA=?dgpie&yuG-"
    "<>>W|H>`L$hgEi@Db&udxxBoqO2@8h&5k6W{;cY)-w8+^~sdfzEd1Jk?JhwFR7(*vx(BfQ=*CXZSSbLmhA`liX&IkI?0{Lk<s5ps`lsx9$9N?m9G--nW"
    "ynr%XK@F#~H8*Pe0KR2PnhqGE|<bcD?_?Fq^WxVfPo<Eo;w02IdTkDhqyk%tl!}(l0zGdaOTpG^$m*cM%bi}{i_w9obS~6BfNw*%iG~}R%OW8uL#n0@>"
    "6{>C<oNDsKP&sO8E%@$Cvh*xcMk^PwGS!*!Eok%*^10<exKge=_4Efr8Su_o4%2%xPqVTS8a|%Bshpv7==LPm{~2AHyeU%cZ#(s&t)aa`_0>D3O|-"
    "s4&#5PATsnfjJ!k#bsgI`_Uml%H<?^`n?^ygcUj8^b7E6D3rE_k%n{r?C46VYSlmh=hGd`wQlvWOMX%l^Pv+Cr^I#;E>71u)g@v29Ml1w!NPkrFUar&O"
    "#wW76Qu1g_1jj!hfg@Did)_zUjL@R$!m(t;XUNZkw#<Ox?S=Q%K9dJ@-fuSY?sR@jSfRidqqI4f!-"
    "A~^)#QMDkJf|c6od)LCriD)BhmUpO*9ZE|I{1x%OgYq}l|z5twCP3*`fcz$lsa=~dOp2t?bg+ghQ6`h-"
    "+*7AS^wIxg*4<Rk8U(Km4oZTuUhXSlbc$l^(fk3kHC2(_kY{khYFnLJX2l+0|MmUV=9`yWhMCk1Hbm-"
    "&$?QM0SJ0XU&;7g;!~RYo?pG5dJGJdAN4B%p0B@S?WF{s|EQ%IkKxx&Q+e=zeeV0dk1ZZqNojuY|N6c0KXrWX>JKB{A3G=N<q410dQ3HC9fw%|pk!%xP"
    "@K|^acj{TL;r%q1;i~Q<M+zPslzn4?8gj^L^s^<%OBqwVlSxWJv_Qc9~a$5U%#{H1Ahmv%%|X&ZvA=K(08mmBmI7M{ObGF(F(}vmUE7wLh#}^KE&F0Wb"
    "M94ZKEF#s%2;owdD&q{|CNQXqQ8I7dv#BdXxgcexrUR;}4g$4^^Lg4vnJ@?j7m-{WJl-T>2xI_LX<(KoLU*_?ZEI{D}9Lm*6eQEf;-X9@c-"
    "CI9TVy6u6XM-GB6`Fgj4AEp>}}H>aBQ-"
    "a)MDcJk<fzp2CM(s;(L_ZTL0ET9ihJX!$1n&TUjPFe3@;X^v)P|3gjG<GU=WR>rGp?>(Q$HTI#@K!$+pO1fA<9m+iH*~vn)9f0!(odDw+BAtC?j?N5PT"
    "#k!bBI=?b}0KRo32rpt6BS2;m4xjd)*L++NXnG^QmX2t#{V>6CXRFkVAd*QJ2ZrBkJS7+}F^9F<SlvzD&VKQP=FO^ANgJ_*1B6;~#c3FqD^`b?0econ1"
    "LMOlR<Q?eH5zslPog`22PwTZBgLai}wSc`y~f4IWM~PqE1nDs#x8Y%zvLQP*7LJ%DxYjb=Q>p*PoT%CHkn#aB><)Gk-Bs;AKBdTm-"
    "W9PbR?=HV;6k8;X3#Hs)7HB=X06h)lFsZ-{rSXDmf(!*~|m7q@K=Drf)*`l~Y`eUj`9S-`QI|>F@C9Hk3UYF2qo4-"
    "!l?HWbCvoimeZfHc+3To0YK>OnD>J0v7SkHx02gU|eP_CE&oei;n-"
    "8X3pd<<irE$(tU`;)(pP^UIDphm%WfBLcty^3naGLPB~H&q_pyGVRSpeG3x@@RHhm)4v%w3S$vq&BR_hZmZgS@V+pl!d(a+h(1ADcQ(Z*7DhOCeoq)*K"
    "K-3-mjy>$BFZqSs@yJ)}fB{xsTyRUuz#Qd{oW(k?M&q)r~eZW{T9G`<)*OXHp`Rv%6D&I1FV0C(-mZ-;$SKPKK)V{~XG-&!)^9z#%$zhd$-epV<{(-"
    "Jw+UBZc7C8tT_xd}CDi9J-9ZayPT7BlDjiZXNM0HJeB341R3&5u5rB@;&eLFMJ#plUwi6)nKooFX+w*dc_Oq=ipB<n(!R`FK_63@EL~(O$D!;RSs24>-"
    "<F1R2cWwh&V)$-"
    "_G}Q>fIcd7PU83A70d^?!1F9!!l&o1NyZAhip0szv{v3G4Nv6rX2eG(oc5$SlvZ<M((#B@%(mfh<?cA(D$*1ZW9Ls9_`AErh7wGAuqaJ+0Y8c%X9r!eB"
    "HYEP~|S^&|UoMH26M@dNdwAOkX-qZ7RA|9G~-"
    "jitqfv^~7_JCrop`!@p5~Tg<k;uOSxj<*I+Us!ew&t}*%sUsjT@NsK3Ek5Jv89ok(3Uk6`){L^|?9j#drp#{G(PhLY^xZX`&$O&JjkBE?CxkI_>bNj-"
    "V>hR?+Cw?kNu-endjm-#7egL1$Z|y%ibAK+~Smo57yN251>&mjum-"
    "I(5JMt=LE%a@ssZi?BZ;bCox6bq`qPd}7CBHRw5gi{1zsh}2U7lY?6EFBHwuoKn;KdtoH50u|@UNikEdw;5tX-"
    "?%MQ89Zjy%U)Evq~a{dF+6UAL)gli_<TbuI38l6IBwS3mf2<Xd<Ok2Y1Mk1SbOS1WoH_=BnJ)Y&=k<ss|;v>-}JnOqu-FFgkKf1zK<Lf#u(%%c1g{4{g"
    "B?dv(*ANJuB$ZyFR*|n#UL&qN5<i!6JJ>fg2e(rtz&nJgG9z)6W7uAWgKYUp=D5r*ObZTZnL$&ZfFNs@Y_|mz0qyo{I9cc}fLH9<H_e<9I8#L3Gy-"
    "u0*5nI6FeR?VbUUVOj1D}iE##fKVLj{6Q2RJ+(8lrpXWKdU|PJ-tY;!uRXtj_YBnmf&@34f7Ce8{SM_%8aL3&DBRxra-Aav5q&ogMdxd!|pEn>kMN1Kr"
    "x!$k0-J*9z)dgR<OnmYgc}%%y~$rh1c)w)C3^Yg3!^#A<k2kJ=u>S5OCA6NhZXr~myNdggLyI{F_DFKpnvKlr~rG(?f%4i&CR|49Fy4sFUn-"
    "x0DaOq(V-"
    "ban>5m+>&hos2i7Zw;UApocWnnY>N|Crj{+wM*qy@0M;|K8ZgeZrvHT5w`==!_=&wLzkKvN?@L<<Tr%zZ`*`x=`3`nKK^i;RFHbn*Wyv#2(8}k(E6%|4"
    "#0~P`kG|;Sn*_p4(@emFzdJ8nQjk$a>0umuOgIn1H681(`~%%TX=C8UQ{a<tk`9K$`)?ZdwgAA@EL~=ukuEzI>gDF##9-"
    "rKS2i?)RLZUjfYomO^4UkIoI78A7-"
    "7~cD9&CPV*`**skoveHU>!O1;>&p^RQ!$EQ`YYd!cs&i&@1E*)J_R;kkkXi{an#)1Dy*7tJM+lW(1x;DUHPIxgL+^+|RP2t_Hg(c~!y~_N`RNj`-"
    "K5(+xIuFsmpnksR*8aZuG3wxO<|zsuY@c&$$-"
    "ho@#xE5E3&H4T6?{dm{K4vc%1>b@Y^t}C+6ZTIli##Aa_Gq@hl+o+={529g6DbgB04r)zvXZ$B{RA=3XMT$&WBt3hv(9oeNJU4WoV(bFCp<CW7UyPIW="
    ">xQ(Hs8&kWzWX3fyCzNI3Rqac0g-"
    "!>g%jh42)z_+yPl0oO@`l)^s@DC0Pg0}_mVm>;baf?I#Lv1ofqgU4c8{q1=JGcJa?bKlSdz3g<pk7SPE!{jBEyrS)RwWwB-"
    "yU!E6v>OjM~%p<BKh3ZVnewtKTrL-OI(}NFZk_ntHT;oKa<}%zgcH^(x=<vwfTWZLFiis^m!z@XY#o#i-xJsScjU-Hq>+`+D3lYG2Z%gn4T_h$hE{!rk"
    "TF)H|`U+S2@EKG2fxkX@(|}uhQJ}IqJfV#C*!t3H+R<FDA|#7_SO$R}kmKF%JC;f1ePyG2k$Y@if1MYvPX%9ffz+JMpK$LqEoY@k_nw_w!XUR1^GP0}n"
    "kiTjv-?Dja_`3j7zQ{$?j`RlvVvOr##~c4%H|Ln&Y(7G3%i{HN;`tgh6@KjLj_O5d6SzgFUV8-"
    "9~V9Zi>#!c47i#XRKYd_DYji2_=X%A>(I(C3=I=MQHS@5s@Ks=wZ=*Lm&ggJ0YVFM3fIzB^M|5wrdAJa#29?gm$N@Jrs3N&0JoziOAafBk*O@|oYmqs="
    "QzD*UZigTvqjI#rbQTq7_2mnYIUcoqD})R1=8a}QDrc-AmO0WG4xSl7i=6YKdR@U}Y-"
    "J@)C`I)B_L*M39AH%iCQy%cnKW3BA^H`bwg;HM#e=?eX3XW~{kI#g%ZIrI!a(2F>CCoj9m<EJhWx?9|-"
    "a}oFf_~o|FZ2|vo`p|2qow`<$x(qMwqB94;f4#-Iba;zXQ}GWy==aOB&U@hh;`m6N{l=-"
    "~Pi<NN&g&4L4XmeH{ZIw0a%k&hn_A&>(t)2m8Lhpv!u0Sjhs<d<l^*WfUuy+;TVEndU#>Y7{Kcjq`k<Tt^PP7SI5k?`C%g1xY2Uf_uIJK3>Z9vjK4oa&"
    "*8CyVN9y||_+F8|W9HBp)oI|-"
    "<MXDLg3kl^x5Q@B<wyBdu)0^B)7iBbJV&#h9oentQNz?1JrAF5Xc>MZh4{3v_9^@>Tw9l*&m#;qr=R_ed<`O>v1#H}9i5+h%#??DvNGO}IJcc1u6$D*N"
    "~r6b$J%G6Hsf1!L?~t<aj0Ub4R~Gwo@X*X^=5<?qVr438R|2|_ui;g$E((gR1`Yj2Ar=X&hN;Jh+Dyfkvh$IV|;87I`0AJL&160uEDyv)=w#gY-"
    "&P3oWkd>qOV^#BwA^5yVSX;spHl;an!*Q4f%6vf=(CnC=EX3DmZ)sPA-"
    "7EJiisw_TgTgENs^{`nqD^e=~Ks{KOK<Cx1P!VAop4=b;<lbl@J&l~B+NuWTjkN=4j96VFV<X>|ESeQfSk!HjmjrVsf5Za3nyM(r)4F}pn)ant;|uVNk"
    "W^NRlaUhjAnLdWK}G-"
    "U&C{={=qZfl>j+{(Pwsn)x|C%$4P{cmpcbI`9jWN+$F&kr^=z~}smUX3B1soI1o?4(1l0u8;Hh@KLM>*V+Pu}I}><kWTiWNG}>&o|KlaGt(fl#bqX>Ro"
    "N>EI7Pt?XL*Vjf1&$@R(C?iqfaxTNaU*s^Fo<`kcz%%c+Xs^X&rP`z+VMd7J1kT|MAXwRtuTpzi;E!8$vaKfe#xu{RDKTwqf(d}+ZuaEtkim&haY4Lrh"
    "+_3Q3??`#h^KTst`A5vVZM%~Yjukcv=D1!5*i{f?sTetFzGrsPp{1AT?OFZAUkJEt_9@YNW)KTmGQgHsNDL(LHeuY)`Y6|r_7kn>FJj=k-"
    "R~y0=xYnUseGF{@4-=?&Eh=$e^Wv4OhFc%@ni>dhn~|48;IQE<bbgLQ7poa^GS72s-"
    "+0Ccq0e(xP){q+Pl4Ml*L?l;vI>z}u?yUmG2~@@4*lSJ;xOiiNF}4sHHiOQ@Z6g?PbH7-"
    "uA%3Q+v$%|quVj?mo`n?R|Lzw)lb=zY<gv#bIp2sSijrH#HdSxOIIqIdd~em1Gl^BJ1eG2P;0M8y*`;b1m4mxex84KmnovlA3aKlvui?K>K*u>jDC1q6"
    "w|0iUJWQ}S59<)%OBQv<kbK2#WcK(S9@aYYSf52O@3GLXKVi=a_;kJApLTFbfhtO4#FQjFI-"
    "5@M2|L|H8tDXug5y)BGGzBBA+_^?b4;Xrk>$fXEL6J{Q3`$QqT2HW!`D12s*g~er>YwTqIO;slVm@46T{v+h;GBI{4FH;p+L)p=J1uCu4na{tTW+<;bN"
    "{J)P=Z%+P20Yxh6&eBillt2|op1RwIP@pW&H^7NtU!qAnSQL1~@sY>X~iLt(OLlVJr_eQzY5CJKTo&>_9Mf5dKz;oGKVVZf-p@u7MI>LH-fuGhHt?$Po"
    ")Ya)!pH((x;B(94+slII(%GUl%;8enEa)V@@F2c@U@mKaz8KwH<kAuJ@L4P0dxB5!E!98AYa+V+c8Z~_;Q9UU^p)fx|KT{LU*=JdXQmRUmoD_J1@&>kg"
    "Lt_=d6X61z7tQq#}{ozf0~T0C|4GL9j$BEWb*e3UZ<m9ez_`MYu#>bo@eSdJW5M^R+pg8#Ko)6RhL3Xn0nm7cOJ?XbaL>>ym~ppr7Vq1T_%q!xi4!!!p"
    "j9?m4p6kcy?2H21u3fgHhsK;8v98jdiNluZFs!J>PA>Z`;Aw(u{hY9UaDRXS2><Wd4Z_toH(o>-Xs%mH%w&FnZSoo$L&*-"
    ")^p;oAdpZG|aAB1&Kd%Wk)BLrmCeTR|3?0lU*tBKaufc=wjO&-{|JnK;1lU*G_!!bnsaZ{}`I#Th)IQsFHu!b=le<gFIejo-"
    "O0QQHvIV3g2qi=Ayp+gqNW=hJOtW2?<b_zIOe=I)lN<cIv{cKFJ!G=vCWPcD<rM-Oc?zDGhE<7gp;9ZvBrws|NGFgx5{T@03BgwDYV}gAN)hNPO-"
    "w&j`jBBt@xkDwn?d*-"
    "$z1o{zjov#th1^J{b}xBj?h=rB6f{3YI}fRr>RK_%$N?>{x@iKrLP@V4Cdy`J&fnA)v7^uIB^xG(&`u$<s9f3zx8b4foKx&fXe!SlRq*88RrS~bEcCw}"
    "IaVblupzK4D@^khywKI&9j^vC-"
    "C%i6d0QndBnQa=6t%B6Nc;;X@5<Bwp4JeYS2D(@Ay95)Q5s_1)XtSI=oTd|~`F7@bBI#b=bpK^Dl2VLRs^g4QeH$bh4XBYabj@J2})ZygZ_4T<+pzfyF"
    "m5;i=-|F|_(W`C|`h<SIN}}&T4~Me;evA){id5ar4t*$Ls4C-;=wx2TpHGa`Bz!`uB8F^?k7hh0<Ked>^%wf-"
    "m*3Du%a4Ko5%9O=vq%WyP$d2Pd2}@^y1LOp4VxXT?CbngxtvY^gQx4zf4?H=)}~nXs^HS`CZ;x7eHwaroI3yGns_y8=+=eXre?ME^)DCj3uB5GP=TM}?"
    "?+QB;MXR`E2FOs(j_SRu3PVI;F&z8;1j~p$u*k`DAh6gk(Z_#GtX!6Z=J^*wzYuLY;vpCHS`=lj-"
    "xLKFNeO3&#%<z!?3lc?C@mzHQ#gZWoGA9pHVK2EoG`b`u00<`<`1pkvm!&w>xzOymh3nS;stMt-jzLeaC6yncdK7;<M_ib(WxX_En@lWpnCGJauh^?_4"
    "tnIB8ZVO6BG_Rfl@l#o9+69SDQ(hu1`F%6q4Vbi$7fvfeYtE1|0oD@SY53#Yy`H1w2tUNFxdaJW8wlu9mls(zTEZmcH;A7Vr2kD#9!U*UUq+w?p2bS~>"
    "D$vS&Bj8x-"
    "rr@sHireg3Y0i5SYSNn91(dn!%6~=EDMwd=Mp{9cW?Z;y^VS!6OH8)g;ct*jah#2V>KA~r%Tgj8)Be)zwz5BN^zTilVPCg?3M@(G;mx1`Goz^)^A7bQ~"
    "=hn!BCcc$1;`uY{obWC}1?d;+7BZBB@zUTeiSey9BEOznT?$`%n0W58&O2s&^QK50+T_rSXg-"
    "(gyWh0Vdc_yrid4xp4s}dN{lc$yLC=%H;m+>CDzn^Ak>Joteyj7jmC%7ArL99(T-sT|)L`zn3Ho-(+JE|CUJX9)l5@YQf}MQd=U3pq3jPqU$@$&-"
    "bk$S<KD8C&F;(!b)R({UGpTQx>cMz2czeM8u74e`<Z^B`xnODlI4?(?{kuFk8j@cFtGIRGkg3(o-<Et$vCb>G5U*@$-"
    "Kvmc%3z&iSZ6xcS?6(H<?HFvV0?9hA<S=`%K#30gvaRDeNH{yZKxDJY7l<uHu?RO5UKJH9cmh7s3pGCxZ(Rfz31_qIut{_rjEPOt8ef4z8`y&E04YfKe"
    "rkfY7Y)ag2T<=ZNWd$D)I{d&>bJh_$9`Z7%zG~TC>4tLTf`;m}evNWWXm}z7wU0ZCsD;ZKF@Qj{bYdSIC+OMarpP4%u`AU0MylGO*6LZR9JLQ%?@q6w;"
    "skQD?J)!}!#(S{>@r`vgOI$jgkU^k&>|U{IV=X1SE7rJ+{vw=TN+APT&!&8OtoE``uHIab4qWz-"
    "$~+qGq3Dvb_@cLIm_hkvR2*U;5}ALY?&uD>)6zcj_SfAUCltcHK2VwO7;OWaPg&KSm*rL)dvi`0%3hbCn+6v=oR@YaLzehV01>`=D!hU`CsPyFyrKlHF"
    ">uo}+yQ#tVV1^jQf_T7RnCr8I<#qUmfTT^bvdxHmSAI*Gq<FuuLODm?C>e54ML_NJ+5<grauXasyY0xTD<EW!c81G3v|9xFv`dpVPtuw!VFW|7Adj~&h"
    "N9R+<6E4NCH&v{QRP(a+-8pr?PrSxIa;e}-"
    "^Xq$#*TKov;^1gkydv(pG+?}`zI^Un?rREu=U$(D#Mh;H38p%tKdq>1=OU;hLFfSMPlpe&_Vmg`T`q(Vf0ZkzhJJRaOac7)c;Eg!`@mcK>AAG7q*JBv?"
    "LSb*=b>XGz}vtpdDMENQ@hAxl>xrrL$8CYFa2XQ`J+=Q0}RD8eu42H86Q0&M$Jr@_H|``bl3?F2Y|PSi}PsP0jDk|8ag`FI<FDk16QezM5=v=Q)B+L=?"
    "&{_$2!X}9%#?0+Hp?3IAPQG{qO_yL8n8AY0p^Q4s+>qVRVK%Rraaxy!%7zWA)D<ms;X)Pr}nT_=KzYji#gWs@zeRy7e_Q;Rm0O>Pfw**fY1zFLtS4HS#"
    "#ZcMi4nTttqQQ5rVDrGKhYU*PFj_!tO&dL4;S^Jxy9$%Kxf&uxjD-8%mc-RnsobR1lr0#`481GnI+QsH1_`O!}=s@cB&o<5CuW~FbPH#kOT7CII6*-"
    "!&`RGj>J$YcAWv2s3gszQD0Y+viV890ZY{O@M0iUi_I+nQgW!9UD+E!LlRdz_9GbA8>L?-O_{!aUzu=e0TVsYrj9CJi(7Jvu*-"
    "I<mExRKHWamTY(F=dq^ztn(R(^J{!xDOY~gI_}ah15Hg}o*~q!nfRAXq4Ap37Cgrj2WuZLYhU*W>wBYE9X;dJwL|z2@G}4$UZ%gUur-"
    "&CMLCtHlA&?<&L{9F7k%ZgyK}2dbEnoeqMpL*p!>e(c8Xkx){Z0a9ld%@ese#-"
    "zfn(@W{uTl`i1MG4PCJIeP+Bq<82~iH9xIO;|3WjG}w0z*G=&GcyqL}-"
    "*T!|4eIG6DI0kS0_Q6Z=hR~GT<oDu7bl>ptaB>k1Ip&or#epc|J$a?#J?E0y2L#FN`fom+?%>z2tBX$#Cmp{I`Sz-"
    "3!A!hrk3$_Kc98r?TdBp{NOmfSV;fUk@fSrqwq5k<Tom9q@wV>7akf)56&xtt0ZuEe`HSO-tAQSN93LHdaUy$_$k*TQm62HA>quw5s!fGJ;lCeoSIFe?"
    ")WL+bejzB*V<E{E&gG}{ycKPiz0IjRT}F1Juw*GZE75?91Wama24N6zWOs>0Y7l$d5o(6=u|eD8rt8dZynI#d&gom>=^T8rcU+pogMXpzGlg?yowKZX="
    "_PSDc}3{L5jAX2WTI!n()GUAAAD36v6dR>5p2?%`eXmbg8|mnOwgeoYVt9G5PZ=sfX+9`{}#E?RD~HCk{J4=T(04+wZ)g`6GP$t<4Iy`uHdv=;)Nu#?T"
    "T=H{i<)bi=VGkNPig>T+*G<5~Y@@KBQbtr-)mzu?R5;fA)rugdtMrQq|vf1FaY{>C#5E#SV^6SunL>!&~B)Fj!Zvy%-"
    "yfM0_cFUCBdR>W#luuCKP+?FGKdX<14K3E*3sS}(!{n@5G*8U`nuVK7(>OAzV_`v5jMGmy~qm*utmxNm}>h{{HfXw8P&piZw?pZjWPTeo%QbqjvMflYe"
    "{B#0WwHL;!axa%AS2t8+F<O31I$(VtPSbqhokOkLp;wEjm)7?cjF)tVE9#>|kH0s*KKnC~zVf_j?KhTD$8-"
    "B>`Cq9ud<8mr4h*@e*#X(qIMAVm8*PfAPak;AdUqTQwF}qSA03K{Gn9<q$VYx-;6<+ABemrXeykThc7*Tyfi1*8vU+X}ZH8X$HI%^p-"
    "ei0{xN^SDqvPkDvYDpFli#P{0gCGVjTjxzPrWWoUHk8RWLg5{`87`2sE;SUH5Fr>+fLs7>9Y!rkJZHlcob;rfaTknKRv!;{M0BtTISTLHT2QMAr1Exot"
    "7FiJ47pHIP@}=p#yW^Cw_hvzU2*OC*>c9^5-(N7Cv_4eurD}Oo`C1<((Q_lDKhStE}^P;JfF$TsjXQcXc-"
    "O3VwZLJsbGkk7=T`?1NKB(WR~Q)mK<gV6b)mYEI3>cf~~#H~i8@^mZLSWMH9i-OAzAfZuFtwj6!7-"
    "g9T2(F=3v!!C!0%=Uc`nbA6bmg}EP3za|q>gr~j(lbwO>V8r5bF3p&i<&r9_?}HaSikS1&qw&&GKkQiJ=9r`p^|uq{p5HlKDqhyOe)#jPbHe6W5nPc;~"
    "D*|b9pkWL9(9;l(EUt3@m`dEBLER8}jOZ)W`UyrW(?BSH;JEpug|EIYHNg+!}G;)WrI}-{~qgky`95Mqleui6pz;)2FpW-;(Lu{>)uM^$&T}psih-"
    "D@bRo_x<rBKa5URhr3?Ao@ZBM{PcM+`aN}c@bYB6TIyBJ!FE04^I8(OYAw*4TFGjj*Q)~M?Rr&}+Wyem!yDY4OICxH9_2r4s$L_;@r?thYsCg7s$CJcw"
    "pTWl7e9H8@fqk?)xq&<R^O#1=-"
    "xr@%kzf+1Fn`9O3>MPjQ@iUSnm_PLnqO(nz@VVyJWXkxJ>=tkUTNI7JPQgQdITn3l933n%0!w?w#*F#0fJCX|ltucJ%w6mgM8TbtXK#f0Lkjr(Mdo5r6"
    "f)@7#yvP-`F2IIW9xseK(olZeYl)|mmk6&f9@Q|QAB`lCGLCEZ8g^T74G71DC2Tia3$)uewen>v-;`SJS)i|Ecew^E%nbTgUHO_NFk@t>(WBrA4?N0-"
    "u=+C$vhk>7#zJO9Qe>Yo=LJ=tbzO?7HJdiA9SxVTqRC5m~q@rtSDtbfz5`ucs?!>$F@DbAxBE$phs=YE5C`G~&0Y*kzz{r%N!Kl!RDx#`W8zz=)H5^^;"
    "1SHyX{=98zPTtAHd=2^p18oSzGwu^RUE^qC-M^1)W&sdk$i>3j}c*U;YnSTy-tyD+rec}>YKHjU(ee8-PFU!E;rMgn%MkVxYs7K8WyGDbLBk-"
    "<vl5~u`92xG`-ASe{pr4hf3&YTvDM9)4w22G<f^Ohv##0yS#98M;#;eRsm*U9dALMb(BeV_wlYVItJ(=v*Z0cGo=0DANXLyl!Kw-4Nts60>GJx*~kEO-"
    "=E#4GVu4wv1yQx9#&}MYI1@m80K{fg8(!jqA{f)k5vi5(YzC5~`Pl4aLGy~uA20Uct`bm-2K2h-+(%mIjKSR^_+)U5O4fAK)QdnUF+^T-"
    "p`1(7`7;u}2&z}@gT!rg;G(L=VmIM3X@Bln(`l_rRw(_ca8&ki6<M-&*9s1YiyKAZL`vCo~ja`|*$9?#<7@rbeznXGp3)JaBb{zu`x3-"
    "d7@=|Mf38mZMuYNb}>I_d8Zih$K_k_cfl+rpte?7D7A9(uk7r2dI_FtQ%Bk%pS<g#7ESkFrE(}%pD3@WFI$pJcc6260zi6_uicp0#yl)hc<)wezD`t;x"
    "X1=c<d=MR@u>D3-"
    "3I_&xx{I{h~8CnM4ols2ombhh`X?{I7Hrm?b4F9k!I={v)b*X+1Q|H>_S<r{p#4Wjef+qayQq6FDOJm=DEGB(o@W^6voO0_<O;c~aMQfi}@AA^yoGq%N"
    "Tin!OQ**(CwTDz1@@_;H)v>8=wYBz*vG&0vFR%FAlv{;#Bhjr>|I!z8Klz^fepkJbn4rVgUAn!&P=D~y172@|zb`u%)ap+z{kYmtXY?faCHIT3Zh9_JG"
    "tzie#^2Nu;uiM-%<<>${N*$_*{gMpO$D*;9q7p=^5ZC4RiiTcD>XQeW#0MV{31Sdc6c+*^bgY5$@Z`NQd-YB@5NtSEm2o%ZU(Amid{Lt$1wP4y|>*pcN"
    "tac6rhd|>|dV`{$&sKoqE!+ZY4eF5~vSv>?+DSyoVX1Z|iiRoFdZ&DEt?@wy=&}=zIe4KK8D(wrxSrJJ^*C{ci=XLeS$*Dx)f1uV!Voe?4!vCOG_-JP!"
    "D?nDVW0<5x{RByMZqqZ3`-"
    "{IG!fUU4aFtf|4o`8Iy3ly&av!9q$Z?N+)(Obj~kj680nUk>hBT!Y*mU2bNoT?2Ud)OQZZ^&7>tBArJ+)iqV>8{dA0?ZNxbsl}D*lUvz}nmSD0KarPf^"
    "hcd57t@)wZhdl^$^-9Slb0dn_xEB&w7Z2{4=&Rug2N<qdm8=M1{Kx$L2mWDU}z`0nddcerXMdeD@g}-"
    "dvvUX`L&<7&gH35#X7&NvVxC!6*a(A2>H&)ct?2rxJOM5=;N;ei|Ny;Po>b|2juHmkCqBb3{v<kbT6;(_nM>hH|6g((zE_SD!0zAs_5O-"
    "15#gjn#W#2>9+)^4|-"
    "b>UEP8|Tw(2rR<)e&AMn?sMRpCTWZ@fKLRT+*S5{dw`)he;^tp!bcga@ZaY2h@Rm<X4pR{%bldpfk)nYzp`IJO0T8~kjfUdUo?Z@aMFL8AW>0`KC9mq?"
    "36W@EXhl*MIpcK`lm2Q2hZOW{J29TFX>e8q_rR3Mvqxi0-"
    "mV=Yg<fTU$YtQ}C`fH#^Z>)W`iAxLOe~I|D^Oja(2jW?s&qWvN692x;A345+c4YMEj|fwx$>TZ3CxQ3YzZ6rZU2bJ~Vdxz9<0LP?vCb+%#Z~K(TTySxB"
    "k^gCes)FAGF_~o_I<rNLqC!L4jiATv(%T~%j>DbPJhi^ZR!E>&&>Ej=JWpktxDDkP`lquwWNRhi2fuI=e2#>Xh^vr6`N+)K6HP?4XFuz!_>%cHLP=x>a"
    "M^S5&yjKr7yab@TQEu#QST}0K2kR_dQSL1b0t&CMm}tuPWf9tmn_F;Xfx6r!Qxdv=TjZJTMiAj+LW-sX|^B&n%&2bZq5NQx(Y9RpLJgd?hX|s>l&;y{u"
    "sBAiVfsea}|H+Q+iER()~nK7HBaYQA%(LW#%Yzsu<Jevjt$HC4N+j~BHV|MW6hp?5rAf3KO&^)0M346O4;izlnmF^|53cgI;@G2)y8@9NJjEwi6T58zi"
    "P>--+Z>$9F0&q`=^q({5ZxBXn-"
    "j`;UuJwNOzt%{>P8XjUQ6Z6(!9BGs1c4cL|;FU4Z)Ftv_ok{RL@re7bo=z<A*OL@e`N{V(#*cxoZ@M&4yL0|pw#d|9?o>L|&$n;S&-"
    ">Hq^IYHl!*i_tpNZ!x>TqK8H|lgWQ1wRHb&LN$68smX{tOsgQuj`HR3x=s%ZOV>@_UCkG%rA%3ifE~Pv+P2C#-W+dt2uXbuX!i-"
    "`&(xQ;X4=h184Y)aAXqimTiW{9!dyYgxAgJpT_oFAXfMbk#kYj<1f#ht*{~tMz=^=5iWf_bL>9r~^(Ch;!zO))|N8<rnHzwLYdUa(yo1xfDHqQ>L8$wR"
    "u&nHP?fme_s2}XV3Gpj1tIWhjOOIQfKSD1^>i(KP3KTnn$H`n!1J_y(Z3oz`K(3%j%Nh)%N11-2A;T{-"
    "Fu~J|MiR&RqBE0X%JspRTD?`XBl4@w~RSR`*v3d1*>r9L)GZKEGqyx=QHjuQ`KFU7>FuKs~L3uPE?s2UYzNsH*79JZnEM^kfQt$FF!3rFt5uvi<G)rk-"
    "z~tIDkZuPLRq&*{<bPWUL|_7^y;h^}5JnyhXy^cVF^b>{lz7T?kH)%Mc*@XW25MbK5QpFkX5T6Jt*lB$+QPm)c&Cyp`1p+9lU|FVJt-"
    "+8D_=sfp%mg|?5B|bUID+f4l-_X<m#_JHDv*5AhopM@s*Q0^>ggcB^W}a@uY0#W<s{4-z4K|e%yiOx-_gUv(J<4hOEsuW9Ysw%lsj1gLf|J^<$|`iPN9"
    "og>>cM)hp#OE5XVQv_YE54=5<fW(eEf-@FTr|#*jG*G$9Yw;kg1{k{~WedkRIQ9xVoOL@oH%i-##P*!Os)$d)-"
    "k@CmZ`~?KM*qsPDbe=lP8PFtdUZvZ4R~n0jKJtHu4cW}WZmmDPBB<Hqvf3cY*4{oaJ9W1Z;FQ@3_xH&u}TUxfQDK%5WuOj6M>kA_8>N(T<-"
    "S^I#o{vB&7XyGM~?&Fsx5r+lL)06SuBg!fBZyrSx|Mpy8o_Ssn&+`w<E7g0CUh}zIS?6Nrslt5~c~DWG3w!lVMN`H3-"
    "1l7H5uHE(xU{A<_9)Ls<Lh%2D_Q?S^x#kUrGMSpd&*D*JpGpIL%^fo){=_K<xye$_I*pQzDNzYueR%w^ydhV2HiLGJ?lKhcys1A50p_EbT#KALo<j&UG"
    "#7#dC2xUQRh?epUq8Gr|u^_@V!U#IjE=}C42Ol@lMpcW#DiMzGZoqLQ3r8R>8-"
    "HDxeR;Sm!0`&X>NB=MT4jvYA?4)3@L9YUY`|tAv8O!pGdE3Rm%+*|v<hoywe~zBfEt0w06G$!_9v7Chu_o~+<pUgd+=C%LZ@;A)noKOal!^>&Z4X99=R"
    "<wyTL&-S)?f*Lh<%f80Y%0BSoj&Faz8iQjLH4A^$%b+Ls?RT_*>)%f*pfgR~YO>V$`W)d^)_D)TEI&Fywa-"
    "#de>N1xeZ6CyBhlRn^gF3c@N)p)i|<_YH+jStb$A-3=9%3(@x_oyUF<_0jAWh<jbe2u!mYkuQ*p$3JM-"
    "@*{u_D~(}0{F707DpDLA>t{O5>6yQQTy;JQa2z*Q7DDfP~`??=6+$(mErt2gj58}Zpd9L|HQrk-RS%H!3OV)%UG?0xU+k5(5esiE}0^K8D~i)*6KFR9P"
    "x@)gjw`fhDpWaut^>}>kSaO(N^YSG%!-"
    "=*bk@pbsmhQxD<b)LYk1dSSr&Mz~R$?~<VCo6dmY*J7o@4B^To1yE}yKQ`~r=ay+b77qd@hB7VNsSJEkFWcGN#`CEWf_I>XWbWM_xlzVQ&3rvRFDKjaU"
    "ldWFL;TJnQmSh6{EDgjWWrQT^%oh*BZQ}h-fGdj-?<IDNQD0j#PL<t;{jZCZ|yo6)Tgn-"
    "@;!zbM~F@{mwbhdCqfo=X<|+u2b1HR)^5B)2|!V5}&3k%*w<+YGj;}EL#=e(0`7=H2>Li;Iv!TIF&a9>wHdtEaFm&{UZG#<JMSJZl^E6hqK|$%dKkr9p"
    "%?&#w)ZeSYGO@3;c7G^(W!OQ;j23Q|(c>gL-FfjUx^v?7M`#_nczu(pICt-|@dM6HOCE4F^7^=@TqvO*G2K-<9O+WAa`$Cq*03&mq-DH;D5i)X@%bUYR"
    ">glc&O~M~$9m-"
    "(K{E1fYMN`Ski)kCuO9RL^y;QZFsU=QQzjcs;Tn7(M$E+6Vq0jeX%HpSn}m=?c&R{EhLSnTucbV={DLbFhX#86Y=&k%PZO`5fiVP}0m`g}}G|b6)!6??"
    "&=I_JiSCR_Z~=7~RL;&%k*z{uVr!sKOju7d9KYh<_3N$S^-0Ul^xdwU$EWo3=S+h5wu-"
    "@_XdpB%McJm6jXbrth@)&nzPjGm4W{80pc@N~10GopSWyOyaO;LyAVhSAky``MAy&@N*WwR@4sB{LUWL{AHM%;Q{8F&#|v+PGlyvHPO+|pRwHkHS*u=9"
    "igx$kAhqQYK0H$sf$bSSoW1v_0IO_9Q>AoUzy}F99;SKCem+hUHh>8_wd!1>9yo>o-0OGsg^E`G>S!c{>Z-DV#)vR1ijzWR^Nq2-"
    "@<P(<aeJxK1YVAy2Muf3Zql#>I3W>NnO`w4%J7*;of?qzRa^%Z{i#F-Z$gbqs&%SBY2oge9%i@VL$Urg3{O7YQ1dqkN<r|<oyo*?s_j--^-"
    "()4jVn#EPdAip8LQn>mF6kQnyBSGfF*%4%th+gpnV2FO>{-DfO^Zb%oM^i{#S(KlQOFb-"
    "3f!zEIQl{J?ztnoIo_&l#wxXDlrqW0cEWdWiThCchmXAEaYxw#L0^wEkUbAo2Nafd9Q(Q7SI8WUXOe?Ej$yUxg<_AI50Vr<Mw~fPZ@F!!_(3%xvQh1*;"
    "uZmg3Z&MrpwgVjY5CvwCRjo*+f%I5puQ^>dh7<o@lWUe#@Psd&Fr1>4bY%w@N|aPGDURZVv5WRp|n%dk^-XYs4~Zht)-"
    "WNBBB(GQ=%4Yg7o>(dgvD(vT0Y_w5eY5Q|Ag%RNYM5H<_cdK!@QPBmoMhSW6e%!5LGDSi1J?&IvIlY(peRU5wF`|!>Zn^Yer&ISf5XW;;F4qs=AF0v;w"
    "-Qe{wFvxNg%6*Kq<2<DX?Lupir<}D`86{^1y~Ga7WMU1><pJiU2^KpL#(X^TVej^ZoFE$+a*u4Q>*-E=M(4d_|>vDTG1UXWwtoAWS*2nUYgN~c59sa-"
    "L*8MpV6P$VElgj`_sqvP1GN6*~$aAn_iO|;D_`$X-rbQw$37to2Xx|6G7hd;qxrdVEvqHYgh&Kh#r{caA+^<$M=a>+L!dfEO;0`iR<i8F7-"
    "PnE=?P2==*P@2jGnw*5`xI7wU$n{B`un8Kb%Ic_->Iin{g{CF}Tfk8U-YHr=1!#i1(f4L8yiao3~D)kdeWzhOBv3I17DlCIRuV9kcdY<Ox4x-"
    "ONv&MqFUwX=d16AnN30#n#$<KGj$Jaz39qMI3kZRa&iqOJ!lU_KrFvO<$d=qrJ;!DSwOBqERRBd6(LP*;sxAE^HUmhMd2"
)


def _decode_nlh_92_f64() -> tuple[np.ndarray, np.ndarray]:
    """Decode symmetric (92, 92, 3) float64 arrays (NLH_AA, NLH_BB) for Z = 1..92 (b in A^-1)."""
    raw = zlib.decompress(base64.b85decode(_NLH_TRIU_I32_ZLIB_B85))
    triu = np.frombuffer(raw, dtype="<i4").astype(np.float64).reshape(4278, 6) / 1e5
    aa = np.zeros((92, 92, 3), dtype=np.float64)
    bb = np.zeros((92, 92, 3), dtype=np.float64)
    iu = np.triu_indices(92)
    aa[iu] = triu[:, :3]
    bb[iu] = triu[:, 3:]
    iu1 = np.triu_indices(92, k=1)
    aa[iu1[1], iu1[0]] = aa[iu1]
    bb[iu1[1], iu1[0]] = bb[iu1]
    return aa, bb


def _so3lr_switch_np(r: np.ndarray, r_off: float = DEFAULT_NLH_R_OFF_A) -> np.ndarray:
    """NumPy reference for SO3LR C^inf switching_fn(r, 0.0, r_off)."""
    c = np.asarray(r, dtype=np.float64) / float(r_off)
    out = np.zeros_like(c, dtype=np.float64)
    out[c <= 1e-3] = 1.0
    m = (c > 1e-3) & (c < 1.0 - 1e-3)
    if np.any(m):
        cm = c[m]
        z = (1.0 - 2.0 * cm) / (cm * (1.0 - cm))
        out[m] = 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))
    return out


def _solve_rmin_matrix_np(
    c_pref_kcal: np.ndarray,
    bb_full: np.ndarray,
    floor_energy_ev: float = DEFAULT_FLOOR_ENERGY_EV,
    r_off: float = DEFAULT_NLH_R_OFF_A,
    with_switch: bool = True,
) -> np.ndarray:
    """Solve [128, 128] pair onset r_min (A) where E_NLH(r_min) = floor_energy_ev via 64-step float64 bisection."""
    target_kcal = float(floor_energy_ev) * EV_TO_KCAL
    hi_init = float(r_off) if with_switch else 4.0
    lo = np.full((128, 128), 0.01, dtype=np.float64)
    hi = np.full((128, 128), hi_init, dtype=np.float64)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore", invalid="ignore"):
            e_kcal = np.sum(c_pref_kcal * np.exp(-bb_full * mid[..., None]), axis=-1) / mid
            if with_switch:
                e_kcal = e_kcal * _so3lr_switch_np(mid, r_off=r_off)
        above = np.nan_to_num(e_kcal, nan=0.0) > target_kcal
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    rmin = 0.5 * (lo + hi)
    rmin[0, :] = 0.0
    rmin[:, 0] = 0.0
    return rmin


def _compute_c2_jet_np(c_pref_kcal: np.ndarray, bb_full: np.ndarray, d_off: np.ndarray) -> np.ndarray:
    """Compute [16384, 4] table [d_off, V(d_off), V'(d_off), V''(d_off)] in kcal/mol, A^-1, A^-2."""
    d_safe = np.where(d_off > 0.0, d_off, 1.0)
    ex = np.exp(-bb_full * d_safe[..., None])
    f0 = np.sum(c_pref_kcal * ex, axis=-1)
    f1 = -np.sum(c_pref_kcal * bb_full * ex, axis=-1)
    f2 = np.sum(c_pref_kcal * (bb_full * bb_full) * ex, axis=-1)
    v0 = np.where(d_off > 0.0, f0 / d_safe, 0.0)
    v1 = np.where(d_off > 0.0, (f1 - v0) / d_safe, 0.0)
    v2 = np.where(d_off > 0.0, (f2 - 2.0 * v1) / d_safe, 0.0)
    return np.stack([d_off, v0, v1, v2], axis=-1).reshape(16384, 4)


def _build_tables_f64() -> tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray, np.ndarray]:
    """Build float64 CPU flat lookup tables for Z in 0..127 (pair_idx = (zi << 7) | zj, shape [16384]):
      - cb_flat [16384, 6]: [COULOMB_K * zi * zj * a0, ..., a2, b0, b1, b2] (0 on z=0 rows/cols)
      - rcov_sum_flat [16384]: r_cov(zi) + r_cov(zj) with Z >= 19 capped at 0.75 A (0 on z=0)
      - rmin_default_flat [16384]: SO3LR r_min(5.0 eV, r_off=1.5 A) in A (0 on z=0)
      - jet_default_flat [16384, 4]: [d_off, V(d_off), V'(d_off), V''(d_off)] for d_off = r_min(5 eV) + 0.05 A
    """
    from ase.data import covalent_radii

    aa92, bb92 = _decode_nlh_92_f64()

    rad = np.array(covalent_radii[:128], dtype=np.float64)
    if rad.size < 128:
        rad = np.concatenate([rad, np.full(128 - rad.size, MAX_HEAVY_ZBL_COV_RAD_A, dtype=np.float64)])
    rad = np.where(np.isfinite(rad) & (rad > 0.0), rad, MAX_HEAVY_ZBL_COV_RAD_A)
    rad[19:] = np.minimum(rad[19:], MAX_HEAVY_ZBL_COV_RAD_A)
    rad[0] = 0.0

    rcov_sum = rad[:, None] + rad[None, :]
    rcov_sum[0, :] = 0.0
    rcov_sum[:, 0] = 0.0

    aa_full = np.zeros((128, 128, 3), dtype=np.float64)
    bb_full = np.zeros((128, 128, 3), dtype=np.float64)

    z_idx = np.arange(1, 128, dtype=np.int64)
    z_clamped = np.minimum(z_idx, MAX_Z_NLH_TABLE) - 1  # 0..91
    zi_c, zj_c = np.meshgrid(z_clamped, z_clamped, indexing="ij")
    aa_sub = aa92[np.minimum(zi_c, zj_c), np.maximum(zi_c, zj_c)]
    bb_sub = bb92[np.minimum(zi_c, zj_c), np.maximum(zi_c, zj_c)]

    # Smooth Thomas-Fermi Z^0.23 screening-rate continuation for transuranics Z > 92 (exact 1.0 for Z <= 92)
    z_flt = z_idx.astype(np.float64)
    z_flt_c = np.minimum(z_flt, float(MAX_Z_NLH_TABLE))
    tf_scale = (z_flt[:, None] ** 0.23 + z_flt[None, :] ** 0.23) / (
        z_flt_c[:, None] ** 0.23 + z_flt_c[None, :] ** 0.23
    )
    bb_sub = bb_sub * tf_scale[..., None]

    aa_full[1:, 1:] = aa_sub
    bb_full[1:, 1:] = bb_sub

    z_all = np.arange(128, dtype=np.float64)
    z_prod = z_all[:, None] * z_all[None, :]  # [128, 128], 0 on row/col 0
    c_pref_kcal = (COULOMB_K * z_prod)[..., None] * aa_full  # [128, 128, 3]

    rmin_5ev = _solve_rmin_matrix_np(
        c_pref_kcal,
        bb_full,
        floor_energy_ev=DEFAULT_FLOOR_ENERGY_EV,
        r_off=DEFAULT_NLH_R_OFF_A,
        with_switch=True,
    )
    d_off_5ev = np.where(rmin_5ev > 0.0, rmin_5ev + DEFAULT_FLOOR_WIDTH_A, 0.0)
    jet_5ev = _compute_c2_jet_np(c_pref_kcal, bb_full, d_off_5ev)

    cb_flat = torch.from_numpy(np.concatenate([c_pref_kcal, bb_full], axis=-1).reshape(16384, 6)).contiguous()
    rcov_sum_flat = torch.from_numpy(rcov_sum.reshape(16384)).contiguous()
    rmin_5ev_flat = torch.from_numpy(rmin_5ev.reshape(16384)).contiguous()
    jet_5ev_flat = torch.from_numpy(jet_5ev).contiguous()
    return cb_flat, rcov_sum_flat, rmin_5ev_flat, jet_5ev_flat, c_pref_kcal, bb_full


_CB_FLAT_F64, _RCOV_SUM_FLAT_F64, _RMIN_5EV_FLAT_F64, _JET_5EV_FLAT_F64, _CPREF_NP_F64, _BB_NP_F64 = _build_tables_f64()
_TABLE_CACHE: dict[tuple[torch.device, torch.dtype], tuple[Tensor, Tensor, Tensor, Tensor]] = {}
_CUSTOM_RMIN_F64_CACHE: dict[tuple[float, float, bool], Tensor] = {
    (DEFAULT_FLOOR_ENERGY_EV, DEFAULT_NLH_R_OFF_A, True): _RMIN_5EV_FLAT_F64.view(128, 128)
}
_CUSTOM_RMIN_DEV_CACHE: dict[tuple[torch.device, torch.dtype, float, float, bool], Tensor] = {}


def nlh_rmin_table_f64(
    floor_energy_ev: float = DEFAULT_FLOOR_ENERGY_EV,
    r_off: float = DEFAULT_NLH_R_OFF_A,
    with_switch: bool = True,
) -> Tensor:
    """Return [128, 128] float64 CPU lookup table of pair onset r_min (A) where E_NLH(r_min) = floor_energy_ev."""
    key = (float(floor_energy_ev), float(r_off), bool(with_switch))
    tab = _CUSTOM_RMIN_F64_CACHE.get(key)
    if tab is None:
        rmin_np = _solve_rmin_matrix_np(
            _CPREF_NP_F64, _BB_NP_F64, floor_energy_ev=key[0], r_off=key[1], with_switch=key[2]
        )
        tab = torch.from_numpy(rmin_np).contiguous()
        _CUSTOM_RMIN_F64_CACHE[key] = tab
    return tab


def nlh_coefficients_f64() -> tuple[Tensor, Tensor]:
    """Return (NLH_AA [92, 92, 3], NLH_BB [92, 92, 3]) in float64 on CPU (exact Nordlund-Lehtola-Hobler table)."""
    aa92, bb92 = _decode_nlh_92_f64()
    return torch.from_numpy(aa92), torch.from_numpy(bb92)


def _nlh_tables(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (cb_flat [16384, 6], rcov_sum_flat [16384], rmin_5ev_flat [16384], jet_5ev_flat [16384, 4]) cached."""
    key = (device, dtype)
    tabs = _TABLE_CACHE.get(key)
    if tabs is None:
        tabs = (
            _CB_FLAT_F64.to(device=device, dtype=dtype),
            _RCOV_SUM_FLAT_F64.to(device=device, dtype=dtype),
            _RMIN_5EV_FLAT_F64.to(device=device, dtype=dtype),
            _JET_5EV_FLAT_F64.to(device=device, dtype=dtype),
        )
        if not torch.compiler.is_compiling() and not isinstance(tabs[0], torch._subclasses.FakeTensor):
            _TABLE_CACHE[key] = tabs
    return tabs


def _nlh_rmin_flat(
    device: torch.device,
    dtype: torch.dtype,
    floor_energy_ev: float = DEFAULT_FLOOR_ENERGY_EV,
    r_off: float = DEFAULT_NLH_R_OFF_A,
    with_switch: bool = True,
) -> Tensor:
    if (
        float(floor_energy_ev) == DEFAULT_FLOOR_ENERGY_EV
        and float(r_off) == DEFAULT_NLH_R_OFF_A
        and bool(with_switch)
    ):
        return _nlh_tables(device, dtype)[2]
    key = (device, dtype, float(floor_energy_ev), float(r_off), bool(with_switch))
    tab = _CUSTOM_RMIN_DEV_CACHE.get(key)
    if tab is None:
        tab = nlh_rmin_table_f64(key[2], key[3], key[4]).reshape(16384).to(device=device, dtype=dtype).contiguous()
        if not torch.compiler.is_compiling() and not isinstance(tab, torch._subclasses.FakeTensor):
            _CUSTOM_RMIN_DEV_CACHE[key] = tab
    return tab


def _so3lr_switch_and_grad(
    r_eval: Tensor, valid: Tensor, valid_f: Tensor, r_off: float, r_cut_envelope: float
) -> tuple[Tensor, Tensor]:
    """SO3LR C^inf switch w(r; 0, r_off) [optionally * PhysNet polynomial cutoff f_cut(r; r_cut_envelope)] and d/dr."""
    inv_roff = 1.0 / float(r_off)
    c = r_eval * inv_roff
    in_sw = valid & (c > 1e-3) & (c < 1.0 - 1e-3)
    c_s = torch.where(in_sw, c, torch.full_like(c, 0.5))
    omc = 1.0 - c_s
    sig = torch.sigmoid((1.0 - 2.0 * c_s) / (c_s * omc))
    w_sw = (
        torch.where(c <= 1e-3, torch.ones_like(c), torch.where(c >= 1.0 - 1e-3, torch.zeros_like(c), sig)) * valid_f
    )
    dw_sw = torch.where(
        in_sw,
        -(sig * (1.0 - sig) * inv_roff) * (1.0 / (c_s * c_s) + 1.0 / (omc * omc)),
        torch.zeros_like(c),
    )
    if r_cut_envelope > 0.0:
        inv_rc = 1.0 / float(r_cut_envelope)
        xc = (r_eval * inv_rc).clamp(0.0, 1.0)
        xc2 = xc * xc
        f_cut = torch.where(
            r_eval < r_cut_envelope,
            1.0 - (xc2 * xc) * (10.0 + xc * (-15.0 + 6.0 * xc)),
            torch.zeros_like(r_eval),
        )
        omx = 1.0 - xc
        df_cut = torch.where(
            valid & (r_eval > 0.0) & (r_eval < r_cut_envelope),
            (-30.0 * inv_rc) * xc2 * (omx * omx),
            torch.zeros_like(r_eval),
        )
        return w_sw * f_cut, dw_sw * f_cut + w_sw * df_cut
    return w_sw, dw_sw


def nlh_switch_so3lr(r: Tensor, r_off: float = DEFAULT_NLH_R_OFF_A) -> Tensor:
    """Differentiable SO3LR C^inf switching function w(r; 0, r_off): 1 for r <= 0, 0 for r >= r_off."""
    c = r / float(r_off)
    in_sw = (c > 1e-3) & (c < 1.0 - 1e-3)
    c_s = torch.where(in_sw, c, torch.full_like(c, 0.5))
    sig = torch.sigmoid((1.0 - 2.0 * c_s) / (c_s * (1.0 - c_s)))
    return torch.where(c <= 1e-3, torch.ones_like(c), torch.where(c >= 1.0 - 1e-3, torch.zeros_like(c), sig))


def nlh_phys_cutoff(r: Tensor, r_cut: float) -> Tensor:
    """PhysNet polynomial cutoff 1 - 10 x^3 + 15 x^4 - 6 x^5 for x = r / r_cut < 1, else 0."""
    xc = (r / float(r_cut)).clamp(0.0, 1.0)
    xc2 = xc * xc
    return torch.where(r < r_cut, 1.0 - (xc2 * xc) * (10.0 + xc * (-15.0 + 6.0 * xc)), torch.zeros_like(r))


def nlh_pair_energy(zi: Tensor, zj: Tensor, r_safe: Tensor) -> Tensor:
    """Analytic bare NLH screened nuclear Coulomb energy E_NLH(r) in kcal/mol for pair (zi, zj) at r_safe > 0 (A)."""
    cb_flat, _, _, _ = _nlh_tables(r_safe.device, r_safe.dtype)
    z1 = zi.to(device=r_safe.device, dtype=torch.long).clamp(0, 127)
    z2 = zj.to(device=r_safe.device, dtype=torch.long).clamp(0, 127)
    p_idx = ((z1 << 7) + z2).reshape(-1)
    ab = cb_flat.index_select(0, p_idx).view(*r_safe.shape, 6)
    c0, c1, c2, b0, b1, b2 = ab.unbind(-1)
    num = c0 * torch.exp(-b0 * r_safe) + c1 * torch.exp(-b1 * r_safe) + c2 * torch.exp(-b2 * r_safe)
    return num / r_safe


class _NLHTanhFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r: Tensor, r_min: Tensor, width: float) -> Tensor:
        w = float(width)
        ctx.width = w
        d_off = r_min + w
        active = (r < d_off) & (r_min > 0.0)
        u = torch.where(active, (d_off - r) / max(w, 1e-12), torch.zeros_like(r))
        th = torch.tanh(u)
        ctx.save_for_backward(th, active)
        return torch.where(active, d_off - w * th, r)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        th, active = ctx.saved_tensors
        dr = torch.where(active, 1.0 - th * th, torch.ones_like(th))
        return grad_out * dr, None, None


def nlh_floor_tanh(r: Tensor, r_min: Tensor, width: float = DEFAULT_FLOOR_WIDTH_A) -> Tensor:
    """Smooth SO3LR Tauri Eq. S5 descriptor floor: r for r >= r_min + w, else (r_min + w) - w tanh((r_min + w - r)/w)."""
    return _NLHTanhFloor.apply(r, r_min, width)


class _NLHClampDense(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dist: Tensor,
        zc: Tensor,
        pmf: Tensor,
        scale: float,
        r_in_ratio: float,
        mode: str,
        floor_energy_ev: float,
        width: float,
    ) -> Tensor:
        ctx.mode = mode
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        p_idx = ((zc.unsqueeze(2) << 7) + zc.unsqueeze(1)).reshape(-1)
        if mode == "tanh":
            rmin_flat = _nlh_rmin_flat(zc.device, dist.dtype, floor_energy_ev=floor_energy_ev)
            r_min = rmin_flat.index_select(0, p_idx).view_as(dist)
            w = float(width)
            d_off = r_min + w
            active = valid & (dist < d_off) & (r_min > 0.0)
            u = torch.where(active, (d_off - dist) / max(w, 1e-12), torch.zeros_like(dist))
            th = torch.tanh(u)
            ctx.save_for_backward(th, pmf, active)
            return torch.where(active, d_off - w * th, dist) * pmf
        _, rcov_flat, _, _ = _nlh_tables(zc.device, dist.dtype)
        rcov_ij = rcov_flat.index_select(0, p_idx).view_as(dist)
        r_out = float(scale) * rcov_ij * valid.to(dist.dtype)
        if mode == "tanh_c3_matched":
            w_t = ((1.0 - float(r_in_ratio)) * r_out).clamp_min(1e-12)
            active = valid & (dist < r_out) & (r_out > 0.0)
            u = torch.where(active, (r_out - dist) / w_t, torch.zeros_like(dist))
            th = torch.tanh(u)
            ctx.save_for_backward(th, pmf, active)
            return torch.where(active, r_out - w_t * th, dist) * pmf
        if mode == "c3":
            r_in = float(r_in_ratio) * r_out
            w_c3 = (r_out - r_in).clamp_min(1e-12)
            u = ((r_out - dist) / w_c3).clamp(0.0, 1.0)
            u2 = u * u
            u4 = u2 * u2
            ig = u - u4 * (2.5 + u * (-3.0 + u))
            clamped = r_out - w_c3 * ig
            ctx.save_for_backward(dist, r_in, r_out, pmf)
            return torch.where((dist >= r_out) | (r_out <= 0.0), dist, clamped) * pmf
        raise ValueError(f"Unsupported nlh_clamp_dense mode {mode!r}; expected 'tanh', 'tanh_c3_matched', or 'c3'")

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        if ctx.mode in ("tanh", "tanh_c3_matched"):
            th, pmf, active = ctx.saved_tensors
            dr = torch.where(active, 1.0 - th * th, torch.ones_like(th))
            return grad_out * dr * pmf, None, None, None, None, None, None, None
        dist, r_in, r_out, pmf = ctx.saved_tensors
        w_c3 = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - dist) / w_c3).clamp(0.0, 1.0)
        u2 = u * u
        dr = torch.where(
            (dist >= r_out) | (r_out <= 0.0),
            torch.ones_like(dist),
            torch.where(dist <= r_in, torch.zeros_like(dist), 1.0 - u2 * u * (10.0 + u * (-15.0 + 6.0 * u))),
        )
        return grad_out * dr * pmf, None, None, None, None, None, None, None


class _NLHClampSparse(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dist: Tensor,
        zc: Tensor,
        recv: Tensor,
        send: Tensor,
        scale: float,
        r_in_ratio: float,
        mode: str,
        floor_energy_ev: float,
        width: float,
    ) -> Tensor:
        ctx.mode = mode
        zr, zs = zc.index_select(0, recv), zc.index_select(0, send)
        valid = (zr > 0) & (zs > 0)
        p_idx = (zr << 7) + zs
        if mode == "tanh":
            rmin_flat = _nlh_rmin_flat(zc.device, dist.dtype, floor_energy_ev=floor_energy_ev)
            r_min = rmin_flat.index_select(0, p_idx)
            w = float(width)
            d_off = r_min + w
            active = valid & (dist < d_off) & (r_min > 0.0)
            u = torch.where(active, (d_off - dist) / max(w, 1e-12), torch.zeros_like(dist))
            th = torch.tanh(u)
            ctx.save_for_backward(th, active)
            return torch.where(active, d_off - w * th, dist)
        _, rcov_flat, _, _ = _nlh_tables(zc.device, dist.dtype)
        rcov_ij = rcov_flat.index_select(0, p_idx)
        r_out = float(scale) * rcov_ij * valid.to(dist.dtype)
        if mode == "tanh_c3_matched":
            w_t = ((1.0 - float(r_in_ratio)) * r_out).clamp_min(1e-12)
            active = valid & (dist < r_out) & (r_out > 0.0)
            u = torch.where(active, (r_out - dist) / w_t, torch.zeros_like(dist))
            th = torch.tanh(u)
            ctx.save_for_backward(th, active)
            return torch.where(active, r_out - w_t * th, dist)
        if mode == "c3":
            r_in = float(r_in_ratio) * r_out
            w_c3 = (r_out - r_in).clamp_min(1e-12)
            u = ((r_out - dist) / w_c3).clamp(0.0, 1.0)
            u2 = u * u
            u4 = u2 * u2
            ig = u - u4 * (2.5 + u * (-3.0 + u))
            clamped = r_out - w_c3 * ig
            ctx.save_for_backward(dist, r_in, r_out)
            return torch.where((dist >= r_out) | (r_out <= 0.0), dist, clamped)
        raise ValueError(f"Unsupported nlh_clamp_sparse mode {mode!r}; expected 'tanh', 'tanh_c3_matched', or 'c3'")

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        if ctx.mode in ("tanh", "tanh_c3_matched"):
            th, active = ctx.saved_tensors
            dr = torch.where(active, 1.0 - th * th, torch.ones_like(th))
            return grad_out * dr, None, None, None, None, None, None, None, None
        dist, r_in, r_out = ctx.saved_tensors
        w_c3 = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - dist) / w_c3).clamp(0.0, 1.0)
        u2 = u * u
        dr = torch.where(
            (dist >= r_out) | (r_out <= 0.0),
            torch.ones_like(dist),
            torch.where(dist <= r_in, torch.zeros_like(dist), 1.0 - u2 * u * (10.0 + u * (-15.0 + 6.0 * u))),
        )
        return grad_out * dr, None, None, None, None, None, None, None, None


class _NLHEnergyDense(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dist_raw: Tensor,
        zc: Tensor,
        pmf: Tensor,
        scale: float,
        r_in_ratio: float,
        mode: str,
        r_off: float,
        r_cut_envelope: float,
    ) -> Tensor:
        ctx.save_for_backward(dist_raw, zc, pmf)
        ctx.scale = float(scale)
        ctx.r_in_ratio = float(r_in_ratio)
        ctx.mode = mode
        ctx.r_off = float(r_off)
        ctx.r_cut_envelope = float(r_cut_envelope)

        cb_flat, rcov_flat, _, jet_flat = _nlh_tables(zc.device, dist_raw.dtype)
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        valid_f = valid.to(dist_raw.dtype)
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        p_idx = ((zc.unsqueeze(2) << 7) + zc.unsqueeze(1)).reshape(-1)

        ab = cb_flat.index_select(0, p_idx).view(*dist_raw.shape, 6)
        c0, c1, c2, b0, b1, b2 = ab.unbind(-1)
        e_nlh = (c0 * torch.exp(-b0 * r_eval) + c1 * torch.exp(-b1 * r_eval) + c2 * torch.exp(-b2 * r_eval)) / r_eval

        if mode == "c2_shift_rmin":
            jet = jet_flat.index_select(0, p_idx).view(*dist_raw.shape, 4)
            d_off, v0, v1, v2 = jet.unbind(-1)
            active = valid & (r_eval < d_off) & (d_off > 0.0)
            dr = r_eval - d_off
            e_shift = torch.where(active, e_nlh - (v0 + dr * (v1 + 0.5 * dr * v2)), torch.zeros_like(e_nlh))
            return 0.5 * (e_shift * pmf).sum(dim=(-1, -2))

        if mode == "so3lr":
            sw, _ = _so3lr_switch_and_grad(r_eval, valid, valid_f, ctx.r_off, ctx.r_cut_envelope)
        elif mode == "c3":
            r_out = ctx.scale * rcov_flat.index_select(0, p_idx).view_as(dist_raw) * valid_f
            r_in = ctx.r_in_ratio * r_out
            w = (r_out - r_in).clamp_min(1e-12)
            u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
            u2 = u * u
            u4 = u2 * u2
            sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
        else:
            raise ValueError(
                f"Unsupported nlh_energy_dense mode {mode!r}; expected 'so3lr', 'c3', or 'c2_shift_rmin'"
            )

        return 0.5 * (sw * e_nlh * pmf).sum(dim=(-1, -2))

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        dist_raw, zc, pmf = ctx.saved_tensors
        cb_flat, rcov_flat, _, jet_flat = _nlh_tables(zc.device, dist_raw.dtype)
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        valid_f = valid.to(dist_raw.dtype)
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        p_idx = ((zc.unsqueeze(2) << 7) + zc.unsqueeze(1)).reshape(-1)

        ab = cb_flat.index_select(0, p_idx).view(*dist_raw.shape, 6)
        c0, c1, c2, b0, b1, b2 = ab.unbind(-1)
        e0 = torch.exp(-b0 * r_eval)
        e1 = torch.exp(-b1 * r_eval)
        e2 = torch.exp(-b2 * r_eval)
        inv_r = 1.0 / r_eval
        e_nlh = (c0 * e0 + c1 * e1 + c2 * e2) * inv_r
        de_nlh_dr = (-(c0 * b0) * e0 - (c1 * b1) * e1 - (c2 * b2) * e2 - e_nlh) * inv_r

        if ctx.mode == "c2_shift_rmin":
            jet = jet_flat.index_select(0, p_idx).view(*dist_raw.shape, 4)
            d_off, _, v1, v2 = jet.unbind(-1)
            active = valid & (r_eval < d_off) & (d_off > 0.0)
            dr = r_eval - d_off
            de_pair = torch.where(active, de_nlh_dr - (v1 + dr * v2), torch.zeros_like(de_nlh_dr))
            de_dr = 0.5 * de_pair * pmf * (dist_raw >= 1e-4).to(dist_raw.dtype)
            return grad_out.view(-1, 1, 1) * de_dr, None, None, None, None, None, None, None

        if ctx.mode == "so3lr":
            sw, dsw_dr = _so3lr_switch_and_grad(r_eval, valid, valid_f, ctx.r_off, ctx.r_cut_envelope)
        else:
            r_out = ctx.scale * rcov_flat.index_select(0, p_idx).view_as(dist_raw) * valid_f
            r_in = ctx.r_in_ratio * r_out
            w = (r_out - r_in).clamp_min(1e-12)
            u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
            u2 = u * u
            u4 = u2 * u2
            sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
            om = 1.0 - u
            dsw_dr = (
                torch.where(
                    (r_eval >= r_out) | (r_eval <= r_in) | (r_out <= 0.0),
                    torch.zeros_like(r_eval),
                    (-140.0 / w) * (u2 * u) * (om * om * om),
                )
                * valid_f
            )

        de_dr = 0.5 * (dsw_dr * e_nlh + sw * de_nlh_dr) * pmf * (dist_raw >= 1e-4).to(dist_raw.dtype)
        return grad_out.view(-1, 1, 1) * de_dr, None, None, None, None, None, None, None


class _NLHEnergySparse(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dist_raw: Tensor,
        zc: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        n_graphs: int,
        scale: float,
        r_in_ratio: float,
        mode: str,
        r_off: float,
        r_cut_envelope: float,
    ) -> Tensor:
        ctx.save_for_backward(dist_raw, zc, recv, send, batch)
        ctx.scale = float(scale)
        ctx.r_in_ratio = float(r_in_ratio)
        ctx.mode = mode
        ctx.r_off = float(r_off)
        ctx.r_cut_envelope = float(r_cut_envelope)

        cb_flat, rcov_flat, _, jet_flat = _nlh_tables(zc.device, dist_raw.dtype)
        zr, zs = zc.index_select(0, recv), zc.index_select(0, send)
        valid = (zr > 0) & (zs > 0)
        valid_f = valid.to(dist_raw.dtype)
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        p_idx = (zr << 7) + zs

        ab = cb_flat.index_select(0, p_idx)
        c0, c1, c2, b0, b1, b2 = ab.unbind(-1)
        e_nlh = (c0 * torch.exp(-b0 * r_eval) + c1 * torch.exp(-b1 * r_eval) + c2 * torch.exp(-b2 * r_eval)) / r_eval

        if mode == "c2_shift_rmin":
            jet = jet_flat.index_select(0, p_idx)
            d_off, v0, v1, v2 = jet.unbind(-1)
            active = valid & (r_eval < d_off) & (d_off > 0.0)
            dr = r_eval - d_off
            e_edge = 0.5 * torch.where(active, e_nlh - (v0 + dr * (v1 + 0.5 * dr * v2)), torch.zeros_like(e_nlh))
            return dist_raw.new_zeros(int(n_graphs)).index_add_(0, batch.index_select(0, recv), e_edge)

        if mode == "so3lr":
            sw, _ = _so3lr_switch_and_grad(r_eval, valid, valid_f, ctx.r_off, ctx.r_cut_envelope)
        elif mode == "c3":
            r_out = ctx.scale * rcov_flat.index_select(0, p_idx) * valid_f
            r_in = ctx.r_in_ratio * r_out
            w = (r_out - r_in).clamp_min(1e-12)
            u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
            u2 = u * u
            u4 = u2 * u2
            sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
        else:
            raise ValueError(
                f"Unsupported nlh_energy_sparse mode {mode!r}; expected 'so3lr', 'c3', or 'c2_shift_rmin'"
            )

        e_edge = 0.5 * sw * e_nlh
        return dist_raw.new_zeros(int(n_graphs)).index_add_(0, batch.index_select(0, recv), e_edge)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        dist_raw, zc, recv, send, batch = ctx.saved_tensors
        cb_flat, rcov_flat, _, jet_flat = _nlh_tables(zc.device, dist_raw.dtype)
        zr, zs = zc.index_select(0, recv), zc.index_select(0, send)
        valid = (zr > 0) & (zs > 0)
        valid_f = valid.to(dist_raw.dtype)
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        p_idx = (zr << 7) + zs

        ab = cb_flat.index_select(0, p_idx)
        c0, c1, c2, b0, b1, b2 = ab.unbind(-1)
        e0 = torch.exp(-b0 * r_eval)
        e1 = torch.exp(-b1 * r_eval)
        e2 = torch.exp(-b2 * r_eval)
        inv_r = 1.0 / r_eval
        e_nlh = (c0 * e0 + c1 * e1 + c2 * e2) * inv_r
        de_nlh_dr = (-(c0 * b0) * e0 - (c1 * b1) * e1 - (c2 * b2) * e2 - e_nlh) * inv_r

        if ctx.mode == "c2_shift_rmin":
            jet = jet_flat.index_select(0, p_idx)
            d_off, _, v1, v2 = jet.unbind(-1)
            active = valid & (r_eval < d_off) & (d_off > 0.0)
            dr = r_eval - d_off
            de_pair = torch.where(active, de_nlh_dr - (v1 + dr * v2), torch.zeros_like(de_nlh_dr))
            de_dr = 0.5 * de_pair * (dist_raw >= 1e-4).to(dist_raw.dtype)
            return (
                grad_out.index_select(0, batch.index_select(0, recv)) * de_dr,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if ctx.mode == "so3lr":
            sw, dsw_dr = _so3lr_switch_and_grad(r_eval, valid, valid_f, ctx.r_off, ctx.r_cut_envelope)
        else:
            r_out = ctx.scale * rcov_flat.index_select(0, p_idx) * valid_f
            r_in = ctx.r_in_ratio * r_out
            w = (r_out - r_in).clamp_min(1e-12)
            u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
            u2 = u * u
            u4 = u2 * u2
            sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
            om = 1.0 - u
            dsw_dr = (
                torch.where(
                    (r_eval >= r_out) | (r_eval <= r_in) | (r_out <= 0.0),
                    torch.zeros_like(r_eval),
                    (-140.0 / w) * (u2 * u) * (om * om * om),
                )
                * valid_f
            )

        de_dr = 0.5 * (dsw_dr * e_nlh + sw * de_nlh_dr) * (dist_raw >= 1e-4).to(dist_raw.dtype)
        return (
            grad_out.index_select(0, batch.index_select(0, recv)) * de_dr,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def nlh_clamp_dense(
    z: Tensor,
    dist: Tensor,
    pmf: Tensor,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
    *,
    mode: str = "tanh",
    floor_energy_ev: float = DEFAULT_FLOOR_ENERGY_EV,
    width: float = DEFAULT_FLOOR_WIDTH_A,
) -> Tensor:
    """Dense [B, N, N] smooth distance floor/clamp for MenaACE.forward geometry featurisation.

    Args:
        mode: 'tanh' (SO3LR Tauri Eq. S5 floor at E_NLH(r_min)=floor_energy_ev, width=0.05 A),
              'tanh_c3_matched' (tanh floor starting at r_out = scale*(r_cov,i + r_cov,j)),
              or 'c3' (septic C3 clamp matching zbl_clamp_dense).
    """
    return _NLHClampDense.apply(dist, z.clamp(0, 127), pmf, scale, r_in_ratio, mode, floor_energy_ev, width)


def nlh_clamp_sparse(
    z: Tensor,
    dist: Tensor,
    recv: Tensor,
    send: Tensor,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
    *,
    mode: str = "tanh",
    floor_energy_ev: float = DEFAULT_FLOOR_ENERGY_EV,
    width: float = DEFAULT_FLOOR_WIDTH_A,
) -> Tensor:
    """Sparse [E] smooth distance floor/clamp for MenaACE.forward_sparse geometry featurisation."""
    return _NLHClampSparse.apply(dist, z.clamp(0, 127), recv, send, scale, r_in_ratio, mode, floor_energy_ev, width)


def nlh_energy_dense(
    z: Tensor,
    dist_raw: Tensor,
    pmf: Tensor,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
    *,
    mode: str = "so3lr",
    r_off: float = DEFAULT_NLH_R_OFF_A,
    r_cut_envelope: float = 0.0,
) -> Tensor:
    """Total NLH short-range energy [B] in kcal/mol for dense batches: 0.5 * sum_{i != j} sw(r_ij) E_NLH(r_ij).

    Args:
        mode: 'so3lr' (SO3LR Tauri Eq. S4 C^inf switch at r_off=1.5 A),
              'c3' (Mena septic C3 switch at r_out = scale*(r_cov,i + r_cov,j)),
              or 'c2_shift_rmin' (C2-Taylor-shifted NLH below d_off = r_min(5 eV) + 0.05 A; zero-retrain drop-in).
    """
    return _NLHEnergyDense.apply(dist_raw, z.clamp(0, 127), pmf, scale, r_in_ratio, mode, r_off, r_cut_envelope)


def nlh_energy_sparse(
    z: Tensor,
    dist_raw: Tensor,
    recv: Tensor,
    send: Tensor,
    batch: Tensor,
    n_graphs: int,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
    *,
    mode: str = "so3lr",
    r_off: float = DEFAULT_NLH_R_OFF_A,
    r_cut_envelope: float = 0.0,
) -> Tensor:
    """Total NLH short-range energy [B] in kcal/mol for sparse edge lists: 0.5 * sum_e sw(r_e) E_NLH(r_e)."""
    return _NLHEnergySparse.apply(
        dist_raw, z.clamp(0, 127), recv, send, batch, int(n_graphs), scale, r_in_ratio, mode, r_off, r_cut_envelope
    )
