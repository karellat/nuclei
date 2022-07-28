function [ral,ralp,ralq]=reduced_associated_legendre(el,em)
% Function [ral,ralp,ralq]=reduced_associated_legendre(el,em)
% generates coefficients of the reduced associated Legendre function. The
% Legendre function is defined

% P(el,em;x) = (-1)^em * (1-x^2)^(em/2) * (d/dx)^em { P(el,x) },
 
% where P(el,x) is the Legendre polynomial of degree el. The reduced
% version is defined without the factor (1-x^2)^(em/2), i.e.

% Pr(el,em;x) = (-1)^em * (d/dx)^em { P(el,x) }.

% The results are in array ral, where ral(k) is the coefficient at
% x^(el-|em|-k+1).

% The vectors ralp and ralq are integers so ral=ralp./ralq.

coef=1;
coefp=1;
coefq=1;
if em<0
    em=-em;
    coefp=(-1)^em*factorial(el-em);
    coefq=factorial(el+em);
    gcdc=gcd(coefp,coefq);
    coefp=coefp/gcdc;
    coefq=coefq/gcdc;
    coef=coefp/coefq;
end
ral =zeros(1,2*el+1);
ralp=zeros(1,2*el+1);
for k=0:el
    ral (2*(el-k)+1)=nchoosek(el,k)*(-1)^(el-k);
    ralp(2*(el-k)+1)=ral(2*(el-k)+1);
end
for k=1:el+em
    ral=polyder(ral);
    ralp=polyder(ralp);
    if k<=el
        ral=ral/2/k;
        coefq=coefq*2*k;
        gcdc=gcdvect([coefq,ralp]);
        coefq=coefq/gcdc;
        ralp=ralp/gcdc;
    end
end
ral=ral*(-1)^em*coef;
ralp=ralp*(-1)^em*coefp;
ralq=coefq;
