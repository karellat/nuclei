function s3ctgc=symb3d_complex_to_geomt(es,el,em)
% function s3ctgc=symb3d_complex_to_geomt(es,el,em)
% generates coefficients for conversion of 3D complex moment to 3D
% geometric moments. 
% The result s3ctgc is a structure with these fields:
% coef  - coefficients of the linear combination
% ind   - indices of the geometric moments
% int   - coefficients in integer form
% ints  - coefficient signums in integer form
% coefp - numerator of the total coefficient
% coefq - denominator of the total coefficient

% The complex moment c_{es,el}^{em} = sum(s3ctgc.coef(k)*m_{pqr}), 
% where k=1:size(s3ctgc.coef,2); 
% p=s3ctgc.ind(k,1); q=s3ctgc.ind(k,2); r=s3ctgc.ind(k,3); 
% s3ctgc.int(k) are complex numbers with integer real and imaginary parts so
% s3ctgc.coef = s3ctgc.int * sqrt( s3ctgc.coefp / s3ctgc.coefq / pi)

[~,ralp,crq]=reduced_associated_legendre(el,em);
coefp=(2*el+1)*factorial(el-em);
coefq=4*factorial(el+em);
gcdc=gcd(coefp,coefq);      %the fraction is reduced successively because of the numerical stability
coefp=coefp/gcdc;
coefq=coefq/gcdc;
coefq=coefq*crq;
gcdc=gcd(coefp,coefq);
coefp=coefp/gcdc;
coefq=coefq/gcdc;
coefq=coefq*crq;
gcdc=gcd(coefp,coefq);
coefp=coefp/gcdc;
coefq=coefq/gcdc;
coef=sqrt(coefp/coefq/pi);

fi(:,:,1)=[0 0;0 0];
fi(:,:,2)=[0 1;1i*sign(em) 0];
rho(:,:,1)=[0 0 0;0 0 0;0 0 1];
rho(:,:,2)=[0 0 0;0 0 0;0 0 0];
rho(:,:,3)=[0 0 1;0 0 0;1 0 0];
auxsum=zeros(el+1,el+1,el+1);
auxsum(el+1,el+1,abs(em)+1)=ralp(1);
%auxsum(el+1,el+1,1)=ralp(1);
auxpower=rho;
for k1=3:2:el+1-abs(em)
    auxprod=zeros(el-k1+2,el-k1+2,el-k1+2);
    auxprod(el-k1+2,el-k1+2,abs(em)+1)=ralp(k1);
    auxsum=auxsum+convn(auxprod,auxpower);
    auxpower=convn(auxpower,rho);
end
auxpower=rho;
for k1=el+1:2:es
    auxsum=convn(auxsum,auxpower);
%    auxpower=convn(auxpower,rho);
end
auxpower=1;
for k1=1:abs(em)
    auxpower=convn(auxpower,fi);
end
auxsum=convn(auxsum,auxpower);
n=1;
for k1=1:size(auxsum,1)
    for k2=1:size(auxsum,2)
        for k3=1:size(auxsum,3)
            if abs(auxsum(k1,k2,k3))
                s3ctgc.coef(n)=auxsum(k1,k2,k3)*coef;
                s3ctgc.ind(n,1)=size(auxsum,1)-k1;
                s3ctgc.ind(n,2)=size(auxsum,2)-k2;
                s3ctgc.ind(n,3)=size(auxsum,3)-k3;
                s3ctgc.int(n)=auxsum(k1,k2,k3);
                s3ctgc.ints(n)=sign(s3ctgc.int(n));
                s3ctgc.int(n)=abs(s3ctgc.int(n))^2;
                n=n+1;
            end
        end
    end
end
% isq=part_to_integer_and_sqrt(coefq);
% gcdc=gcdvect([isq,real(s3ctgc.int),imag(s3ctgc.int)]);
% s3ctgc.coefp=coefp;
% s3ctgc.coefq=coefq/gcdc;
% s3ctgc.int=s3ctgc.int/gcdc;
gcdc=gcdvect(s3ctgc.int);
coefp=coefp*gcdc;
s3ctgc.int=s3ctgc.int/gcdc;
gcdc=gcd(coefp,coefq);
s3ctgc.coefp=coefp/gcdc;
s3ctgc.coefq=coefq/gcdc;



