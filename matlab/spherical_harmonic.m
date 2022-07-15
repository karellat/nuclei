function y=spherical_harmonic(n,m,theta,phi)
% y=spherical_harmonic(n,m,theta,phi) computes spherical surface harmonic
% of degree n and order m in the point with spherical coordinates theta and
% phi.

if n<0 || n-floor(n)>0
    error('Parameter n must be non-negative integer, not %g',n);
end
if m<-n || m>n || m-floor(m)>0
    error('Absolute value of the parameter m must be non-negative integer <= n, not %g',m);
end
sztheta=size(theta);
szphi=size(phi);
if length(sztheta)~=length(szphi)
    error('The numbers of dimensions of theta (%d) and phi (%d) must be the same',length(sztheta),length(szphi));
end
if sztheta~=szphi
    strtheta = sprintf('%dx',sztheta);
    strphi   = sprintf('%dx',szphi);
    error(['Sizes of theta ',strtheta(1:end-1),' and phi ',strphi(1:end-1),' must be the same']);
end
p = legendre(n,cos(theta(:)));
y = p(abs(m)+1,:)';
y = y.*(cos(m*phi(:))+1i*sin(abs(m)*phi(:)));
y = y*sqrt((2*n+1)/4/pi*factorial(n-abs(m))/factorial(n+abs(m)));
if m<0
    y = conj(y)*(-1)^(-m);
end
y=reshape(y,sztheta);
