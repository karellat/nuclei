function [poly] = cc3dpoly(szm, rank, norm)

[x,y,z]= meshgrid(linspace(-1,1,szm), linspace(-1,1,szm), linspace(-1,1,szm));
x = x(:)';
y = y(:)';
z = z(:)';
% Spherical coordinates
r=sqrt(x.^2+y.^2+z.^2);
% set middle point to zero angle
theta=acos(z./r);
assert(sum(isnan(theta)) == 1);
theta(isnan(theta)) = 0;
phi=atan2(y,x);

poly=zeros(rank+1,floor(rank/2)+1,2*rank+1, szm^3);
for es=0:rank %order
    for el=mod(es,2):2:es  %latitudinal repetition
        for em=-el:el  %longitudinal repetition
            poly(es+1,floor(el/2)+1,em+el+1, :)= r.^es.*spherical_harmonic(el,em,theta,phi);
        end
    end
end