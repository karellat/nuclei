function [poly] = zm3dpoly(szm, rank)

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

%Kintner method
poly=zeros(rank+1,floor(rank/2)+1,2*rank+1, szm^3);
for el=0:rank  %latitudinal repetition
    for em=-el:el  %longitudinal repetition
        rmn0=r.^el;
        vmn=conj(rmn0.*spherical_harmonic(el,em,theta,phi));
        poly(el+1,floor(el/2)+1,em+el+1, :)=(el+1)/pi*vmn;
        if(rank-el>=2)
            rmn2=(el+2)*r.^(el+2)-(el+1)*r.^el;
            vmn=conj(rmn2.*spherical_harmonic(el,em,theta,phi));
            poly(el+3,floor(el/2)+1,em+el+1, :)=(el+3)/pi*vmn;
        end
        for es=el+4:2:rank %order
            k1=(es+el)*(es-el)*(es-2)/2;
            k2=2*es*(es-1)*(es-2);
            k3=-el^2*(es-1)-es*(es-1)*(es-2);
            k4=-es*(es+el-2)*(es-el-2)/2;
            rmn4=((k2*r.^2+k3).*rmn2+k4*rmn0)/k1;
            vmn=conj(rmn4.*spherical_harmonic(el,em,theta,phi));
            poly(es+1,floor(el/2)+1,em+el+1, :)=(es+1)/pi*vmn;
            rmn0=rmn2;
            rmn2=rmn4;
        end
    end
end
end