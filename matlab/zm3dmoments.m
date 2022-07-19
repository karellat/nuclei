function [A]=zm3dmoments(img,rd, mask_sphere, normalization)
% [A]=zm3d(img,rd,norm) computes Zernike moments up the rd-th order of the
% 3d volumetric image img(n1,n2,n3).
% if norm=0, origin in the centroid, the most distant corner is mapped to unit sphere.
% if norm=1, origin in the centroid, mapping by m00.
% if norm=2, origin in the center, the most distant corner is mapped to unit sphere.
% if norm=3, origin in the centroid, the most distant point with non-zero value is mapped to unit sphere.
% c the radius mapped to unit sphere is multipled by the coefficient c.

% The moment A_{es,el}^{em} = A(es+1,floor(el/2)+1,em+el+1)
% i.e. In A(p,q,r), there is A_{es,el}^{em}, where es=p-1, el=2*(q-1)+mod(es,2) and em=r-1-el.
% g is the radius mapped to the surface of the unit sphere


szm = max(size(img));
v=img(:)';
clear img;

[x,y,z]= meshgrid(linspace(-1,1,szm), linspace(-1,1,szm), linspace(-1,1,szm));
x = x(:)';
y = y(:)';
z = z(:)';

r=sqrt(x.^2+y.^2+z.^2);
theta=acos(z./r);
assert(sum(isnan(theta)) == 1);
theta(isnan(theta)) = 0;
phi=atan2(y,x);
if mask_sphere
    sphere_mask = (r <= 1.0);
    v = v .* sphere_mask;
end

%Kintner method
A=zeros(rd+1,floor(rd/2)+1,2*rd+1);
for el=0:rd  %latitudinal repetition
    for em=-el:el  %longitudinal repetition
        rmn0=r.^el;
        vmn=rmn0.*spherical_harmonic(el,em,theta,phi);
        prodiv=v.*conj(vmn);
        A(el+1,floor(el/2)+1,em+el+1)=(el+1)/pi*sum(prodiv(:));
        if(rd-el>=2)
            rmn2=(el+2)*r.^(el+2)-(el+1)*r.^el;
            vmn=rmn2.*spherical_harmonic(el,em,theta,phi);
            prodiv=v.*conj(vmn);
            A(el+3,floor(el/2)+1,em+el+1)=(el+3)/pi*sum(prodiv(:));
        end
        for es=el+4:2:rd %order
            k1=(es+el)*(es-el)*(es-2)/2;
            k2=2*es*(es-1)*(es-2);
            k3=-el^2*(es-1)-es*(es-1)*(es-2);
            k4=-es*(es+el-2)*(es-el-2)/2;
            rmn4=((k2*r.^2+k3).*rmn2+k4*rmn0)/k1;
            vmn=rmn4.*spherical_harmonic(el,em,theta,phi);
            prodiv=v.*conj(vmn);
            A(es+1,floor(el/2)+1,em+el+1)=(es+1)/pi*sum(prodiv(:));
            rmn0=rmn2;
            rmn2=rmn4;
        end
    end
end
%normalization to density of sampling
if normalization==1
    A=A/A(1,1,1);
elseif normalization==0
   A = A / szm ^3;
end
end