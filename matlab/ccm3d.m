function [cr,t1,t2,t3]=ccm3d(img,rd,norm)
% [cr,t1,t2,t3]=ccm3d(img,rd,norm) computes central complex moments
% up the rd-th order of the 3D volumetric image img(n1,n2,n3).
% if norm=1, origin in the center.
% if norm=2, origin in the centroid (default).
% t1,t2,t3 are coordinates of the centroid of the iamge

% The moment C_{es,el}^{em} = cr(es+1,floor(el/2)+1,em+el+1)
% i.e. In cr(p,q,r), there is C_{es,el}^{em}, where es=p-1, el=2*(q-1)+mod(es,2) and em=r-1-el.

if nargin<2
    rd=6;
end
if nargin<3
    norm=2;
end

[n1,n2,n3]=size(img);
cr=zeros(rd+1,floor(rd/2)+1,2*rd+1);

cd = find(img(:));
if isempty(cd)
    return
end
v=img(cd);
[x, y, z] = ind2sub(size(img),cd);
clear img;

sv=sum(v);
if norm==1
    t1=(n1-1)/2;
    t2=(n2-1)/2;
    t3=(n3-1)/2;
else
    if sv~=0
        t1 = sum(x .* v) / sv;
        t2 = sum(y .* v) / sv;
        t3 = sum(z .* v) / sv;
    else
        t1=(n1-1)/2;
        t2=(n2-1)/2;
        t3=(n3-1)/2;
    end
end
x=x-t1;
y=y-t2;
z=z-t3;

r=sqrt(x.^2+y.^2+z.^2);
theta=acos(z./r);
assert(sum(isnan(theta)) == 1);
theta(isnan(theta)) = 0;
phi=atan2(y,x);

for es=0:rd %order
    for el=mod(es,2):2:es  %latitudinal repetition
        for em=-el:el  %longitudinal repetition
            rmn=r.^es;
            vmn=rmn.*spherical_harmonic(el,em,theta,phi);
%             vmn=rmn.*spherical_harmonic(el,em,theta,phi).*r.^2.*sin(theta);
            prodiv=v.*vmn;
            cr(es+1,floor(el/2)+1,em+el+1)=sum(prodiv(:));
        end
    end
end
%normalization to density of sampling
%A=A/rmax^3;
%normalization to density of sampling + contrast
%cr=cr/cr(1,1,1);