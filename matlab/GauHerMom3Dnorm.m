function [MomArray] = GauHerMom3Dnorm(im, or, sigma, normcoef)
% [MomArray,cx,cy,cz] = GauHerMom3Dnorm(im, or, sigma, normcoef)
% computes Gaussian Hermite moments of the 3D image im up to the order or
% with standard deviation sigma.
% normcoef is normalizing coefficient.
% normcoef=1, ghm_{p0} and ghm_{0q} are normalized precisely,
% normcoef=0, ghm_{p/2,p/2} are normalized precisely. Default is 0.5.
% normsize is way of normalization of coordinates to size:
% if normsize=1, the size of image im, each coordinate is used separately,
% if normsize=2, the biggest size of image,
% if normsize=3, the most distant point with non-zero value,
% if normsize=4, moment m000.

% Result MomArray is 3D array, where MomArray(p+1,q+1,r+1)=GHM_pqr.
% [cx,cy,cz] are centroid coordinates.

if nargin<3
    sigma=0.3;
end
if nargin<4
    normcoef=0.5;
end

MomArray = zeros(or+1, or+1, or+1);

assert(min(im(:)) >= 0);
assert(max(im(:)) <= 1);
v = im(:);


szm = max(size(im));
[x,y,z]= meshgrid(linspace(-1,1,szm), linspace(-1,1,szm), linspace(-1,1,szm));
x = x(:)' / sigma;
y = y(:)' / sigma;
z = z(:)' / sigma;

% kerx, kery, kerz
kerx = zeros(or+1, length(x));
kerx(1,:) = exp(-x.^2/2) / pi^0.25;
%kerx(2,:) = 2 * x .* exp(x.^2/(-2));   %without normalization
kerx(2,:) = 2^0.5 * x .* exp(-x.^2/2) / pi^0.25;
for dx = 2 : or
%    kerx(dx+1,:) = 2 * x .* kerx(dx,:) - 2 *(dx-1) * kerx(dx-1,:);   %without normalization
    kerx(dx+1,:) = 1/sqrt(2*dx) * x .* kerx(dx,:) - sqrt(1-1/dx) * kerx(dx-1,:);
end
clear x;

kery = zeros(or+1, length(y));
kery(1,:) = exp(-y.^2/2) / pi^0.25;
%kery(2,:) = 2 * y .* exp(y.^2/(-2));   %without normalization
kery(2,:) = 2^0.5 * y .* exp(-y.^2/2) / pi^0.25;
for dy = 2 : or
%     kery(dy+1,:) = 2 * y .* kery(dy,:) - 2 *(dy-1) * kery(dy-1,:);   %without normalization
    kery(dy+1,:) = 1/sqrt(2*dy) * y .* kery(dy,:) - sqrt(1-1/dy) * kery(dy-1,:);
end
clear y;

kerz= zeros(or+1, length(z));
kerz(1,:) = exp(-z.^2/2) / pi^0.25;
%kerz(2,:) = 2 * z .* exp(z.^2/(-2));   %without normalization
kerz(2,:) = 2^0.5 * z .* exp(-z.^2/2) / pi^0.25;
for dz = 2 : or
%     kerz(dz+1,:) = 2 * z .* kerz(dz,:) - 2 *(dz-1) * kerz(dz-1,:);   %without normalization
    kerz(dz+1,:) = 1/sqrt(2*dz) * z .* kerz(dz,:) - sqrt(1-1/dz) * kerz(dz-1,:);
end
clear z;



for rx = 0 : or
    for ry = 0 : or-rx
        for rz = 0 : or-rx-ry
             MomArray(rx+1, ry+1, rz+1) = sum(kerx(rx+1,:) .* kery(ry+1,:) .* kerz(rz+1,:) .* v') ...
                 *exp(gammaln(rx+1)+gammaln(ry+1)+gammaln(rz+1)-gammaln(rx+ry+rz+1)*normcoef/2-gammaln((rx+ry+rz)/2+1)*(1-normcoef));
%             *factorial(rx)*factorial(ry)*factorial(rz)/(factorialhalf(rx+ry+rz)^normcoef*factorialhalf((rx+ry+rz)/2)^(1-normcoef))^0.5;
        end
    end
end

% Normalizace
MomArray = MomArray * (2.0 / szm) ^ 3;
clear kerx kery kerz v;