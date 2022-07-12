function [poly] = GauHerPolym3D(szm, rank, sigma)
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


[x,y,z]= meshgrid(linspace(-1,1,szm), linspace(-1,1,szm), linspace(-1,1,szm));
x = x(:)' / sigma;
y = y(:)' / sigma;
z = z(:)' / sigma;
poly = zeros(rank+1, rank+1, rank+1, szm ^ 3);

kerx = zeros(rank+1, length(x));
kerx(1,:) = exp(-x.^2/2) / pi^0.25;
kerx(2,:) = 2^0.5 * x .* exp(-x.^2/2) / pi^0.25;
for dx = 2 : rank
     kerx(dx+1,:) = 1/sqrt(2*dx) * x .* kerx(dx,:) - sqrt(1-1/dx) * kerx(dx-1,:);
end
clear x;

kery = zeros(rank+1, length(y));
kery(1,:) = exp(-y.^2/2) / pi^0.25;
kery(2,:) = 2^0.5 * y .* exp(-y.^2/2) / pi^0.25;
for dy = 2 : rank
    kery(dy+1,:) = 1/sqrt(2*dy) * y .* kery(dy,:) - sqrt(1-1/dy) * kery(dy-1,:);
end
clear y;

kerz= zeros(rank+1, length(z));
kerz(1,:) = exp(-z.^2/2) / pi^0.25;
kerz(2,:) = 2^0.5 * z .* exp(-z.^2/2) / pi^0.25;
for dz = 2 : rank
    kerz(dz+1,:) = 1/sqrt(2*dz) * z .* kerz(dz,:) - sqrt(1-1/dz) * kerz(dz-1,:);
end
clear z;

for rx = 0 : rank
    for ry = 0 : rank-rx
        for rz = 0 : rank-rx-ry
            for a = 1 : szm ^ 3
                        poly(rx+1, ry+1, rz+1, a) = kerx(rx + 1, a) .* kery(ry + 1, a) .* kerz(rz + 1, a);
            end
        end
    end
end
end