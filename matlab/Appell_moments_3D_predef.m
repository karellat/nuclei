function f = Appell_moments_3D_predef(im, pl, m, n, o)
%UV = Appell_moments_3D_predef(im, pl, m, n, o) computes Appell polynomials
%in 3D from precomputed matrix of the polynomials
% Direct computation by definition.
%Parameters:
% im 3D array with the 3D image
% pl precomputed matrix of the polynomials,
%   numel(im) and size(pl,4) must be the same
% m,n,o orders
% f are computed features

f = zeros(m+1, n+1, o+1);
v = reshape(im,numel(im),1);
sc = (min(size(im))-1)/2;

for p=0:m
    for q=0:n
        for r=0:o
            plm=squeeze(pl(p+1,q+1,r+1,:));
            f(p+1,q+1,r+1)=sum(plm.*v)/sc^3;%*3/4/pi/sc^3;
        end
    end
end



end

