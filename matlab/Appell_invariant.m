function [ y ] = Appell_invariant(img, szm, fname, or, typec, s, w)
    % Moment invariant transition
    % What is ors ?
    f=readinv3dst(fname);
    ors=zeros(1,or);
    for i=1:or
        j=1;
        while j<=length(f) && f(j).ind(1,end-2)+f(j).ind(1,end-1)+f(j).ind(1,end)<=i
            j=j+1;
        end
        ors(i)=j-1;
    end

    [x,y,z] = meshgrid(linspace(-1,1,szm),
                       linspace(-1,1,szm),
                       linspace(-1,1,szm));

    xr = reshape(x,numel(x),1);
    yr = reshape(y,numel(y),1);
    zr = reshape(y,numel(z),1);
    % Calculate polynomials
    P = Appell_poly_univ3Drecursive(or, or, or, xr, yr, zr, typc, s, w);
    % Calculate moments
    mr = Appell_moments_3D_predef(img, P, or, or, or);
    % Calculate invariants
    ec=cafmi3dst(f(1:ors(or)),mr,0,2);



end
