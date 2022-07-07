function UV = Appell_poly_univ3D(m, n, o, x, y, z, t, s, w)
%UV = Appell_poly_univ(m, n, x, y, t, s, w) computes Appell polynomials in 3D.
% Direct computation by definition.
%Parameters:
% m,n,o orders
% x,y,z coordinates
% t type t=1 U polynomials are computed
%        t=0 V polynomials are computed
% s parameter s of the polynomials
% w weight w=0 polynomials are not weighted
%          w=1 weighted polynomials are computed
%          w=2 polynomials are normalized by m!n!o!
%          w=3 U*sqrt((2*(m+n+o)+3)*m!*n!*o!/4/pi)/(m+n+o)!^(9/32);
%              V*sqrt((2*(m+n+o)+3)*m!*n!*o!/4/pi)/(m+n+o)!^(23/32);
%          w=4 polynomials are normalized by ((m+n+o)!)^(5/32)
%          w=5 U*sqrt((2(m+n+o)+3)/(((m+n+o)/3)!)^(3/2)/((m+n+o)!)^(1/2)/4/pi)/((m+n+o)!)^(9/32);
%              V*sqrt((2(m+n+o)+3)/(((m+n+o)/3)!)^(3/2)/((m+n+o)!)^(1/2)/4/pi)/((m+n+o)!)^(23/32);

if nargin<8
    s=1;
end

if nargin<9
    w=1;
end

UV = zeros(size(x));

if t==1     % U polynomials
    for i = 0:m/2
        for j = 0:n/2
            for k = 0:o/2
                UV = UV +(-1)^(i+j+k)*pochham(-m,2*i)*pochham(-n,2*j)*pochham(-o,2*k)/...
                    (4^(i+j+k)*factorial(i)*factorial(j)*factorial(k)*pochham(s,i+j+k))*...
                    x.^(m-2*i).*y.^(n-2*j).*z.^(o-2*k).*(1-x.^2-y.^2-z.^2).^(i+j+k);
            end
        end
    end
    UV = UV*pochham(2*s-1,m+n+o);
else     % V polynomials
    for i = 0:m/2
        for j = 0:n/2
            for k = 0:o/2
                %
%                 UV = UV + pochham(s+1/2,m+n+o-i-j-k)*pochham(i-m,i)*pochham(j-n,j)*pochham(k-o,k)/...
%                     (factorial(i)*factorial(j)*factorial(k)*factorial(m-i)*factorial(n-j)*factorial(o-k)*4^(i+j+k))*...
%                     x.^(m-2*i).*y.^(n-2*j).*z.^(o-2*k);
                UV = UV + pochham(s+1/2,m+n+o-i-j-k)*pochham(i-m,i)*pochham(j-n,j)*pochham(k-o,k)*...
                    nchoosek(m,i)*nchoosek(n,j)*nchoosek(o,k)/4^(i+j+k)*...
                    x.^(m-2*i).*y.^(n-2*j).*z.^(o-2*k);
            end
        end
    end
    UV = UV*2^(m+n+o);
end

if w==1
    UV = UV*sqrt((m+n+o+s+1/2)*factorial(m)*factorial(n)*factorial(o)*gamma(s+3/2)/(pi^(3/2)*gamma(s)*(s+1/2)*pochham(2*s-1,m+n+o)));
    UV = UV.*(1-x.^2-y.^2-z.^2).^((s-1)/2);
elseif w==2
    UV = UV/factorial(m)/factorial(n)/factorial(o);
elseif w==3
    if t==1
        UV = UV*sqrt((2*(m+n+o)+3)*factorial(m)*factorial(n)*factorial(o)/4/pi)/factorial(m+n+o)^(9/32);
    else
        UV = UV*sqrt((2*(m+n+o)+3)*factorial(m)*factorial(n)*factorial(o)/4/pi)/factorial(m+n+o)^(23/32);
    end
    UV = UV.*(1-x.^2-y.^2-z.^2).^((s-1)/2);
elseif w==4
    if t==1
        UV = UV*factorial(m+n+o)^(5/32);
    else
        UV = UV/factorial(m+n+o)^(5/32);
    end
elseif w==5
    if t==1
        UV = UV*sqrt((2*(m+n+o)+3)/gamma((m+n+o)/3+1)^(3/2)/gamma(m+n+o+1)^(1/2)/4/pi)/factorial(m+n+o)^(9/32);
    else
        UV = UV*sqrt((2*(m+n+o)+3)/gamma((m+n+o)/3+1)^(3/2)/gamma(m+n+o+1)^(1/2)/4/pi)/factorial(m+n+o)^(23/32);
    end
end

UV=real(UV);

end




