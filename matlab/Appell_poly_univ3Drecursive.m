function UV = Appell_poly_univ3Drecursive(m, n, o, x, y, z, t, s, w)
%UV = Appell_poly_univ3Drecursive(m, n, x, y, t, s, w) computes Appell polynomials in 3D.
% Recursive computation.
%Parameters:
% m,n,o maximum orders
% x,y,z vectors of coordinates
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

vl=min([length(x),length(y),length(z)]);
xd=zeros(1,1,1,vl);
xd(1,1,1,:)=x;
yd=zeros(1,1,1,vl);
yd(1,1,1,:)=y;
zd=zeros(1,1,1,vl);
zd(1,1,1,:)=z;
UV=zeros(m+1,n+1,o+1,vl);

if t==1
    UV(1,1,1,:)=ones(1,1,1,vl);
%     UV(2,1,1,:)=xd;
%     UV(1,2,1,:)=yd;
%     UV(1,1,2,:)=zd;
%     UV(3,1,1,:)=3*xd.^2+yd.^2+zd.^2-1;
%     UV(1,3,1,:)=xd.^2+3*yd.^2+zd.^2-1;
%     UV(1,1,3,:)=xd.^2+yd.^2+3*zd.^2-1;
%     UV(2,2,1,:)=2*xd.*yd;
%     UV(2,1,2,:)=2*xd.*zd;
%     UV(1,2,2,:)=2*yd.*zd;
    UV(2,1,1,:)=(2*s-1)*xd;
    UV(1,2,1,:)=(2*s-1)*yd;
    UV(1,1,2,:)=(2*s-1)*zd;
    UV(3,1,1,:)=(2*s-1)*((2*s+1)*xd.^2+yd.^2+zd.^2-1);
    UV(1,3,1,:)=(2*s-1)*(xd.^2+(2*s+1)*yd.^2+zd.^2-1);
    UV(1,1,3,:)=(2*s-1)*(xd.^2+yd.^2+(2*s+1)*zd.^2-1);
    UV(2,2,1,:)=(2*s-1)*2*s*xd.*yd;
    UV(2,1,2,:)=(2*s-1)*2*s*xd.*zd;
    UV(1,2,2,:)=(2*s-1)*2*s*yd.*zd;
    mr=max([m,n,o]);
    for or=2:mr-1
        for m=0:or
            for n=0:or-m
                o=or-m-n;
%                disp([m,n,o])
                UV111=UV(m+1,n+1,o+1,:);
                UV110=0;
                if o>0
                    UV110=UV(m+1,n+1,o,:);
                end
                UV101=0;
                if n>0
                    UV101=UV(m+1,n,o+1,:);
                end
                UV011=0;
                if m>0
                    UV011=UV(m,n+1,o+1,:);
                end
                UV100=0;
                if n>0 && o>0
                    UV100=UV(m+1,n,o,:);
                end
                UV010=0;
                if m>0 && o>0
                    UV010=UV(m,n+1,o,:);
                end
                UV001=0;
                if m>0 && n>0
                    UV001=UV(m,n,o+1,:);
                end
                UV000=0;
                if m>0 && n>0 && o>0
                    UV000=UV(m,n,o,:);
                end
                UV(m+2,n+1,o+1,:)=(2*m+n+o+1)*xd.*UV111+m*o*xd.*zd.*UV110+m*n*xd.*yd.*UV101+2*m*n*o*xd.*yd.*zd.*UV100+...
                    m*((yd.^2+zd.^2-1)*m+(yd.^2+2*zd.^2-1)*o+(2*yd.^2+zd.^2-1)*n).*UV011+...
                    m*o*zd.*((yd.^2-1)*(m+o-1)+(3*yd.^2-1)*n).*UV010+...
                    m*n*yd.*((3*zd.^2-1)*o+(zd.^2-1)*(m+n-1)).*UV001-2*m*n*o*yd.*zd*(m+n+o-2).*UV000;
                UV(m+1,n+2,o+1,:)=(m+2*n+o+1)*yd.*UV111+n*o*yd.*zd.*UV110+m*n*xd.*yd.*UV011+2*m*n*o*xd.*yd.*zd.*UV010+...
                    n*((xd.^2+zd.^2-1)*n+(xd.^2+2*zd.^2-1)*o+(2*xd.^2+zd.^2-1)*m).*UV101+...
                    n*o*zd.*((xd.^2-1)*(n+o-1)+(3*xd.^2-1)*m).*UV100+...
                    m*n*xd.*((3*zd.^2-1)*o+(zd.^2-1)*(m+n-1)).*UV001-2*m*n*o*xd.*zd*(m+n+o-2).*UV000;
                UV(m+1,n+1,o+2,:)=(m+n+2*o+1)*zd.*UV111+m*o*xd.*zd.*UV011+n*o*yd.*zd.*UV101+2*m*n*o*xd.*yd.*zd.*UV001+...
                    o*((xd.^2+yd.^2-1)*o+(2*xd.^2+yd.^2-1)*m+(xd.^2+2*yd.^2-1)*n).*UV110+...
                    m*o*xd.*((yd.^2-1)*(m+o-1)+(3*yd.^2-1)*n).*UV010+...
                    n*o*yd.*((xd.^2-1)*(n+o-1)+(3*xd.^2-1)*m).*UV100-2*m*n*o*xd.*yd*(m+n+o-2).*UV000;
            end
        end
    end
else
    UV(1,1,1,:)=ones(1,1,1,vl);
%     UV(2,1,1,:)=(s+2)*xd;
%     UV(1,2,1,:)=(s+2)*yd;
%     UV(1,1,2,:)=(s+2)*zd;
%     UV(3,1,1,:)=(s+2)*((s+4)*xd.^2-1);
%     UV(1,3,1,:)=(s+2)*((s+4)*yd.^2-1);
%     UV(1,1,3,:)=(s+2)*((s+4)*zd.^2-1);
%     UV(2,2,1,:)=(s+2)*(s+4)*xd.*yd;
%     UV(2,1,2,:)=(s+2)*(s+4)*xd.*zd;
%     UV(1,2,2,:)=(s+2)*(s+4)*yd.*zd;
    UV(2,1,1,:)=(2*s+1)*xd;
    UV(1,2,1,:)=(2*s+1)*yd;
    UV(1,1,2,:)=(2*s+1)*zd;
    UV(3,1,1,:)=(2*s+1)*((2*s+3)*xd.^2-1);
    UV(1,3,1,:)=(2*s+1)*((2*s+3)*yd.^2-1);
    UV(1,1,3,:)=(2*s+1)*((2*s+3)*zd.^2-1);
    UV(2,2,1,:)=(2*s+1)*(2*s+3)*xd.*yd;
    UV(2,1,2,:)=(2*s+1)*(2*s+3)*xd.*zd;
    UV(1,2,2,:)=(2*s+1)*(2*s+3)*yd.*zd;
    mr=max([m,n,o]);
    for or=2:mr-1
        for m=0:or
            for n=0:or-m
                o=or-m-n;
 %               disp([m,n,o])
                UV111=UV(m+1,n+1,o+1,:);
                UV2_11=0;
                if n>1
                    UV2_11=UV(m+2,n-1,o+1,:);
                end
                UV21_1=0;
                if o>1
                    UV21_1=UV(m+2,n+1,o-1,:);
                end
                UV011=0;
                if m>0
                    UV011=UV(m,n+1,o+1,:);
                end
                UV_121=0;
                if m>1
                    UV_121=UV(m-1,n+2,o+1,:);
                end
                UV12_1=0;
                if o>1
                    UV12_1=UV(m+1,n+2,o-1,:);
                end
                UV101=0;
                if n>0
                    UV101=UV(m+1,n,o+1,:);
                end
                UV_112=0;
                if m>1
                    UV_112=UV(m-1,n+1,o+2,:);
                end
                UV1_12=0;
                if n>1
                    UV1_12=UV(m+1,n-1,o+2,:);
                end
                UV110=0;
                if o>0
                    UV110=UV(m+1,n+1,o,:);
                end
                UV(m+2,n+1,o+1,:)=(2*(m+n+o+1)+s)*xd.*UV111+n*(n-1)*UV2_11+o*(o-1)*UV21_1-m*(m+2*n+2*o+s+1)*UV011;
                UV(m+1,n+2,o+1,:)=(2*(m+n+o+1)+s)*yd.*UV111+m*(m-1)*UV_121+o*(o-1)*UV12_1-n*(2*m+n+2*o+s+1)*UV101;
                UV(m+1,n+1,o+2,:)=(2*(m+n+o+1)+s)*zd.*UV111+m*(m-1)*UV_112+n*(n-1)*UV1_12-o*(2*m+2*n+o+s+1)*UV110;
            end
        end
    end
end

if w==1
    for or=0:mr
        for m=0:or
            for n=0:or-m
                o=or-m-n;
                UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)*sqrt((m+n+o+s+1/2)*factorial(m)*factorial(n)*factorial(o)*gamma(s+3/2)/(pi^(3/2)*gamma(s)*(s+1/2)*pochham(2*s-1,m+n+o)));
                UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:).*(1-xd.^2-yd.^2-zd.^2).^((s-1)/2);
            end
        end
    end
elseif w==2
    for or=0:mr
        for m=0:or
            for n=0:or-m
                o=or-m-n;
                UV(m+1,n+1,o+1,:)=UV(m+1,n+1,o+1,:)/factorial(m)/factorial(n)/factorial(o);
            end
        end
    end
elseif w==3
    for or=0:mr
        for m=0:or
            for n=0:or-m
                o=or-m-n;
                if t==1
                    UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)*sqrt((2*(m+n+o)+3)*factorial(m)*factorial(n)*factorial(o)/4/pi)/factorial(m+n+o)^(9/32);
                else
                    UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)*sqrt((2*(m+n+o)+3)*factorial(m)*factorial(n)*factorial(o)/4/pi)/factorial(m+n+o)^(23/32);
                end
                UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:).*(1-xd.^2-yd.^2-zd.^2).^((s-1)/2);
            end
        end
    end
elseif w==4
    for or=0:mr
        for m=0:or
            for n=0:or-m
                o=or-m-n;
                if t==1
                    UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)*factorial(m+n+o)^(5/32);
                else
                    UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)/factorial(m+n+o)^(5/32);
                end
            end
        end
    end
elseif w==5
    for or=0:mr
        for m=0:or
            for n=0:or-m
                o=or-m-n;
                if t==1
                    UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)*sqrt((2*(m+n+o)+3)/gamma((m+n+o)/3+1)^(3/2)/gamma(m+n+o+1)^(1/2)/4/pi)/factorial(m+n+o)^(9/32);
                else
                    UV(m+1,n+1,o+1,:) = UV(m+1,n+1,o+1,:)*sqrt((2*(m+n+o)+3)/gamma((m+n+o)/3+1)^(3/2)/gamma(m+n+o+1)^(1/2)/4/pi)/factorial(m+n+o)^(23/32);
                end
            end
        end
    end
end

UV=real(UV);

end
