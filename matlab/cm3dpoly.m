function [mr,t1,t2,t3]=cm3dpoly(szm,or,typc)
% mr=cm3d(image,or,typc,h)
% It computes central 3D moments \mu_{pqr} up to the order 'or'
% from the 3D image 'image'.
% The moment \mu_{pqr} = mr(p+1,q+1,r+1)
% typc type of centering:
% typc=0 no centering (left-top-front corner - geometric moments),
% typc=1 center,
% typc=2 centroid (default),
% t1,t2,t3 are coordinates of the centroid of the image

if nargin<3
    typc=2;
end

or=round(or);

pp=1;


n1 = szm;
n2 = szm;
n3 = szm;

v1=linspace(1,n1,n1)';
v2=linspace(1,n2,n2);
v3=zeros(1,1,n3);
v3(1,1,:)=linspace(1,n3,n3);
if typc==0
    t1=0;
    t2=0;
    t3=0;
elseif typc==1
    t1=(n1+1)/2;
    t2=(n2+1)/2;
    t3=(n3+1)/2;
end
v1=v1-t1;
v2=v2-t2;
v3=v3-t3;
v1a=zeros(n1,n2,n3,or+1);
v2a=zeros(n1,n2,n3,or+1);
v3a=zeros(n1,n2,n3,or+1);
for p=0:or
    v1a(:,:,:,p+1)=repmat(v1.^p,[1 n2 n3]);
    v2a(:,:,:,p+1)=repmat(v2.^p,[n1 1 n3]);
    v3a(:,:,:,p+1)=repmat(v3.^p,[n1 n2 1]);
end
mr=zeros(or+1,or+1,or+1, n1 * n2 * n3);
for p=0:or
    v1c=v1a(:,:,:,p+1);
    for q=0:or-p
        v2c=v2a(:,:,:,q+1);
        for r=0:or-p-q
            v3c=v3a(:,:,:,r+1);
            pp=pp+1;
            mr(p+1,q+1,r+1, :)=v1c(:).*v2c(:).*v3c(:);
        end
    end
end
