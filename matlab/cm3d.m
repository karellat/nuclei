function [mr,t1,t2,t3]=cm3d(image,or,typc,h)
% mr=cm3d(image,or,typc,h)
% It computes central 3D moments \mu_{pqr} up to the order 'or' 
% from the 3D image 'image'.
% The moment \mu_{pqr} = mr(p+1,q+1,r+1)
% typc type of centering: 
% typc=0 no centering (left-top-front corner - geometric moments), 
% typc=1 center,
% typc=2 centroid (default),
% h is handle of a waitbar, if h==0, no waitbar is used (default)
% t1,t2,t3 are coordinates of the centroid of the iamge

if nargin<3
    typc=2;
end
if nargin<4
    h=0;
end
or=round(or);
if h>0
    sza=sprintf('Computation of 3D central geometric moments to the order %d',or);
    waitbar(0,h,sza);
end
pp=1;
ppn=4+nchoosek(or+3,3);

[n1,n2,n3]=size(image);

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
elseif typc==2
    mp=zeros(2,2,2);
    for p=0:1
        v1c=repmat(v1.^p,[1 n2 n3]);
        for q=0:1-p
            v2c=repmat(v2.^q,[n1 1 n3]);
            for r=0:1-p-q
                v3c=repmat(v3.^r,[n1 n2 1]);
                if h>0
                    waitbar(pp/ppn,h);
                end
                pp=pp+1;
                mp(p+1,q+1,r+1)=sum(v1c(:).*v2c(:).*v3c(:).*image(:));
            end
        end
    end
    if mp(1,1,1)==0
        t1=(n1+1)/2;
        t2=(n2+1)/2;
        t3=(n3+1)/2;
    else
        t1=mp(2,1,1)/mp(1,1,1);
        t2=mp(1,2,1)/mp(1,1,1);
        t3=mp(1,1,2)/mp(1,1,1);
    end
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
mr=zeros(or+1,or+1,or+1);
for p=0:or
    v1c=v1a(:,:,:,p+1);
    for q=0:or-p
        v2c=v2a(:,:,:,q+1);
        for r=0:or-p-q
            v3c=v3a(:,:,:,r+1);
            if h>0
                waitbar(pp/ppn,h);
            end
            pp=pp+1;
            mr(p+1,q+1,r+1)=sum(v1c(:).*v2c(:).*v3c(:).*image(:));
        end
    end
end
 
