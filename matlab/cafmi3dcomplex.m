function [e,ew]=cafmi3dcomplex(f,m,typs,typg)
% [e,ew]=cafmi3dcomplex(f,m) computes values of the invariants
% in the array of structures f from the values of the complex moments 
% in the 3D array m and store them to the vector e
% ew contains weights of the invariants. 
% If typs==0, no normalization to scaling is used, 
% if typs==1, the moments are normalized to scaling as the volume moments, 
% if typs==2, they are normalized as surface moments.
% if typs==3, the volume moments are normalized to scaling without magnitude normalization to order, 
% if typs==4, the surface moments are normalized to scaling without magnitude normalization to order.
% If typg==0, no magnitude normalization is used, 
% If typg==1, the magnitude normalization to weight is used, 
% if typg==2, the magnitude normalization to degree is used.

order=size(m,1)-1;
mr=m;
m000=abs(mr(1,1,1));
if typs==1 || typs==3  %volume moments
    for p=0:order
            %normalization to scaling
        mr(p+1,:,:)=mr(p+1,:,:)/m000^(p/3+1);
        if typs==1
            %magnitude normalization
            mr(p+1,:,:)=mr(p+1,:,:)*pi^(p/6)*(p+3)/1.5^(p/3+1)/2;
        end
    end
elseif typs==2 || typs==4   %surface moments
    for p=0:order
            %normalization to scaling
        mr(p+1,:,:)=mr(p+1,:,:)/m000^(p/2+1);
        if typs==2
            %magnitude normalization
            mr(p+1,:,:)=mr(p+1,:,:)*pi^(p/4)*2^(p/2);
        end
    end
end
e=zeros(1,length(f));
ew=zeros(1,length(f));
for i1=1:length(f)
    if i1==4
        p=0;
    end
    k=length(f(i1).coef);
    j=size(f(i1).ind,2)/3;
    maux=zeros(j,k);
    for p=1:k
        for q=1:3:3*j-2
%             disp([i1,p,q])
%             disp([f(i1).ind(p,q)+1,floor(f(i1).ind(p,q+1)/2)+1,f(i1).ind(p,q+2)+f(i1).ind(p,q+1)+1])
            maux((q+2)/3,p)=mr(f(i1).ind(p,q)+1,floor(f(i1).ind(p,q+1)/2)+1,f(i1).ind(p,q+2)+f(i1).ind(p,q+1)+1);
        end
    end
    if j>1
        maux=prod(maux);
    end

%    e(i1)=real(maux*(f(i1).coef)');
    e(i1)=sum(real(maux.*f(i1).coef));
    ew(i1)=sum(f(i1).ind(1,1:3:3*j-2))/3;	%weight of the invariant
    if typg==1
        e(i1)=sign(e(i1))*abs(e(i1))^(1/ew(i1)); %magnitude normalization to weight
    elseif typg==2
        e(i1)=sign(e(i1))*abs(e(i1))^(1/j); %magnitude normalization to degree
    end
end

