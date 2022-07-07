function [e,ew]=cafmi3d(f,m,typ)
% [e,ew]=cafmi3d(f,m,typ) computes values of the invariants in the array f
% from the values of the geometric moments in the 3D array m and store them
% to the vector e.
% ew contains weights of the invariants.
% If typ==1, the moments are normalized to scaling as the volume moments,
% if typ==2, they are normalized as surface moments

s=size(f);
if size(s)<3
    s(3)=1;
end
ms=size(m);
v=4-typ;

order=min(ms)-1;
mr=m;
m000=mr(1,1,1);
for p=0:order;
    for q=0:order-p;
        for r=0:order-p-q;
            	%normalization to scaling
            mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)/m000^((p+q+r+v)/v);
                %magnitude normalization
%            mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)*pi^((p+q+r)/v)*((p+q+r)/v+1); %ze 2D
            if typ==1    %volume moments
                 mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)*pi^((p+q+r)/6)*((p+q+r)+3)/1.5^((p+q+r)/3+1);
            elseif typ==2   %surface moments
                 mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)*pi^((p+q+r)/4)*2^((p+q+r)/2);
            end
        end
    end
end

for i1=1:s(3)
    k=1;
    while k<s(1) && f(k,1,i1)~=0	%How many coefficients ?
        k=k+1;
    end
    if f(k,1,i1)==0
        k=k-1;
    end
    j=1;
    while j<(s(2)-1)/3 && f(1,3*j-1,i1)+f(1,3*j,i1)+f(1,3*j+1,i1)~=0	%How many moments is in one term ?
        j=j+1;
    end
    if f(1,3*j-1,i1)+f(1,3*j,i1)+f(1,3*j+1,i1)==0
        j=j-1;
    end
    maux=zeros(j,k);
    for p=1:k
        for q=2:3:3*j-1
            maux((q+1)/3,p)=mr(f(p,q,i1)+1,f(p,q+1,i1)+1,f(p,q+2,i1)+1);
        end
    end
    if j>1
        maux=prod(maux);
    end

    e(i1)=maux*f(1:k,1,i1);
    ew(i1)=sum(f(1,2:3*j+1,i1))/v;	%weight of the invariant
%    e(i1)=sign(e(i1))*abs(e(i1))^(1/ew(i1)); %magnitude normalization to degree by weight
    e(i1)=sign(e(i1))*abs(e(i1))^(1/j); %magnitude normalization to degree by degree
end