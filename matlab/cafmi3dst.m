function [e,ew]=cafmi3dst(coef, ind, m, typs, typg)
% [e,ew]=cafmi3dst(f,m,typ) computes values of the invariants in the array of
% structures f from the values of the geometric moments in the 3D array m
% and store them to the vector e.
% ew contains weights of the invariants.
% If typs==0, no normalization to scaling is used,
% if typs==1, the moments are normalized to scaling as the volume moments,
% if typs==2, they are normalized as surface moments.
% if typs==3, the volume moments are normalized to scaling without magnitude normalization to order,
% if typs==4, the surface moments are normalized to scaling without magnitude normalization to order.
% If typg==0, no final magnitude normalization is used,
% If typg==1, the magnitude normalization to weight is used,
% if typg==2, the magnitude normalization to degree is used.

if nargin<4
    typg=2;
end
if nargin<3
    typs=1;
end

ms=size(m);

order=min(ms)-1;
mr=m;
m000=mr(1,1,1);
if typs>0
    v=2+mod(typs,2);
    for p=0:order
        for q=0:order-p
            for r=0:order-p-q
                    %normalization to scaling
                mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)/m000^((p+q+r+v)/v);
                    %magnitude normalization
    %            mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)*pi^((p+q+r)/v)*((p+q+r)/v+1); %ze 2D
                if typs==1    %volume moments
                     mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)*pi^((p+q+r)/6)*((p+q+r)+3)/1.5^((p+q+r)/3+1);
                elseif typs==2   %surface moments
                     mr(p+1,q+1,r+1)=mr(p+1,q+1,r+1)*pi^((p+q+r)/4)*2^((p+q+r)/2);
                end
            end
        end
    end
else
    v=3;
end

for i1=1:length(ind)
%     k=length(f(i1).coef);
%     maux=zeros(j,k);
%     for p=1:k
%         for q=1:3:3*j-2
%             maux((q+2)/3,p)=mr(f(i1).ind(p,q)+1,f(i1).ind(p,q+1)+1,f(i1).ind(p,q+2)+1);
%         end
%     end
%     if j>1
%         maux=prod(maux);
%     end
%     e(i1)=sum(maux.*f(i1).coef);

    i = [ind(i1){:}];
    j=size(i,2)/3;
    inds = sub2ind(size(mr),i(:,1:3:3*j-2)+1,i(:,2:3:3*j-1)+1,i(:,3:3:3*j)+1);
    e(i1)=sum(prod([[coef(i1){:}]',mr(inds)],2));

    ew(i1)=sum(i(1,:))/v;	%weight of the invariant
    if typg==1
        e(i1)=sign(e(i1))*abs(e(i1))^(1/ew(i1)); %magnitude normalization to weight
    elseif typg==2
        e(i1)=sign(e(i1))*abs(e(i1))^(1/j); %magnitude normalization to degree
    end
end
