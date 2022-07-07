function [ y ] = pochham( x,k )
%It computes Pochhammer symbol (x)_k=x(x+1)(x+2)...(x+k-1)

if numel(x)==1 && numel(k)==1
    if k-floor(k)==0
        y=1;
        for m=1:k
            y=y*x;
            x=x+1;
        end
    else
        y=gamma(x+k)/gamma(x);
    end
elseif numel(k)==1
    y=zeros(size(x));
    for i=1:numel(x)
        xi=x(i);
        if k-floor(k)==0
            yi=1;
            for m=1:k
                yi=yi*xi;
                xi=xi+1;
            end
        else
            yi=gamma(xi+k)/gamma(xi);
        end
        y(i)=yi;
    end
elseif numel(x)==1 
    y=zeros(size(k));
    for j=1:numel(k)
        xi=x;
        kj=k(j);
        if kj-floor(kj)==0
            yj=1;
            for m=1:kj
                yj=yj*xi;
                xi=xi+1;
            end
        else
            yj=gamma(x+kj)/gamma(x);
        end
        y(j)=yj;
    end
else
    y=zeros(numel(x),numel(k));
    for i=1:numel(x)
        for j=1:numel(k)
            xi=x(i);
            kj=k(j);
            if kj-floor(kj)==0
                yij=1;
                for m=1:kj
                    yij=yij*xi;
                    xi=xi+1;
                end
            else
                yij=gamma(xi+kj)/gamma(xi);
            end
            y(i,j)=yij;
        end
    end
end

end

