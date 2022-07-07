function f=readinv3dst(fname)
%f=readinv(fname) reads 3d invariants from the file fname and store them
% to the array of structures f
fid=fopen(fname,'rt');
if fid==-1
    return
end
i=1;
while(~feof(fid))
    m=fscanf(fid,'%d',1);   %weight
    c=fscanf(fid,'%d',1);   %number of terms
    r=fscanf(fid,'%d',1);   %degree
    f(i).weight=m;
    f(i).nterms=c;
    f(i).degree=r;
%    [i,m,c,r]
    for k=1:c
        f(i).coef(k)=fscanf(fid,'%g',1);   %coefficients
        for l=1:3*r
            v=fscanf(fid,'%d',1);
            f(i).ind(k,l)=v;  %indices
        end
    end
    i=i+1;
end
fclose(fid);


