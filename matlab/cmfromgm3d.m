function mr=cmfromgm3d(gm,or)
%mr=cmfromgm3d(gm,or)
%compute complex 3D moments c_{es,el}^{em} from geometric  m_{pqr} up to the order 'or'.
% The moment m_{pqr} = gm(p+1,q+1,r+1)
% The moment c_{es,el}^{em} = mr(es+1,floor(el/2)+1,em+el+1)
% i.e. In mr(p,q,r), there is c_{es,el}^{em}, where es=p-1,
% el=2*(q-1)+mod(es,2) and em=r-1-el.

or=round(or);
%sza=sprintf('Výpoèet 3D komplexních momentù do øádu %d z geometrických',or);
%h=waitbar(0,sza);

warning('OFF','MATLAB:gcd:largestFlint');
mr =zeros(or+1,floor(or/2)+1,2*or+1);

pp=1;
%ppn=nchoosek(or+3,3);
for es=0:or
    for el=rem(es,2):2:es
        for em=-el:el
%            waitbar(pp/ppn,h);
            pp=pp+1;
%            disp([es,el,em]);
            s3ctgc=symb3d_complex_to_geomt(es,el,em);
            k=length(s3ctgc.coef);
            maux=zeros(1,k);
            for p=1:k
                maux(p)=gm(s3ctgc.ind(p,1)+1,s3ctgc.ind(p,2)+1,s3ctgc.ind(p,3)+1);
            end
            v=sum(maux.*s3ctgc.coef);
            mr(es+1,floor(el/2)+1,em+el+1)=v;
        end
    end
end
pp=pp-1;
 
%close(h);
