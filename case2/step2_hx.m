function step2_hx()
load('data/PDF.mat','ff_x','ff_f','kT','dt','v_square','mass','M');
dx=0.5;
hx_x=(1.5:dx:17)';
x0=hx_x(1)-dx/2;
Nbin=length(hx_x);
hx=zeros(Nbin,2);
parfor i=1:M
    data=load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
    a_tmp = derivative_1st(dt, data.v);
    f_tmp = kT*interp1(ff_x,ff_f, data.x, 'linear','extrap');

    hx_square=(mass*a_tmp.*a_tmp-f_tmp.*a_tmp)/v_square;
    x_index=floor((data.x-x0)./dx)+1;
    hx_tmp=zeros(Nbin,2);
    for j=1:Nbin
        index=x_index==j;
        hx_tmp(j,1)=sum(hx_square(index));
        hx_tmp(j,2)=sum(index);
    end
    hx=hx+hx_tmp;
end
hx_raw=hx;
hx=hx(:,1)./hx(:,2);
hx=sqrt(hx);

save('data/PDF.mat','hx_x','hx','hx_raw','-append')
end


