function step4_hx_corr()
load('data/PDF.mat','ff_x','ff_f','kT','dt','hx_x','hx','mass','x_mean','M')
hx_x=[0;hx_x;40];
hx=[hx(1);hx;hx(end)];

load('data/corr.mat','N_corr','corr_t')

corr_invh_maf_hx=zeros(2*N_corr+1,1);
corr_invh_maf_hv=zeros(2*N_corr+1,1);
corr_invh_maf_x=zeros(2*N_corr+1,1);
corr_invh_maf_v=zeros(2*N_corr+1,1);

corr_hv_hx=zeros(2*N_corr+1,1);
corr_hv_hv=zeros(2*N_corr+1,1);
corr_hv_x=zeros(2*N_corr+1,1);
corr_hv_v=zeros(2*N_corr+1,1);

parfor i=1:M
    data=load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
    x_tmp=data.x;
    v_tmp=data.v;
    hx_tmp= interp1(hx_x,hx, x_tmp, 'linear','extrap');
    a_tmp = derivative_1st(dt, v_tmp);
    f_tmp = kT*interp1(ff_x,ff_f, x_tmp, 'linear','extrap');
    
    x_tmp=x_tmp-x_mean;
    
    invh_maf_tmp=(mass*a_tmp-f_tmp)./hx_tmp;
    hx_x_tmp=x_tmp.*hx_tmp;
    hx_v_tmp=v_tmp.*hx_tmp;
    
    corr_invh_maf_hx = corr_invh_maf_hx + xcorr(invh_maf_tmp, hx_x_tmp, N_corr,'unbiased');
    corr_invh_maf_hv = corr_invh_maf_hv + xcorr(invh_maf_tmp, hx_v_tmp, N_corr,'unbiased');
    corr_invh_maf_x = corr_invh_maf_x + xcorr(invh_maf_tmp, x_tmp, N_corr,'unbiased');
    corr_invh_maf_v = corr_invh_maf_v + xcorr(invh_maf_tmp, v_tmp, N_corr,'unbiased');

    corr_hv_hx = corr_hv_hx + xcorr(hx_v_tmp, hx_x_tmp, N_corr,'unbiased');
    corr_hv_hv = corr_hv_hv + xcorr(hx_v_tmp, hx_v_tmp, N_corr,'unbiased');
    corr_hv_x = corr_hv_x + xcorr(hx_v_tmp, x_tmp, N_corr,'unbiased');
    corr_hv_v = corr_hv_v + xcorr(hx_v_tmp, v_tmp, N_corr,'unbiased');
end

corr_invh_maf_hx=corr_invh_maf_hx(N_corr+1:end)/M;
corr_invh_maf_hv=corr_invh_maf_hv(N_corr+1:end)/M;
corr_invh_maf_x=corr_invh_maf_x(N_corr+1:end)/M;
corr_invh_maf_v=corr_invh_maf_v(N_corr+1:end)/M;

corr_hv_hx=corr_hv_hx(N_corr+1:end)/M;
corr_hv_hv=corr_hv_hv(N_corr+1:end)/M;
corr_hv_x=corr_hv_x(N_corr+1:end)/M;
corr_hv_v=corr_hv_v(N_corr+1:end)/M;

Vars=who('corr_*');

save('data/hx_corr.mat',Vars{:})
end
