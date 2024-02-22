function step2_std_corr()
load('data/PDF.mat','M','dt','x_mean')
N_corr = 400/dt;
N_xcorr = 3/dt;
xcorr_x=(2.6:0.05:4.2)';
corr=0;xcorr=0;xcorr_count=0;
t_A_B=cell(1,M);t_B_A=cell(1,M);

parfor i=1:M
    data=load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
    [corr_tmp,xcorr_tmp,xcorr_count_tmp,t_A_B{i},t_B_A{i}] = ...
        compute_correlation_function(data.x,data.v,x_mean,xcorr_x,N_corr,N_xcorr);
    corr=corr+corr_tmp;
    xcorr=xcorr+xcorr_tmp;
    xcorr_count=xcorr_count+xcorr_count_tmp;
    disp(i)
end

t_A_B=cell2mat(t_A_B);
t_B_A=cell2mat(t_B_A);

corr_xx=corr(:,1)/M;corr_vv=corr(:,2)/M;
corr_aa=corr(:,3)/M;corr_vx=corr(:,4)/M;
corr_fx=corr(:,5)/M;corr_fv=corr(:,6)/M;
corr_av=corr(:,7)/M;corr_fa=corr(:,8)/M;

xcorr_vv=xcorr(:,:,1)./xcorr_count;
xcorr_vx=xcorr(:,:,2)./xcorr_count;
xcorr_fx=xcorr(:,:,3)./xcorr_count;
xcorr_afx=xcorr(:,:,4)./xcorr_count;
xcorr_xx=xcorr(:,:,5)./xcorr_count;

corr_t=(-N_corr:N_corr)'*dt;
xcorr_t=(0:N_xcorr-1)'*dt;

Vars1=who('corr_*');
Vars2=who('xcorr_*');

save('data/corr.mat',Vars1{:},Vars2{:},'t_A_B','t_B_A')
end

