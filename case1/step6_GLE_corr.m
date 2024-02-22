function step6_GLE_corr(name,save_name,loop_num,sample_per_loop)
M=0;x_mean=0;
parfor i=1:loop_num
    if ~exist(feval(name,i),'file');continue;end
    data=load(feval(name,i),'x');
    x_mean=x_mean+mean(data.x,'all');
    M=M+1;
end
x_mean=x_mean/M;

M=0;
load('data/PDF.mat','dt','ff_x','bd')
load('data/corr.mat','xcorr_x')
N_corr = 400/dt;
N_xcorr = 3/dt;

pdf=0;
corr=0;xcorr=0;xcorr_count=0;
t_A_B=cell(1,loop_num);t_B_A=cell(1,loop_num);
tic
for i=1:loop_num
    if ~exist(name(i),'file');continue;end
    M=M+sample_per_loop;
    data=load(name(i));
    x_tmp=data.x;    v_tmp=data.v;
    t_A_B_tmp=cell(1,sample_per_loop);    t_B_A_tmp=cell(1,sample_per_loop);
    parfor j=1:sample_per_loop 
        x=x_tmp(:,j);
        v=v_tmp(:,j);
        pdf=pdf+ksdensity(x,ff_x,'Bandwidth',bd);
        [corr_tmp,xcorr_tmp,xcorr_count_tmp,t_A_B_tmp{j},t_B_A_tmp{j}] = ...
            compute_correlation_function(x,v,x_mean,xcorr_x,N_corr,N_xcorr);
        corr=corr+corr_tmp;
        xcorr=xcorr+xcorr_tmp;
        xcorr_count=xcorr_count+xcorr_count_tmp;
    end
    t_A_B{i}=cell2mat(t_A_B_tmp);
    t_B_A{i}=cell2mat(t_B_A_tmp);
    disp([num2str(i),' ',num2str(toc)])
end

t_A_B=cell2mat(t_A_B);
t_B_A=cell2mat(t_B_A);

pdf=pdf/M;
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

save(save_name,Vars1{:},Vars2{:},'t_A_B','t_B_A','pdf')
end

