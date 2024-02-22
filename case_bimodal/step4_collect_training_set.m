function step4_collect_training_set(folder)
N_full=50;
file_name='dx_0.2_w_301';

M=512;
Px_vv0_Px=0;PP_corr=0;PPv_corr=0;
maf_v0=0;maf_v0_count=0;
parfor i=1:M
    data=load([folder,file_name,'_',num2str(i),'.mat'],'Px_vv0_Px','PP_corr','PPv_corr','maf_v0','maf_v0_count');
    Px_vv0_Px=Px_vv0_Px+data.Px_vv0_Px;
    maf_v0=maf_v0+data.maf_v0;
    maf_v0_count=maf_v0_count+data.maf_v0_count;
    PP_corr=PP_corr+data.PP_corr;
    PPv_corr=PPv_corr+data.PPv_corr;
    disp(i)
end
load([folder,file_name,'_1.mat'],'Px_x')
Px_vv0_Px=Px_vv0_Px/M;
PP_corr=PP_corr/M;
PPv_corr=PPv_corr/M;
maf_v0=maf_v0./maf_v0_count;


N_corr=size(PP_corr,3);
PP_corr_xi=zeros(size(PP_corr,1),2,N_corr-N_full);
for i=N_full+1:N_corr
    tmp=PP_corr(:,:,i);
    [V,D] = eig(tmp);
    for j=1:2
        PP_corr_xi(:,j,i-N_full)=V(:,j)*sqrt(D(j,j));
    end
end
PP_corr=PP_corr(:,:,1:N_full);

save(['./data/',file_name,'.mat'],'Px_vv0_Px','PPv_corr','PP_corr_xi','PP_corr','maf_v0','Px_x','-v7.3')
end











