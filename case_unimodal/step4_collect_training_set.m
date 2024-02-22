function step4_collect_training_set(folder,file_name)
load('data/PDF.mat','M')

Px_vv0_Px=0;Px_vv0_Px_overall=0;Px_vx0_Px_overall=0;
maf_v0=0;maf_v0_count=0;
parfor i=1:M
    data=load([folder,file_name,'_',num2str(i),'.mat'],'Px_vv0_Px','maf_v0','maf_v0_count','Px_vv0_Px_overall','Px_vx0_Px_overall','Px_x');
    Px_vv0_Px=Px_vv0_Px+data.Px_vv0_Px;
    maf_v0=maf_v0+data.maf_v0;
    maf_v0_count=maf_v0_count+data.maf_v0_count;
    Px_vv0_Px_overall=Px_vv0_Px_overall+data.Px_vv0_Px_overall;
    Px_vx0_Px_overall=Px_vx0_Px_overall+data.Px_vx0_Px_overall;
    disp(i)
end
load([folder,file_name,'_1.mat'],'Px_x')
Px_vv0_Px=Px_vv0_Px/M;
Px_vv0_Px_overall=Px_vv0_Px_overall/M;
Px_vx0_Px_overall=Px_vx0_Px_overall/M;
maf_v0=maf_v0./maf_v0_count;

save(['./data/',file_name,'.mat'],'Px_vv0_Px_overall','Px_vx0_Px_overall','Px_vv0_Px','maf_v0','Px_x','-v7.3')
end








