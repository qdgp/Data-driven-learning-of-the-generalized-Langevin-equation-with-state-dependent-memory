


i=str2double(getenv('SLURM_ARRAY_TASK_ID'));

sample_per_loop=32;
Folder='/mnt/gs21/scratch/gepei/GLE_state_dependent/case1_GLE_data/';

% if ~exist([Folder,'GLE_1D_rng_',num2str(i),'.mat'],'file')
%     step5_std_GLE(40,1,sample_per_loop,[Folder,'GLE_1D_rng_',num2str(i),'.mat'],i);
% end
% if ~exist([Folder,'GLE_4D_std_rng_',num2str(i),'.mat'],'file')
%     step5_hx_GLE(40,1,sample_per_loop,'ML_ND_4_std.mat',[Folder,'GLE_4D_std_rng_',num2str(i),'.mat'],i);
% end
% if ~exist([Folder,'GLE_4D_rng_',num2str(i),'.mat'],'file')
%     step5_hx_GLE(40,1,sample_per_loop,'ML_ND_4.mat',[Folder,'GLE_4D_rng_',num2str(i),'.mat'],i);
% end
if ~exist([Folder,'GLE_v2_4D_rng_',num2str(i),'.mat'],'file')
    step5_hx_GLE(40,1,sample_per_loop,'ML_ND_4_v2.mat',[Folder,'GLE_4D_rng_',num2str(i),'.mat'],i);
end
