
i=str2double(getenv('SLURM_ARRAY_TASK_ID'));
sample_per_loop=32;
Folder='GLE_data/';

% step5_std_GLE(100,1,sample_per_loop,[Folder,'GLE_1D_rng_',num2str(i),'.mat'],i);
% step5_hx_GLE_1D(100,1,sample_per_loop,[Folder,'hx_GLE_1D_rng_',num2str(i),'.mat'],i);
step5_hx_GLE_ND(150,1,sample_per_loop,'ML_ND_2.mat',[Folder,'hx_GLE_2D_rng_',num2str(i),'.mat'],i);