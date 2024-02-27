%% For generating training sets and simulations.

parpool(16)   % use parfor with 16 threads
M=512; % specify the number of trajectories 
step1_PDF(M);  % compute PDF with ksdensity
step2_std_corr(); % compute correlation function and state-dependent correlation function

%% generate training set (three points correlation functions and others)
folder='/mnt/gs21/scratch/gepei/GLE_state_dependent/case1_training_set/';  
parfor i=1:M
    step3_training_set(folder,i);
end
step4_collect_training_set(folder);

%%
% train model with tensorflow
% excute 'python3 train.py'

%%  Perform standard and state-dependent GLE simulation
sample_per_loop=32;
loop_num=100;
Folder='/mnt/gs21/scratch/gepei/GLE_state_dependent/case1_GLE_data/';
parfor i=1:loop_num
    step5_std_GLE(40,1,sample_per_loop,[Folder,'GLE_1D_rng_',num2str(i),'.mat'],i);
    step5_hx_GLE(40,1,sample_per_loop,'ML_ND_4_lite.mat',[Folder,'GLE_4D_rng_',num2str(i),'.mat'],i);
    % replace 'ML_ND_4_lite.mat' to other saved file name to use different models
end

%% compute correlation function from GLE models
name=@(i) [Folder,'GLE_1D_rng_',num2str(i),'.mat'];
save_name='data/corr_GLE.mat';
step6_GLE_corr(name,save_name,loop_num,sample_per_loop)

name=@(i) [Folder,'GLE_4D_rng_',num2str(i),'.mat'];
save_name='data/corr_ML_4D.mat';
step6_GLE_corr(name,save_name,loop_num,sample_per_loop)

%% check result
corr1=load('data/corr.mat');
corr2=load('data/corr_GLE.mat');
corr3=load('data/corr_ML_4D.mat');
corr4=load('data/corr_ML_4D_std.mat');
PDF=load('data/PDF.mat');

close all
figure(1);hold on;box on;
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
plot(PDF.ff_x,PDF.pdf)
xlim([2.8,4.1])
title('Probability Distribution','Interpreter','latex')
xlabel('$q$','Interpreter','latex')
ylabel('$\rho(q)$','Interpreter','latex')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/PDF.png')

figure(2);hold on;box on;
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
plot(PDF.ff_x,-log(PDF.pdf))
xlim([2.8,4.1])
title('Free Energy','Interpreter','latex')
xlabel('$q$','Interpreter','latex');
ylabel('$U(q)/k_BT$','Interpreter','latex')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/FreeEnergy.png')

figure(3);hold on;box on;
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
corr_ver='corr_vv';
plot(corr1.corr_t,corr1.(corr_ver),'Displayname','MD')
plot(corr2.corr_t,corr2.(corr_ver),'Displayname','GLE')
plot(corr3.corr_t,corr3.(corr_ver),'Displayname','SD-GLE')
xlim([0,3])
legend
title('Velocity Correlation','Interpreter','latex')
xlabel('$t$','Interpreter','latex');
ylabel('$\langle v(t),v(0) \rangle$','Interpreter','latex')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/corr_vv.png')

fig=figure(4);hold on;box on;
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
corr_ver='xcorr_vv';
i=1;
for bin=[12,19,22,27]
    subplot(2,2,i);i=i+1;
    hold on;box on;
    title(['$q^* \in [',num2str(hx_x(bin)),',',num2str(hx_x(bin)+hx_x(2)-hx_x(1)),']$'],'Interpreter','latex')
    set(gca,'ColorOrderIndex',1)
    plot(corr1.xcorr_t,corr1.(corr_ver)(bin,:),'Displayname','MD')
    plot(corr2.xcorr_t,corr2.(corr_ver)(bin,:),'Displayname','GLE')
    plot(corr3.xcorr_t,corr3.(corr_ver)(bin,:),'Displayname','SD-GLE')
    set(gca,'FontSize',16,'LineWidth',2.0)
    legend
end
axs=axes(fig,'visible','off'); 
axs.Title.Visible='on';
axs.XLabel.Visible='on';
axs.YLabel.Visible='on';
ylabel(axs,'$\langle v(t),v(0) |q(0)=q^* \rangle$','Interpreter','latex');
xlabel(axs,'$t$','Interpreter','latex');
set(axs,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/xcorr_vv.png')

%%
close all
figure(5);hold on;box on;
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
t_sample = 10:10:8000;bd=80;
plot(t_sample,ksdensity(corr1.t_A_B,t_sample,"Bandwidth",bd),'Displayname','MD')
plot(t_sample,ksdensity(corr2.t_A_B,t_sample,"Bandwidth",bd),'Displayname','GLE')
plot(t_sample,ksdensity(corr3.t_A_B,t_sample,"Bandwidth",bd),'Displayname','SD-GLE')
set(gca, 'YScale', 'log')
ylim([10^-6,10^-2])
xlim([10,4000])
legend
xlabel('time period','Interpreter','latex');ylabel('distribution','Interpreter','latex')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/trates.png')



