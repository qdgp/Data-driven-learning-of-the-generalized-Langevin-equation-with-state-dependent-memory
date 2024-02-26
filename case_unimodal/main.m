M=128;
% step0_collect_data(M); % get data from MD trajectories
step1_PDF(M); % compute Probability Distribution Function with ksdensity
step2_hx(); % compute h(x)

%% compute correlation functions of x,v,a,f
step3_corr();

%% compute correlation functions with h(x) to eval 1D state-dependent kernel
step4_hx_corr;

%% compute trainging set for ND state-dependent kernel
folder='/mnt/gs21/scratch/gepei/GLE_state_dependent/GLE_case2_training_set/';
for i=1:M
    step3_training_set(folder,i);
end
file_name='dx_10_w_501';
step4_collect_training_set(folder,file_name);

%% train model on Python
% excute 'python3 train.py'

%% simulation of standard GLE and state-dependent GLE
sample_per_loop=32;
loop_num=32;
for i=1:loop_num
    step5_std_GLE(100,1,sample_per_loop,['GLE_data/GLE_1D_rng_',num2str(i),'.mat'],i);
    step5_hx_GLE_1D(100,1,sample_per_loop,['GLE_data/hx_GLE_1D_rng_',num2str(i),'.mat'],i);
    step5_hx_GLE_ND(100,1,sample_per_loop,'ML_ND_2.mat',['GLE_data/hx_GLE_2D_rng_',num2str(i),'.mat'],i);
end

%% compute correlation functions of standard GLE and state-dependent GLE
sample_per_loop=32;
loop_num=32;
name=@(i) ['GLE_data/GLE_1D_rng_',num2str(i),'.mat'];
save_name='data/corr_GLE.mat';
step6_GLE_corr(name,save_name,loop_num,sample_per_loop);

name=@(i) ['GLE_data/hx_GLE_1D_rng_',num2str(i),'.mat'];
save_name='data/corr_hx_GLE_1D.mat';
step6_GLE_corr(name,save_name,loop_num,sample_per_loop);

name=@(i) ['GLE_data/hx_GLE_2D_rng_',num2str(i),'.mat'];
save_name='data/corr_hx_GLE_2D.mat';
step6_GLE_corr(name,save_name,loop_num,sample_per_loop);

%% visualize the results
close all

load('data/PDF.mat')
corr1=load('data/corr.mat');
corr2=load('data/corr_GLE.mat');
corr3=load('data/corr_hx_GLE_1D.mat');
corr4=load('data/corr_hx_GLE_2D.mat');

figure(1);hold on;box on
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
plot(ff_x,pdf)
title('PDF')
xlabel('x');ylabel('\rho (x)')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/PDF.png')

figure(2);hold on;box on
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
corr_ver='corr_vv';
plot(corr1.corr_t,corr1.(corr_ver),'Displayname','MD')
plot(corr2.corr_t,corr2.(corr_ver),'Displayname','GLE')
plot(corr3.corr_t,corr3.(corr_ver),'Displayname','SD-GLE-1D')
plot(corr4.corr_t,corr4.(corr_ver),'Displayname','SD-GLE-2D')
xlim([0,40])
legend
title('<v(t),v(0)>')
xlabel('x');ylabel('\rho (x)')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/corr_vv.png')

figure(3);hold on;box on
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
corr_ver='xcorr_vv';
i=1;
for bin=8:8:32
    subplot(2,2,i);i=i+1;
    hold on;box on;
    title(['x=[',num2str(hx_x(bin)),',',num2str(hx_x(bin)+hx_x(2)-hx_x(1)),']'])
    set(gca,'ColorOrderIndex',1)
    plot(corr1.xcorr_t,corr1.(corr_ver)(bin,:),'Displayname','MD')
    plot(corr2.xcorr_t,corr2.(corr_ver)(bin,:),'Displayname','GLE')
    plot(corr3.xcorr_t,corr3.(corr_ver)(bin,:),'Displayname','SD-GLE-1D')
    plot(corr4.xcorr_t,corr4.(corr_ver)(bin,:),'Displayname','SD-GLE-2D')
    legend
    set(gca,'FontSize',16,'LineWidth',2.0)
end
saveas(gcf,'fig/xcorr_vv.png')

figure(4);hold on;box on
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
tail_x=(0:500)*0.02;bd=0.08;
plot(tail_x,ksdensity(corr1.tail,tail_x,'Bandwidth',bd),'Displayname','MD')
plot(tail_x,ksdensity(corr2.tail,tail_x,'Bandwidth',bd),'Displayname','GLE')
plot(tail_x,ksdensity(corr3.tail,tail_x,'Bandwidth',bd),'Displayname','SD-GLE-1D')
plot(tail_x,ksdensity(corr4.tail,tail_x,'Bandwidth',bd),'Displayname','SD-GLE-2D')
xlim([0,5])
set(gca, 'YScale', 'log')
legend
title('Distribution for the time that x>15')
xlabel('time period');ylabel('distribution')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/tail.png')

%%
close all
figure(5);hold on;box on
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
tail_x=(0:500)*0.02;bd=0.08;
plot(tail_x,ksdensity(corr1.tail,tail_x,'Bandwidth',bd),'Displayname','MD')
plot(tail_x,ksdensity(corr2.tail,tail_x,'Bandwidth',bd),'Displayname','GLE')
plot(tail_x,ksdensity(corr3.tail,tail_x,'Bandwidth',bd),'Displayname','SD-GLE-1D')
plot(tail_x,ksdensity(corr4.tail,tail_x,'Bandwidth',bd),'Displayname','SD-GLE-2D')
xlim([0,5])
set(gca, 'YScale', 'log', 'XScale', 'log')
legend('Location','Best')
xlabel('time period');ylabel('distribution')
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/tail_v2.png')

figure(6);hold on;box on
set(gcf, 'DefaultLineLineWidth', 3.0,'DefaultLineMarkerSize',12);
plot(ff_x,-log(pdf))
ylim([0,15])
xlabel('$q$','Interpreter','latex');
ylabel('$U(q)/k_BT$','Interpreter','latex')
xticks([0,10,20])
yticks([0,5,10,15])
set(gca,'FontSize',30,'LineWidth',2.0)
saveas(gcf,'fig/FreeEnergy.png')









