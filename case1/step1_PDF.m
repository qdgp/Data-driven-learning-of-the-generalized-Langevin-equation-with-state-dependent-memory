function step1_PDF(M)
dt=0.01;
kT=2.4750;
dx=0.005;
bd=0.002;
ff_x=(2.8:dx:4.1)';

v_square=0;pdf=0;x_mean=0;
parfor i=1:M
    data=load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
    x_mean=x_mean+mean(data.x,'all');
    pdf=pdf+ksdensity(data.x,ff_x,'Bandwidth',bd);
    v_square=v_square+mean((data.v).^2,'all');
end
pdf=pdf/M;
x_mean=x_mean/M;
v_square=v_square/M;
mass = kT/v_square;
ff_f = -derivative_1st(dx,-log(pdf));
save('data/PDF.mat','ff_x','ff_f','bd','dt','pdf','kT','mass','v_square','x_mean','M')
end