function step1_PDF(M)
dt=0.02;
kT=1;
dx=0.10;
bd=0.05;
ff_x=(1:dx:20)';

v_square=0;pdf=0;x_mean=0;
parfor i=1:M
    data=load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
    x_mean=x_mean+mean(data.x,'all');
    pdf=pdf+ksdensity(data.x,ff_x,'Bandwidth',bd);
    v_square=v_square+mean((data.v).^2,'all');
end
pdf=pdf/M;
x_mean=x_mean/M;
v_square = v_square/M;
mass = kT/v_square;

ff_f= -derivative_1st(dx,-log(pdf));
p=polyfit(ff_x(ff_x>16 & ff_x <18),ff_f(ff_x>16 & ff_x <18),2);
ff_f(ff_x>16)=polyval(p,ff_x(ff_x>16));

save('data/PDF.mat','ff_x','ff_f','bd','dt','pdf','kT','mass','v_square','x_mean','M')
end