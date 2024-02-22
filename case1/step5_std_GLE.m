function step5_std_GLE(T_scale,N_per_step,M,save_file_name,random_seed)
theta_cut=80;
load('data/PDF.mat','mass','ff_x','ff_f','kT','v_square','dt');
load('data/corr.mat','corr_av','corr_fv','corr_vv');
fx=@(x) kT*interp1(ff_x,ff_f, x, 'linear','extrap');

T_cut=200;T=T_cut*T_scale;
dw=2*pi/T/2;

N_sample = T_cut/dt-1; 
g = mass*corr_av((end+1)/2:end) - corr_fv((end+1)/2:end); 
h = corr_vv((end+1)/2:end);
g=g(1:N_sample+1);g(1)=g(1)/2;G=fft(g,N_sample*2-1);
h=h(1:N_sample+1);h(1)=h(1)/2;H=fft(h,N_sample*2-1);
Theta=2.0*max(real(-G./H),0);
Theta=Theta(1:floor(60/(pi/T_cut)));

NTheta=size(Theta,1);
Theta=interp1((1:NTheta)',Theta,(1:1/T_scale:NTheta)','linear','extrap');

dt=dt/N_per_step; N=T/dt;Ntheta=int64(theta_cut/dt);
theta=2.0*real(ifft(Theta,N*2+1))/dt;
theta=theta(1:Ntheta,:);

Ex=(cumsum(ff_f)-ff_f(1)/2-ff_f/2)*(ff_x(2)-ff_x(1));
Ex=cumsum(exp(Ex-mean(Ex)-20));
Ex=Ex./Ex(end);

rng(random_seed)

Theta_tri=sqrt(2*dw)*sqrt(kT*1.0/(2.0*pi))*sqrt(abs(Theta));
xi=randn(size(Theta_tri,1),M)-1i*randn(size(Theta_tri,1),M);   
R=real(fft(Theta_tri.*xi,N*2+1));
R=R(1:N,:);

v=zeros(N,M);
x=zeros(N,M);
vh=zeros(N+Ntheta,M);

for i=1:M
    x(1,i)=ff_x(find(rand(1)<Ex,1,'first'));
end
v(1,:)=randn(1,M)*sqrt(v_square);


vh(Ntheta,:)=v(1,:)/2; % /2 for trapz

Fv=fx(x(1,:))+R(1,:);
v_tmp=v(1,:)+Fv*dt/mass/2;
x(2,:)=x(1,:)+v_tmp*dt;
v(2,:)=v_tmp+Fv*dt/mass/2;
theta0=theta(1);

tic
for n=2:N-1
    F_history=(theta'*vh(n+Ntheta-1:-1:n,:))*dt; %(1,M)
    Fv=fx(x(n,:))-F_history-1/2*theta0*v(n,:)*dt+R(n,:);   %(1,M)
    v(n,:)=v_tmp+Fv*dt/mass/2;
    vh(n+Ntheta-1,:)=v(n,:); 
    
    Fv=fx(x(n,:))-F_history-1/2*theta0*v(n,:)*dt+R(n,:);   %(1,M)
    
    v_tmp=v(n,:)+Fv*dt/mass/2;
    x(n+1,:)=x(n,:)+v_tmp*dt;    
    v(n+1,:)=v_tmp+Fv*dt/mass/2;

    if mod(n,1000*N_per_step)==0
        vv0=mean(v(1:n,:).*v(1:n,:),'all');
        disp([num2str(double(n)*dt),' ',num2str(toc),' ',num2str(vv0)])
    end
end
toc

N_drop=theta_cut/dt;
x=x(N_drop:N_per_step:end,:);
v=v(N_drop:N_per_step:end,:);

save(save_file_name,'x','v','-v7.3')
end