function step5_hx_GLE(T_scale,N_per_step,M,ML_file,save_file_name,random_seed)
theta_cut=80;
load('data/PDF.mat','mass','ff_x','ff_f','kT','v_square','dt');
fx=@(x) kT*interp1(ff_x,ff_f, x, 'linear','extrap');
dt=dt/N_per_step;
Ntheta=int64(theta_cut/dt);

ML=load(ML_file,'T_cut','hx','hx_x','Theta','a','ND');
a=double(ML.a);
ND=double(ML.ND);
T_cut=double(ML.T_cut);
step=size(ML.Theta,1);

T=T_cut*T_scale;
N=T/dt;
disp(T)

Theta=double(squeeze(ML.Theta(step,:,:,:)));
[R,Theta]=generate_noise(Theta,T_cut,dt,T_scale,kT,a,M,random_seed);

theta=2.0*real(ifft(Theta,N*2))/dt;
theta=permute(reshape(theta(1:Ntheta,:),Ntheta,ND,ND),[2,3,1]);

ML_hx=squeeze(ML.hx(step,:,:))';
ML_hx=double([ML_hx(1,:);ML_hx;ML_hx(end,:)]);
ML_hx_x=[-100;ML.hx_x;100];
hx=@(x) interp1(ML_hx_x,ML_hx, x, 'linear','extrap');   % (1,3)

Ex=(cumsum(ff_f)-ff_f(1)/2-ff_f/2)*(ff_x(2)-ff_x(1));
Ex=cumsum(exp(Ex-mean(Ex)-20));
Ex=Ex./Ex(end);

rng(random_seed)
v=zeros(N,M);
x=zeros(N,M);
vh=zeros(M,ND,N+Ntheta);

for i=1:M
    x(1,i)=ff_x(find(rand(1)<Ex,1,'first'));
end
v(1,:)=randn(1,M)*sqrt(v_square);

disp('begin')
tic

hx_x=hx(x(1,:));
vh(:,:,Ntheta)=v(1,:)'.*hx_x/2; % /2 for trapz

R_tmp=squeeze(R(1,:,:));
Fv=fx(x(1,:))+sum(hx_x'.*R_tmp,1);
v_tmp=v(1,:)+Fv*dt./mass/2;
x(2,:)=x(1,:)+v_tmp*dt;
v(2,:)=v_tmp+Fv*dt./mass/2;
theta0=theta(:,:,1);

for n=2:N-1
    R_tmp=squeeze(R(n,:,:)); % (ND,M)
    hx_x=hx(x(n,:)); %generate_noise hx(x(n,:)) (M,ND)  fx(x(n,:)) (1,M)
    hx_theta=pagemtimes(hx_x,theta);  % (M,ND,Ntheta)
    F_history=sum(hx_theta.*vh(:,:,n+Ntheta-1:-1:n),[2,3])*dt; %(M,1)
    Fv=fx(x(n,:))-F_history'-1/2*sum((hx_x*theta0)'.*v(n,:).*hx_x',1)*dt+sum(hx_x'.*R_tmp,1);   %(1,M)
    v(n,:)=v_tmp+Fv*dt./mass/2;
    vh(:,:,n+Ntheta-1)=v(n,:)'.*hx_x;
    
    Fv=fx(x(n,:))-F_history'-1/2*sum((hx_x*theta0)'.*v(n,:).*hx_x',1)*dt+sum(hx_x'.*R_tmp,1);   %(1,M)
    
    v_tmp=v(n,:)+Fv*dt./mass/2;
    x(n+1,:)=x(n,:)+v_tmp*dt;    
    v(n+1,:)=v_tmp+Fv*dt./mass/2;

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




function [R,Theta]=generate_noise(Theta,T_cut,dt,T_scale,kT,a,M,random_seed)
T=T_cut*T_scale;
N=T/dt;
dw=2*pi/T/2;

ND=size(Theta,2);
Theta=reshape(Theta,[],ND*ND);

theta=2.0*real(ifft(real(Theta),int64(T_cut/dt)))/dt;
theta=repmat(theta,[T_scale,1]);
theta=theta.*exp(-(0:size(theta,1)-1)'*dt*a);
Theta=real(fft([theta;theta(end:-1:2,:)]))*dt/2;

NTheta=floor(size(Theta,1)/2)-1;
Theta(2:2+NTheta,:)=Theta(2:2+NTheta,:)+Theta(end:-1:end-NTheta,:);
Theta=Theta(1:end-NTheta-1,:);

Theta_tri=zeros(size(Theta));
for i=1:size(Theta,1)
    Theta_tmp=real(reshape(Theta(i,:),ND,ND));
    Theta_tri(i,:)=reshape(chol(Theta_tmp)',1,ND*ND);    
end
Theta_tri=sqrt(2*dw)*sqrt(1.0/(2.0*pi))*Theta_tri;
NTheta=size(Theta_tri,1);
Theta_tri=reshape(sqrt(kT)*Theta_tri,[NTheta,ND,ND]);


rng(random_seed)
R=zeros(N,ND,M);

for i=1:M
    xi=randn(NTheta,ND)+1i*randn(NTheta,ND);
    noise=zeros(N*2,ND);
    for j=1:ND
        Theta_xi=sum(squeeze(Theta_tri(:,j,:)).*xi,2);
        noise(:,j)=real(fft(Theta_xi,N*2));
    end
    R(:,:,i)=noise(1:N,:);
end
end
