function step5_hx_GLE_fast_conv(T_scale,N_per_step,M,ML_file,save_file_name,random_seed)
load('../data/PDF.mat','mass','ff_x','ff_f','kT','v_square','dt');
fx=@(x) kT*interp1(ff_x,ff_f, x, 'linear','extrap');
dt=dt/N_per_step;

ML=load(ML_file,'T_cut','hx','hx_x','Theta','a','ND');
a=double(ML.a);
ND=double(ML.ND);
T_cut=double(ML.T_cut);
step=size(ML.Theta,1);

T=T_cut*T_scale;
N=T/dt;
disp(T)

Theta=double(squeeze(ML.Theta(step,:,:,:)));
[R,~]=generate_noise(Theta,T_cut,dt,T_scale,kT,a,M,random_seed);

ML_hx=squeeze(ML.hx(step,:,:))';
ML_hx=double([ML_hx(1,:);ML_hx;ML_hx(end,:)]);
ML_hx_x=[-100;ML.hx_x;100];
hx=@(x) interp1(ML_hx_x,ML_hx, x, 'linear','extrap');   % (1,3)

B=2;K=1024; a1=8; a2=0.6*256; dw=2*pi/T_cut;
[FC,~,~]=FastConv_init(Theta,ND,dw,dt,B,N,K,a1,a2,a,M);

disp('begin')
[x,v] = generate_xv_init(ff_x,ff_f,v_square,N,M,random_seed);

tic
% n=1
hx_x=hx(x(1,:));% hx(x(n,:)) (M,ND) 
hxv1=hx_x'.*v(1,:);

R_tmp=squeeze(R(1,:,:));
Fv=fx(x(1,:))+sum(hx_x'.*R_tmp,1);
v_tmp=v(1,:)+Fv*dt/mass/2;
x(2,:)=x(1,:)+v_tmp*dt;
v(2,:)=v(1,:)+Fv*dt/mass;

% n=2
R_tmp=squeeze(R(2,:,:)); % (ND,M)
hx_x=hx(x(2,:));% hx(x(n,:)) (M,ND) 
hxv2=hxv1; hxv1=hx_x'.*v(2,:); % (ND,M) 

u_tmp=FC.phi1*hxv2+FC.phi2*(hxv1-hxv2)/dt; % phi1 (ND,ND) u_tmp (ND,M)
Fv=fx(x(2,:))-sum(hx_x'.*real(u_tmp),1)+sum(hx_x'.*R_tmp,1);   %(1,M)
v_tmp=v(2,:)+Fv*dt/mass/2;
x(3,:)=x(2,:)+v_tmp*dt;
v(3,:)=v(2,:)+Fv*dt/mass;

for n=3:N-1
    R_tmp=squeeze(R(n,:,:)); % (ND,M)
    hx_x=hx(x(n,:)); % hx(x(n,:)) (M,ND)  fx(x(n,:)) (1,M)
    [FC,u_tmp] = FastConv_update(FC,hxv1,hxv2,n,B,dt);

    hxv2=hxv1; hxv1=hx_x'.*v(n,:); % (ND,M) 
    u=u_tmp+FC.phi1*hxv2+FC.phi2*(hxv1-hxv2)/dt+sum(pagemtimes(FC.EFw{1},FC.YT{1}),3);    
    Fv=fx(x(n,:))-sum(hx_x'.*real(u),1)+sum(hx_x'.*R_tmp,1);   %(1,M)
    
    v(n,:)=v_tmp+Fv*dt/mass/2;
    hxv1=hx_x'.*v(n,:);

    u=u_tmp+FC.phi1*hxv2+FC.phi2*(hxv1-hxv2)/dt+sum(pagemtimes(FC.EFw{1},FC.YT{1}),3);    
    Fv=fx(x(n,:))-sum(hx_x'.*real(u),1)+sum(hx_x'.*R_tmp,1);   %(1,M)
    v_tmp=v(n,:)+Fv*dt/mass/2;
    
    x(n+1,:)=x(n,:)+v_tmp*dt;    
    v(n+1,:)=v(n,:)+Fv*dt/mass;
%     toc
    if mod(n,100*N_per_step)==0
        vv0=mean(v(1:n,:).*v(1:n,:),'all');
        disp([num2str(double(n)*dt),' ',num2str(toc),' ',num2str(vv0)])
    end
end
toc

N_drop=100/dt;
x=x(N_drop:N_per_step:end,:);
v=v(N_drop:N_per_step:end,:);
save(['/mnt/gs21/scratch/gepei/GLE_state_dependent/GLE_xv_data/',save_file_name],'x','v','-v7.3')
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

function [x,v] = generate_xv_init(fdata_x,fdata_f,v_square,N,M,random_seed)
Ex=(cumsum(fdata_f)-fdata_f(1)/2-fdata_f/2)*(fdata_x(2)-fdata_x(1));
Ex=cumsum(exp(Ex-mean(Ex)-20));
Ex=Ex./Ex(end);

rng(random_seed)
v=zeros(N,M);
x=zeros(N,M);

for i=1:M
    x(1,i)=fdata_x(find(rand(1)<Ex,1,'first'));
end
v(1,:)=randn(1,M)*sqrt(v_square);
end




function [FC,Lmax,K1]=FastConv_init(Theta,ND,dw,dt,B,N,K,a1,a2,a,M)
Lmax=ceil(log((N+1)/2)/log(B));
if N<2*B^Lmax
    Lmax=Lmax+1;
end

FC.lambda=cell(1,Lmax); 
FC.ELambda=cell(1,Lmax);
FC.ELambdaJ=cell(1,Lmax);
FC.EFw=cell(1,Lmax);

for l=1:Lmax
    [FC.lambda{l},Ew]=talbotcontour(K,l,dt,B,a1,a2);
    FC.ELambda{l}=exp(FC.lambda{l}*dt);
    FC.ELambdaJ{l}=exp(FC.lambda{l}*dt*B^(l-1));
    F_tmp=0;
    s=FC.lambda{l}+a;
    for j=1:size(Theta,1)
        F_tmp=F_tmp+reshape(Theta(j,:),ND,ND).*s./(s.^2+((j-1)*dw)^2)*0.01;
    end
    FC.EFw{l}=F_tmp.*Ew;
end
K1=size(FC.lambda{1},3);
FC.phi1=sum(FC.EFw{1}./FC.lambda{1},3);
FC.phi2=sum(FC.EFw{1}./FC.lambda{1}.^2,3); 
FC.L=1;
FC.B_power_L=B.^(0:Lmax);
FC.Lmax=Lmax;
FC.K=K1;

FC.Y=cell(1,Lmax); FC.YA=cell(1,Lmax);FC.YM=cell(1,Lmax);FC.YT=cell(1,Lmax);
for l=1:Lmax
    FC.Y{l}=zeros(ND,M,K1);    FC.YA{l}=zeros(ND,M,K1);
    FC.YM{l}=zeros(ND,M,K1);    FC.YT{l}=zeros(ND,M,K1);
end

end

function [lambda,Ew]=talbotcontour(K,l,dt,B,mu0,nu0)
% mu0=8;
% nu0=0.6;
sigma=0;
nu=nu0;
theta=(2*(-K:1:K-1)+1)*pi/(2*K);
theta=reshape(theta,1,1,[]);
mu=mu0/(dt*(2*B^l-1));
% mu=mu0/(dt*(B^l-1));
lambda=sigma+mu*(theta.*cot(theta)+1i*nu*theta);
w=mu*(cot(theta)-theta./(sin(theta)).^2+1i*nu)./(2*1i*K);
Ew=exp(lambda*dt*B^(l-1)).*w;
% EFW=exp(lambda*dt*B^(l-1)).*F(lambda).*w;
end

function [lambda,Ew]=hyperbolacontour(K,l,dt,B,tau,miu)

% tau=0.08;
% alpha=1;
% miu=3.6/(2*B^l*dt-dt);

alpha=1;
miu=miu/(2*B^l*dt-dt);

theta=(-K:K)*tau;
theta=reshape(theta,1,1,[]);
lambda=miu*(1-sin(alpha+1i*theta));
w=miu*tau/(2*pi)*cos(alpha+1i*theta);
Ew=exp(lambda*dt*B^(l-1)).*w;
end

function [FC,u_tmp] = FastConv_update(FC,hxv1,hxv2,n,B,dt)
dg=hxv1-hxv2;
    for l=1:FC.Lmax
        Phi=(FC.ELambda{l}-1)./FC.lambda{l};
        FC.Y{l}=FC.Y{l}+Phi.*(FC.lambda{l}.*FC.Y{l}+hxv2+dg./dt./FC.lambda{l})-dg./FC.lambda{l};
    end
    
%%%%%%%%%%%%%%  update fake Y
    if 2*B^FC.L==n
        FC.L=FC.L+1;
    end

    for l=1:FC.Lmax
        if mod(n-1,FC.B_power_L(l))==0
            if mod(n-1,FC.B_power_L(l+1))==0
                FC.YA{l}=FC.Y{l};% YM is the Y that currently be used   
                FC.YM{l}=FC.Y{l};
                FC.Y{l}=FC.Y{l}*0;
            else      
                FC.YM{l}=FC.Y{l};
                break
            end
        end  
    end

%%%%%%%%%%%%%%   update true Y
    for l=1:FC.L
        if mod(n,FC.B_power_L(l))==0
            FC.YT{l}=FC.YM{l};  
            if FC.B_power_L(l) < mod(n,FC.B_power_L(l+1))
                % Y renewed to zero when mod(n-1,B^l)==0, but actually for l=1, 4,5,6,7,8 are used.  
                % So, when useing Y(6dt,0,lambda(l)), it's divided into
                % Y(5dt,0,lambda(l)) which is YA, and Y(6dt,5,lambda(l)) which is YM
                FC.YA{l}=FC.ELambdaJ{l}.*FC.YA{l};
                FC.YT{l}=FC.YT{l}+FC.YA{l};
            end
        end
    end
%%%%%%%%%%%%%% end
    u_tmp=0;
    for l=2:FC.L
        u_tmp=u_tmp+sum(pagemtimes(FC.EFw{l},FC.YT{l}),3);
        FC.YT{l}=FC.ELambda{l}.*FC.YT{l};
    end

end

