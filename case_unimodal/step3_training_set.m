function step3_training_set(folder,i)
load('data/PDF.mat','ff_x','ff_f','kT','dt','mass','hx_x','x_mean')
dx=10;
Px_x=(0:dx:20)';
Px_Nbin=length(Px_x);
Corr_Nbin=length(hx_x);
rng(i)

w_max=500+1;
w_overall=10000+1;

tic
load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
maf_tmp = mass*derivative_1st(dt, v)-...
    kT*interp1(ff_x,ff_f, x, 'linear','extrap');
Nlist=length(x);

x_bin=floor((x-hx_x(1))/(hx_x(2)-hx_x(1)))+1;
x_bin=x_bin(1:Nlist-w_max-1);
x_bin(x_bin<1)=1;
x_bin(x_bin>Corr_Nbin-1)=Corr_Nbin-1;

x_index=floor((x-Px_x(1))/dx)+1;
x_index(x_index<1)=1;
x_index(x_index>Px_Nbin-1)=Px_Nbin-1;
x_remain=x-(x_index-1)*dx-Px_x(1);

Fx1=sparse(x_index,1:length(x),dx-x_remain,Px_Nbin,length(x));
Fx2=sparse(x_index+1,1:length(x),x_remain,Px_Nbin,length(x));
Fx=full(Fx1+Fx2)/dx;
Fxv=Fx.*v';

toc

maf_v0=zeros(Corr_Nbin-1,w_max+1);
maf_v0_count=zeros(Corr_Nbin-1,1);
for bin_id=1:Corr_Nbin-1
    maf_v0_count(bin_id)=sum(x_bin==bin_id);
end
v_tmp=sparse(1:Nlist-w_max-1,x_bin,v(1:Nlist-w_max-1),Nlist-w_max-1,Corr_Nbin-1);   
for n=1:w_max+1
    maf_v0(:,n)=sum(maf_tmp(n:end-w_max-1+n-1).*v_tmp,1)';
end
toc

Px_vv0_Px=zeros(Px_Nbin,Px_Nbin,(2+w_max)*(w_max-1)/2,Corr_Nbin-1);
for bin_id=1:Corr_Nbin-1
    dict_tmp = reshape(find(x_bin==bin_id),[],1);
    if length(dict_tmp)>4*10^4
        dict_tmp=dict_tmp(sort(randperm(length(dict_tmp),4*10^4)));
    end
    Nsample=length(dict_tmp);

    num=0;
    for w1=2:w_max
        dict_k=(dict_tmp+(0:w1-1))';
        Fxv_k=reshape(Fxv(:,dict_k(:)),Px_Nbin*w1,Nsample).*v(dict_tmp)';
        Fx_i=Fx(:,dict_tmp+w1-1)';
        Px_vv0_Px_tmp=Fxv_k*Fx_i/Nsample;
        Px_vv0_Px_tmp=permute(reshape(Px_vv0_Px_tmp,Px_Nbin,w1,Px_Nbin),[1,3,2]);
        Px_vv0_Px(:,:,num+1:num+w1,bin_id)=Px_vv0_Px_tmp;
        num=num+w1;
    end
end

Px_vv0_Px_overall=zeros(Px_Nbin,Px_Nbin,w_overall);
Px_vx0_Px_overall=zeros(Px_Nbin,Px_Nbin,w_overall);
Fxv_i=Fx(:,w_overall:10:end).*v(1:10:end-w_overall+1)';
Fxx_i=Fx(:,w_overall:10:end).*(x(1:10:end-w_overall+1)-x_mean)';
Nsample=size(Fxv_i,2);
for w1=1:w_overall
    Px_vv0_Px_overall(:,:,w1)=Fxv(:,w1:10:end-w_overall+w1)*Fxv_i'/Nsample;
    Px_vx0_Px_overall(:,:,w1)=Fxv(:,w1:10:end-w_overall+w1)*Fxx_i'/Nsample;
end

toc

save([folder,'dx_',num2str(dx),'_w_',num2str(w_max),'_',num2str(i),'.mat'],...
      'Px_vv0_Px','maf_v0','maf_v0_count','Px_vv0_Px_overall','Px_vx0_Px_overall','Px_x','-v7.3')
end


