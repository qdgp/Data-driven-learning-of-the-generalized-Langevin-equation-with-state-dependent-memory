function step3_training_set(folder,i)
load('data/PDF.mat','ff_x','ff_f','kT','dt','mass','x_mean')
dx=0.2;
Px_x=(2.8:dx:4.2)';
Px_Nbin=length(Px_x);
corr_x=(2.8:0.05:4.1)';
Corr_Nbin=length(corr_x);
rng(i)

w_max=300+1;
PP_corr_max=10000+1;

tic
load(['MD_data/xv_',num2str(i),'.mat'],'x','v');
maf_tmp = mass*derivative_1st(dt, v)-...
    kT*interp1(ff_x,ff_f, x, 'linear','extrap');
Nlist=length(x);

x_bin=floor((x-corr_x(1))/(corr_x(2)-corr_x(1)))+1;
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
toc

PP_corr=zeros(Px_Nbin,Px_Nbin,PP_corr_max+1);
PPv_corr=zeros(Px_Nbin,Px_Nbin,PP_corr_max+1);
N = 2^ceil(log2(Nlist));    
scaleUnbiased = (Nlist - (0:PP_corr_max)).';
for bin_id_i=1:Px_Nbin
    Fxi_fft=fft(full(Fx(bin_id_i,:)'),N,1);
    Fxvi_fft=fft(full(Fxv(bin_id_i,:)'),N,1);
    corr = ifft(Fxi_fft.*conj(Fxi_fft),[],1,'symmetric');
    PP_corr(bin_id_i,bin_id_i,:) = corr(1:PP_corr_max+1)./scaleUnbiased; 
    corr = ifft(Fxvi_fft.*conj(Fxi_fft),[],1,'symmetric');
    PPv_corr(bin_id_i,bin_id_i,:) = corr(1:PP_corr_max+1)./scaleUnbiased; 
    for bin_id_j=bin_id_i+1:Px_Nbin
        Fxj_fft=fft(full(Fx(bin_id_j,:)'),N,1);
        corr = ifft(Fxi_fft.*conj(Fxj_fft),[],1,'symmetric'); 
        PP_corr(bin_id_i,bin_id_j,:) = corr(1:PP_corr_max+1)./scaleUnbiased; 
        PP_corr(bin_id_j,bin_id_i,:) = corr([1,N-PP_corr_max+(PP_corr_max:-1:1)])./scaleUnbiased;
        corr = ifft(Fxvi_fft.*conj(Fxj_fft),[],1,'symmetric'); 
        PPv_corr(bin_id_i,bin_id_j,:) = corr(1:PP_corr_max+1)./scaleUnbiased; 
        PPv_corr(bin_id_j,bin_id_i,:) = corr([1,N-PP_corr_max+(PP_corr_max:-1:1)])./scaleUnbiased;             
    end
end
disp([num2str(i),' ',num2str(toc)])

save([folder,'dx_',num2str(dx),'_w_',num2str(w_max),'_',num2str(i),'.mat'],...
      'Px_vv0_Px','PP_corr','PPv_corr','maf_v0','maf_v0_count','Px_x','-v7.3')
end


