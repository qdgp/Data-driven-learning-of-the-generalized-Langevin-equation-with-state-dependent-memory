function [corr,xcorr,xcorr_count,tail_dist] = compute_correlation_function(x,v,x_mean,xcorr_x,N_corr,N_xcorr)
load('data/PDF.mat','ff_x','ff_f','kT','dt')
right_wall=15;
x_tmp=double(x>=right_wall);
dx=x_tmp(2:end)-x_tmp(1:end-1);
index=find(abs(dx)==1);
if x_tmp(1)==1
    index=index(2:end);
end
tail_dist=(index(2:2:end)-index(1:2:end-mod(length(index),2)))*dt;

xcorr_dx=xcorr_x(2)-xcorr_x(1);
xcorr_x0=xcorr_x(1);
xcorr_N=length(xcorr_x);

a=derivative_1st(dt, v);
f=kT*interp1(ff_x,ff_f, x, 'linear','extrap');
        
xindex=floor((x-xcorr_x0)/xcorr_dx)+1;
xindex=xindex(1:end-N_xcorr*2);

x=x-x_mean;

N = 2^ceil(log2(length(x)));
x_fft = fft(x,N,1);    v_fft = fft(v,N,1);
a_fft = fft(a,N,1);    f_fft = fft(f,N,1);

corr=cell(1,8);
scaleUnbiased = (length(x) - abs(-N_corr:N_corr)).';

corr{1}=xcorr_tool(x_fft,x_fft,N,N_corr,scaleUnbiased);
corr{2}=xcorr_tool(v_fft,v_fft,N,N_corr,scaleUnbiased);
corr{3}=xcorr_tool(a_fft,a_fft,N,N_corr,scaleUnbiased);
corr{4}=xcorr_tool(v_fft,x_fft,N,N_corr,scaleUnbiased);
corr{5}=xcorr_tool(f_fft,x_fft,N,N_corr,scaleUnbiased);
corr{6}=xcorr_tool(f_fft,v_fft,N,N_corr,scaleUnbiased);
corr{7}=xcorr_tool(a_fft,v_fft,N,N_corr,scaleUnbiased);
corr{8}=xcorr_tool(f_fft,a_fft,N,N_corr,scaleUnbiased);

xcorr=zeros(xcorr_N,N_xcorr,4);
xcorr_count=zeros(xcorr_N,1);
for k=1:xcorr_N
    index = reshape(find(xindex==k),[],1);
    index = index(1:10:end);
    index2=(index+(0:N_xcorr-1))';

    xcorr(k,:,1)=(reshape(v(index2(:)),N_xcorr,[])*v(index))';
    xcorr(k,:,2)=(reshape(v(index2(:)),N_xcorr,[])*x(index))';
    xcorr(k,:,3)=(reshape(f(index2(:)),N_xcorr,[])*x(index))';
    xcorr(k,:,4)=(reshape(a(index2(:))-f(index2(:)),N_xcorr,[])*x(index))';
    xcorr(k,:,5)=(reshape(x(index2(:)),N_xcorr,[])*x(index))';
    xcorr_count(k)=length(index);
end      

corr=cell2mat(corr);
end

function corr=xcorr_tool(x_fft,y_fft,N,N_corr,scaleUnbiased)
corr = ifft(x_fft.*conj(y_fft),[],1,'symmetric');
corr = corr([N-N_corr+(1:N_corr),1:N_corr+1])./scaleUnbiased; 
end

