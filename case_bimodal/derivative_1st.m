function [ df ] = derivative_1st(delx, f)

%delx=x(2)-x(1);
%f=sin(x);
df=zeros(length(f),1);
%Central difference
df(3:end-2)=(-2/3*f(2:end-3)+2/3*f(4:end-1)+1/12*f(1:end-4)-1/12*f(5:end))/delx;

df(1)=(-25*f(1)+48*f(2)-36*f(3)+16*f(4)-3*f(5))/12/delx;
df(2)=(-3*f(1)-10*f(2)+18*f(3)-6*f(4)+f(5))/12/delx;

df(end)=(3*f(end-4)-16*f(end-3)+36*f(end-2)-48*f(end-1)+25*f(end))/12/delx;
df(end-1)=(-1*f(end-4)+6*f(end-3)-18*f(end-2)+10*f(end-1)+3*f(end))/12/delx;

end

