function step0_collect_data(M)
Natom=16;
Nloop=floor(80000001/8);

parfor i=1:M
    fid = fopen(['MD_data/traj',num2str(i),'.lammpstrj'], 'r');
    x_tmp=zeros(Nloop,3);
    v_tmp=zeros(Nloop,3);
    for j=1:Nloop
        for l = 1:9; fgets(fid); end   
        data = fscanf(fid, '%d %f %f %f %f %f %f',[7 Natom])';
        fgets(fid);
        data(data(:,1),:)=data;
        x_tmp(j,:)=data(1,2:4)-data(end,2:4);
        v_tmp(j,:)=data(1,5:7)-data(end,5:7);
    end
    fclose(fid);
    x=sqrt(sum(x_tmp.^2,2));
    v=sum(x_tmp.*v_tmp,2)./x;
    save_data(x,v,i)
    disp([i,min(x,[],'all'),max(x,[],'all')])
end
end

function save_data(x,v,i)
    save(['MD_data/xv_',num2str(i),'.mat'],'x','v','-v7.3')
end