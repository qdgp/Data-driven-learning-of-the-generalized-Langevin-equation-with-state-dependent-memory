import numpy as np
import time
import h5py
import os
import scipy.io
import sys
import tensorflow as tf
from get_model import get_model

ND=4
Nbin=26
N_hx=8

save_file_name=f'ML_ND_{ND}_v2.mat'
if os.path.isfile(save_file_name):
    raise KeyError('file exists')

train_step=20000
PP_corr_max=300
corr_cut_off=20000
Tcut=200
a=0.02

lr=0.01
decay_steps=1000
decay_rate=0.9

kT=2.4750
v_square=0.4117
mass=kT/v_square
dt=0.01
N_theta=int(Tcut/2/dt)


FT=tf.float32
CT=tf.complex64

data = h5py.File('data/dx_0.2_w_301.mat','r')
PP_corr_xi=tf.transpose(tf.constant(data.get('PP_corr_xi')[:9950,:,:],dtype=FT),[0,2,1])
PP_corr=tf.transpose(tf.constant(data.get('PP_corr'),dtype=FT),[0,2,1])
PPv_corr=tf.transpose(tf.constant(data.get('PPv_corr')[:10000,:,:],dtype=FT),[0,2,1])
Px_vv0_Px=np.array(data.get('Px_vv0_Px'),dtype='float32')
maf_v0 = tf.constant(np.array(data.get('maf_v0')).transpose([1,0]),dtype=FT)
hx_x=np.array(data.get('Px_x'),dtype='float64').T

Px_vv0_Px_flip=[]
num=0    
for corr_num in range(2,PP_corr_max):
    tmp=Px_vv0_Px[:,num:(num+corr_num),:,:]  ###  !!!! mat file need transpose which is different from h5 file
    tmp[:,-1,:,:]=tmp[:,-1,:,:]/2
    tmp[:,0,:,:]=tmp[:,0,:,:]/2
    tmp=np.flip(tmp,axis=1)
    Px_vv0_Px_flip.append(tf.constant(tmp,dtype=FT))
    num=num+corr_num

data=scipy.io.loadmat('data/corr.mat')
corr_vv=data['corr_vv'].T
corr_vv=corr_vv[:,int((corr_vv.shape[1]-1)/2):]
corr_vv=tf.reshape(tf.constant(corr_vv[:,:corr_cut_off],dtype=FT),[1,-1,1])
corr_vv=tf.concat([tf.zeros([1,N_theta-1,1],dtype=FT),corr_vv],axis=1)
corr_maf_v=mass*data['corr_av'].T-data['corr_fv'].T
corr_maf_v=corr_maf_v[:,int((corr_maf_v.shape[1]-1)/2):]
corr_maf_v=tf.constant(corr_maf_v[:,0:corr_cut_off],dtype=FT)

print('load train data',flush=True)

model=get_model(Tcut,N_hx,hx_x,ND,a)
loss_his=np.zeros((train_step+1,5))
    
trainable_variables = list(model.weights.values())
start_time=time.time()
    
trapz_scale=tf.cast(tf.concat([1/2*tf.ones([1,1,1]),tf.ones([1,1,N_theta-1])],axis=-1),dtype=FT)

def ut_loss(theta,hx): # hx [ND,self.N_hx] theta [ND,ND,N_theta]
    hx_PP1_hx=tf.matmul( hx @ PP_corr ,hx ,transpose_b=True) # [N_corr,ND,ND]
    hx_PP2=hx @ PP_corr_xi
    hx_PP2_hx=tf.matmul( hx_PP2 ,hx_PP2 ,transpose_b=True) # [N_corr,ND,ND]
    hx_PP_hx=tf.concat([hx_PP1_hx,hx_PP2_hx],axis=0)
    hx_PP_hx=tf.transpose(hx_PP_hx,[1,2,0])

    theta_hx2=tf.reduce_sum(hx_PP_hx*theta,axis=[0,1],keepdims=True)*trapz_scale
    theta_reverse=tf.transpose(tf.reverse(tf.reshape(theta_hx2,[1,-1,1]),[1]),[1,0,2])

    ut= tf.nn.conv1d(input=corr_vv,filters=theta_reverse, stride=1, padding="VALID")
    ut=-tf.squeeze(ut,axis=2)*dt
    loss1=tf.reduce_sum(tf.square(ut-corr_maf_v))*50.0

    hx_PPv_hx=tf.matmul( hx @ PPv_corr ,hx ,transpose_b=True) # [N_corr,ND,ND]
    theta_hx_PPv_hx=tf.reduce_sum(tf.transpose(theta,[2,0,1]) * hx_PPv_hx,axis=[1,2])
    M_mean=(tf.math.cumsum(theta_hx_PPv_hx)-0.5*theta_hx_PPv_hx)*dt

    loss2=tf.reduce_sum(tf.square(M_mean))*200.0

    return loss1,loss2,ut
    
print('loss1 train begin',flush=True)

optimizer=tf.keras.optimizers.Adam(learning_rate=0.04)
for step in range(4000+1):
    with tf.GradientTape() as tape:
        Theta,theta,hx=model.get_Theta()
        theta=theta[:,:,:N_theta]
        loss1,loss2,ut=ut_loss(theta,hx)
        loss=loss1+loss2
        
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    if step % 200 == 0:
        elapsed_time = time.time() - start_time
        print("step: %5u, loss: %.4f, lr: %.4f, elapsed time %.2f" %(step,loss.numpy(),optimizer._decayed_lr(FT).numpy(), elapsed_time),flush=True)

print('loss train begin',flush=True)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate)
optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
hx_ut_v0=np.zeros([PP_corr_max,Nbin])
hx_ut_v0_loss=np.zeros([train_step+1,Nbin])
for step in range(train_step+1):    
    with tf.GradientTape() as tape:
        Theta,theta,hx=model.get_Theta()
        theta=theta[:,:,:N_theta]
        loss1,loss2,ut=ut_loss(theta,hx)
        loss3=tf.constant(0.0,dtype=FT)
        for corr_id in range(2,PP_corr_max):     
            Px_vv0_Px_tmp=Px_vv0_Px_flip[corr_id-2]
            hx_Px_vv0_Px_hx=hx @ Px_vv0_Px_tmp @ tf.transpose(hx)
            hx_Px_vv0_Px_hx=tf.transpose(hx_Px_vv0_Px_hx,[3,2,1,0])
            hx_Px_vv0_Px_hx=tf.reduce_sum(hx_Px_vv0_Px_hx*tf.expand_dims(theta[:,:,0:corr_id],axis=3),axis=[0,1,2])*dt
            
            LminusR_square=tf.square(hx_Px_vv0_Px_hx+maf_v0[:,corr_id-1])
            loss3=loss3+tf.reduce_sum(LminusR_square)
            hx_ut_v0[corr_id,:]=hx_Px_vv0_Px_hx.numpy()
            hx_ut_v0_loss[step,:]=hx_ut_v0_loss[step,:]+LminusR_square.numpy()
        
        loss=loss1+loss2+loss3
                
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    elapsed_time = time.time() - start_time 
    loss_his[step,:]=np.array([loss.numpy(),loss1.numpy(),loss2.numpy(), loss3.numpy(), elapsed_time])

    if step % 400 == 0:
        print("step: %5u, loss: %.4f, lr: %.4f, elapsed time %.2f" %(step,loss.numpy(),optimizer._decayed_lr(FT).numpy(), elapsed_time),flush=True)
        print("  loss1: %.4f, loss2: %.4f, loss3: %.4f" %(loss1.numpy(),loss2.numpy(), loss3.numpy()),flush=True)  
        record_variable={'loss':loss_his,'hx_ut_v0_loss':hx_ut_v0_loss,'hx_ut_v0':hx_ut_v0,'ut':ut.numpy()}
        model.record_save(record_variable,save_file_name)
print(save_file_name,flush=True)
