import numpy as np
import time
import h5py
import os
import scipy.io
import sys
import tensorflow as tf
from get_model import get_model

ND=2
Nbin=31
N_hx=3

save_file_name=f'ML_ND_{ND}.mat'

train_step=40000
PP_corr_max=500
corr_cut_off=3000
Tcut=120
a=0.1

lr=0.001
decay_steps=2000
decay_rate=0.9

mass=0.468918023982147
dt=0.02
N_theta=corr_cut_off


FT=tf.float32
data = h5py.File('data/dx_10_w_501.mat','r')
Px_vv0_Px_overall=np.expand_dims(np.array(data.get('Px_vv0_Px_overall'),dtype='float32'),axis=0)
Px_vx0_Px_overall=np.expand_dims(np.array(data.get('Px_vx0_Px_overall'),dtype='float32'),axis=0)
Px_vv0_Px=np.array(data.get('Px_vv0_Px'),dtype='float32')
maf_v0 = tf.constant(np.array(data.get('maf_v0')).transpose([1,0]),dtype=FT)
hx_x=np.array(data.get('Px_x'),dtype='float64').T
Px_vv0_Px_overall=tf.constant(np.concatenate((Px_vv0_Px_overall[:,:corr_cut_off,:,:],Px_vx0_Px_overall[:,:corr_cut_off,:,:]),axis=0),dtype=FT)

data=scipy.io.loadmat('data/corr.mat')
corr_maf_x=-mass*data['corr_vv'].T-data['corr_fx'].T
corr_maf_x=corr_maf_x[:,int((corr_maf_x.shape[1]-1)/2):]
corr_maf_v=mass*data['corr_av'].T-data['corr_fv'].T
corr_maf_v=corr_maf_v[:,int((corr_maf_v.shape[1]-1)/2):]
corr_maf_v=tf.constant(np.concatenate((corr_maf_v[:,:corr_cut_off],corr_maf_x[:,:corr_cut_off]),axis=0),dtype=FT)

Px_vv0_Px_flip=np.zeros([Nbin,PP_corr_max-1,PP_corr_max,N_hx,N_hx],dtype='float32')
num=0
for corr_num in range(2,PP_corr_max+1):
    tmp=Px_vv0_Px[:,num:(num+corr_num),:,:]
    tmp[:,-1,:,:]=tmp[:,-1,:,:]/2
    tmp[:,0,:,:]=tmp[:,0,:,:]/2
    tmp=np.flip(tmp,axis=1)
    Px_vv0_Px_flip[:,corr_num-2,0:corr_num,:,:]=tmp
    num=num+corr_num
Px_vv0_Px_flip=tf.constant(Px_vv0_Px_flip,dtype=FT)



print('load data',flush=True)

model=get_model(Tcut,N_hx,hx_x,ND,a)
loss_his=np.zeros((train_step+1,4))
    
trainable_variables = list(model.weights.values())
start_time=time.time()

trapz_scale=tf.cast(tf.concat([1/2*tf.ones([1,1,1]),tf.ones([1,1,N_theta-1])],axis=-1),dtype=FT)
def overall_loss(theta,hx): # hx [ND,N_hx] theta [ND,ND,N_theta]
    theta_reverse=tf.transpose(tf.reverse(tf.reshape(theta*trapz_scale,[ND*ND,N_theta,1]),[1]),[1,0,2])

    hx_vv0_hx=tf.matmul( hx @ Px_vv0_Px_overall ,hx ,transpose_b=True) # [2,corr_cut_off,ND,ND]
    hx_vv0_hx=tf.reshape(hx_vv0_hx,[2,corr_cut_off,ND*ND])
    hx_vv0_hx_pad=tf.concat([tf.zeros([2,N_theta-1,ND*ND],dtype=FT),hx_vv0_hx],axis=1)
    ut= tf.nn.conv1d(input=hx_vv0_hx_pad,filters=theta_reverse, stride=1, padding="VALID")
    ut0= hx_vv0_hx[:,0,:] @ tf.reshape(theta,[ND*ND,N_theta])
    ut= tf.squeeze(ut)*dt-tf.squeeze(ut0)*0.5*dt
    loss=tf.reduce_sum(tf.square(ut+corr_maf_v))
    return loss,ut
    
print('loss1 train begin',flush=True)

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
for step in range(2000+1):
    with tf.GradientTape() as tape:
        Theta,theta,hx=model.get_Theta()
        theta=theta[:,:,:N_theta]
        loss,ut=overall_loss(theta,hx)
        
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    if step % 200 == 0:
        elapsed_time = time.time() - start_time
        print("step: %5u, loss: %.4f, lr: %.4f, elapsed time %.2f" %(step,loss.numpy(),optimizer._decayed_lr(FT).numpy(), elapsed_time),flush=True)

print('loss train begin',flush=True)


@tf.function()
def train(): # hx [ND,N_hx] theta [ND,ND,N_theta]
    with tf.GradientTape() as tape:
        Theta,theta,hx=model.get_Theta()
        theta=theta[:,:,:N_theta]
        loss1,ut=overall_loss(theta,hx)
        loss3=tf.constant(0.0,dtype=FT)

        hx_Px_vv0_Px_hx=hx @ Px_vv0_Px_flip @ tf.transpose(hx) # [Nbin,PP_corr_max-1,PP_corr_max,ND,ND]
        hx_Px_vv0_Px_hx=tf.transpose(hx_Px_vv0_Px_hx,[4,3,2,1,0])
        hx_Px_vv0_Px_hx=tf.reduce_sum(hx_Px_vv0_Px_hx*tf.reshape(theta[:,:,0:PP_corr_max],[ND,ND,PP_corr_max,1,1]),axis=[0,1,2])*dt
        hx_Px_vv0_Px_hx=tf.transpose(hx_Px_vv0_Px_hx) #  [Nbin,PP_corr_max-1]
        LminusR_square=tf.square(hx_Px_vv0_Px_hx+maf_v0[:,1:PP_corr_max])
        loss3=loss3+tf.reduce_sum(LminusR_square)
        loss=loss1*0.9+loss3*0.1
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss,loss1,loss3,hx_Px_vv0_Px_hx,LminusR_square,ut


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate)
optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
hx_ut_v0=np.zeros([PP_corr_max,Nbin])
hx_ut_v0_loss=np.zeros([train_step+1,Nbin])
for step in range(train_step+1):
    loss,loss1,loss3,hx_Px_vv0_Px_hx,LminusR_square,ut=train()
    hx_ut_v0=hx_Px_vv0_Px_hx.numpy().transpose()
    hx_ut_v0_loss[step,:]=tf.reduce_sum(LminusR_square,axis=1).numpy()
    elapsed_time = time.time() - start_time 
    loss_his[step,:]=np.array([loss.numpy(),loss1.numpy(),loss3.numpy(), elapsed_time])

    if step % 200 == 0:
        print("step: %5u, loss: %.4f, lr: %.4f, elapsed time %.2f" %(step,loss.numpy(),optimizer._decayed_lr(FT).numpy(), elapsed_time),flush=True)
        print("  loss1: %.4f, loss3: %.4f" %(loss1.numpy(),loss3.numpy()),flush=True)  
        record_variable={'loss':loss_his,'hx_ut_v0_loss':hx_ut_v0_loss,'hx_ut_v0':hx_ut_v0,'ut':ut.numpy()}
        model.record_save(record_variable,save_file_name)
print(save_file_name,flush=True)
