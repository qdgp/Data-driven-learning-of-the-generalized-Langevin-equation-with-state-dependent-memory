import numpy as np
import h5py
import tensorflow as tf
import scipy.io

FT=tf.float32
CT=tf.complex64
class get_model(object):
    def __init__(self,T_cut,N_hx,hx_x,ND,a):
        random_normal = tf.keras.initializers.Ones()
        self.ND=ND
        self.dt=0.01
        self.T_cut=T_cut
        self.N=int(T_cut/self.dt)
        self.N_Theta=int(self.N/10)
        self.N_pad=self.N-self.N_Theta
        self.N_hx=N_hx
        self.a=a
        self.weights={}
        self.weights['Theta_sym'] =tf.Variable(0.001*random_normal([self.N_Theta,ND,ND],dtype=FT),trainable=True,name='Theta_sym')
        self.weights['hx']=tf.Variable(random_normal([ND,self.N_hx]),trainable=True,name='hx')

        self.history={'N':self.N,'a':self.a,'ND':self.ND,'N_Theta':self.N_Theta,'T_cut':self.T_cut,'hx_x':hx_x,
         'hx': [],'Theta': [],'theta': [],'loss':[]}

    def get_Theta(self):
        Theta_tri=(tf.linalg.band_part(self.weights['Theta_sym'], -1, 0)-
                   tf.linalg.band_part(self.weights['Theta_sym'], 0, 0)+
                   tf.abs(tf.linalg.band_part(self.weights['Theta_sym'], 0, 0)))
        Theta_tri=Theta_tri+0.000001*tf.eye(self.ND, batch_shape=[self.N_Theta])
        Theta_sym=tf.matmul(Theta_tri,Theta_tri,transpose_b=True)
        theta0=tf.reduce_sum(Theta_sym,axis=0)
        theta0_chol=tf.linalg.inv(tf.linalg.cholesky(theta0))
        Theta_sym=(theta0_chol @ Theta_sym @ tf.transpose(theta0_chol) )*float(self.T_cut)/2.0

        Theta_tmp=tf.transpose(tf.dtypes.complex(Theta_sym,0.0),[1,2,0])
        Theta_tmp=tf.concat([Theta_tmp,tf.zeros([self.ND,self.ND,self.N_pad],dtype=CT)],axis=2)

        theta_scale=tf.constant(np.exp(-np.arange(self.N)*self.dt*self.a),dtype=FT)
        theta_scale=tf.reshape(theta_scale,[1,1,self.N])        
        theta=2.0*tf.math.real(tf.signal.ifft(Theta_tmp))/self.dt
        theta=theta*theta_scale

        hx=(100.0*tf.math.sigmoid(self.weights['hx'])+1)/np.sqrt(float(self.ND))
        return Theta_sym,theta,hx

    def load_model(self,name):
        history_tmp=scipy.io.loadmat(name)
        self.weights['Theta_sym']=tf.Variable(history_tmp['weights_Theta_sym'],dtype=FT,trainable=True,name='Theta_sym')
        self.weights['hx']=tf.Variable(history_tmp['weights_hx'],dtype=FT,trainable=True,name='hx')

    def record_save(self,record_variable,save_name):
        Theta,theta,hx=self.get_Theta()
        hx=hx.numpy()
        Theta=Theta.numpy()
        theta=theta.numpy()

        self.history['weights_Theta_sym']=self.weights['Theta_sym'].numpy()
        self.history['weights_hx']=self.weights['hx'].numpy()

        self.history['Theta'].append(Theta)
        self.history['theta'].append(theta)
        self.history['hx'].append(hx)
        
        self.history.update(record_variable)

        scipy.io.savemat(save_name, self.history)