"""
Modified by Henry Schneiderman
- Added code to properly reverse output from intermediate RNN
- Added functionality to work with Dataset allowing a list of input
files

Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE+RRTMGP (entire radiation scheme)

This program takes existing input-output data generated with RTE+RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

Contributions welcome!

@author: Peter Ukkonen
"""
from tensorflow.keras.layers import Dense, TimeDistributed
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses, optimizers, layers, Input, Model
import tensorflow as tf
import os
import gc
import numpy as np
import time

import matplotlib.pyplot as plt

# import ml_loaddata_rnn
import data_generation


def calc_heatingrate(F, p):
    dF = F[:, 1:] - F[:, 0:-1]
    dp = p[:, 1:] - p[:, 0:-1]
    dFdp = dF/dp
    g = 9.81  # m s-2
    cp = 1004  # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp)  # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day


def rmse(predictions, targets, ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))


def mse(predictions, targets, ax=0):
    return ((predictions - targets) ** 2).mean(axis=ax)


def mae(predictions, targets, ax=0):
    diff = predictions - targets
    return np.mean(np.abs(diff), axis=ax)


def mape(a, b):
    mask = a != 0
    return 100*(np.fabs(a[mask] - b[mask])/a[mask]).mean()


def plot_flux_and_hr_error(rsu_true, rsd_true, rsu_pred, rsd_pred, pres):
    pres_lay = 0.5 * (pres[:, 1:] + pres[:, 0:-1])

    dTdt_true = calc_heatingrate(rsd_true - rsu_true, pres)
    dTdt_pred = calc_heatingrate(rsd_pred - rsu_pred, pres)

    bias_tot = np.mean(dTdt_pred.flatten()-dTdt_true.flatten())
    rmse_tot = rmse(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_tot = mae(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_percent = 100 * np.abs(mae_tot / dTdt_true.mean())
    # mae_percent = mape(dTdt_true.flatten(), dTdt_pred.flatten())
    r2 = np.corrcoef(dTdt_pred.flatten(), dTdt_true.flatten())[0, 1]
    r2 = r2**2
    str_hre = 'Heating rate error \nR$^2$: {:0.4f} \nBias: {:0.3f} \nRMSE: {:0.3f} \nMAE: {:0.3f} ({:0.1f}%)'.format(r2,
                                                                                                                     bias_tot, rmse_tot, mae_tot, mae_percent)
    mae_rsu = mae(rsu_true.flatten(), rsu_pred.flatten())
    mae_rsd = mae(rsd_true.flatten(), rsd_pred.flatten())
    mae_rsu_p = 100 * np.abs(mae_rsu / rsu_true.mean())
    mae_rsd_p = 100 * np.abs(mae_rsd / rsd_true.mean())

    str_rsu = 'Upwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(
        mae_rsu, mae_rsu_p)
    str_rsd = 'Downwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(
        mae_rsd, mae_rsd_p)
    errfunc = mae
    # errfunc = rmse
    ind_p = 0
    hr_err = errfunc(dTdt_true[:, ind_p:], dTdt_pred[:, ind_p:])
    fluxup_err = errfunc(rsu_true[:, ind_p:], rsu_pred[:, ind_p:])
    fluxdn_err = errfunc(rsd_true[:, ind_p:], rsd_pred[:, ind_p:])
    fluxnet_err = errfunc((rsd_true[:, ind_p:] - rsu_true[:, ind_p:]),
                          (rsd_pred[:, ind_p:] - rsu_pred[:, ind_p:]))
    yy = 0.01*pres[:, :].mean(axis=0)
    yy2 = 0.01*pres_lay[:, :].mean(axis=0)

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True)
    ax0.plot(hr_err,  yy2[ind_p:], label=str_hre)
    ax0.invert_yaxis()
    ax0.set_ylabel('Pressure (hPa)', fontsize=15)
    ax0.set_xlabel('Heating rate (K h$^{-1}$)', fontsize=15)
    ax1.set_xlabel('Flux (W m$^{-2}$)', fontsize=15)
    ax1.plot(fluxup_err,  yy[ind_p:], label=str_rsu)
    ax1.plot(fluxdn_err,  yy[ind_p:], label=str_rsd)
    ax1.plot(fluxnet_err,  yy[ind_p:], label='Net flux error')
    ax0.legend()
    ax1.legend()
    ax0.grid()
    ax1.grid()

# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------


datadir = "/data-T1/hws/models/"
fpath = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"

# ----------- config ------------

scale_inputs = True

# Model training and evaluation: use GPU or CPU?
use_gpu = True


# model_name = 'Ukkonen/MODEL.RNN_1_ecRad_Dataset.'
# model_name = 'Ukkonen/MODEL.RNN_2_ecRad_Dataset.'
training_model_name = 'models/rnn.v2.'
evaluation_model_name = 'models/rnn.'
is_train = True #False  # True

#weight_prof_1 = data_generation.get_weight_profile(fpath)
#print(f'shape of weight profile {weight_prof_1.shape}')
#print(f'weight_profile = {weight_prof_1}')

if True:
    weight_prof = np.array([[ 1.0,         2.5419967],
                        [ 1.0001812  , 2.5423949],
                        [ 1.0005001  , 2.5427516],
                        [ 1.0009831  , 2.5431285],
                        [ 1.0017012  , 2.5435448],
                        [ 1.0027397  , 2.54398  ],
                        [ 1.004103   , 2.5444002],
                        [ 1.005808   , 2.5447874],
                        [ 1.0074853  , 2.5451324],
                        [ 1.0098314  , 2.545418 ],
                        [ 1.0133003  , 2.5456414],
                        [ 1.0149127  , 2.5457938],
                        [ 1.016727   , 2.5458374],
                        [ 1.0190794  , 2.5457652],
                        [ 1.0214754  , 2.5455785],
                        [ 1.0244205  , 2.5452962],
                        [ 1.0280783  , 2.5449915],
                        [ 1.0310009  , 2.5447795],
                        [ 1.033982   , 2.544909 ],
                        [ 1.037012   , 2.5456038],
                        [ 1.040134   , 2.5471582],
                        [ 1.0437244  , 2.5496995],
                        [ 1.0470997  , 2.5532656],
                        [ 1.0498221  , 2.5579112],
                        [ 1.0523707  , 2.5638719],
                        [ 1.0550792  , 2.571749 ],
                        [ 1.058282   , 2.5825958],
                        [ 1.0623422  , 2.597505 ],
                        [ 1.0676103  , 2.617394 ],
                        [ 1.0746381  , 2.6432145],
                        [ 1.0838649  , 2.675958 ],
                        [ 1.0953643  , 2.7149792],
                        [ 1.1083289  , 2.758064 ],
                        [ 1.1223836  , 2.8030307],
                        [ 1.1372775  , 2.848576 ],
                        [ 1.1529655  , 2.893857 ],
                        [ 1.1695075  , 2.9391644],
                        [ 1.1875534  , 2.9879549],
                        [ 1.2087643  , 3.0503156],
                        [ 1.235187   , 3.1392288],
                        [ 1.2663157  , 3.2547712],
                        [ 1.2998307  , 3.385529 ],
                        [ 1.3317308  , 3.506738 ],
                        [ 1.3638818  , 3.6296408],
                        [ 1.3992813  , 3.7747912],
                        [ 1.4406325  , 3.9637785],
                        [ 1.4893966  , 4.214814 ],
                        [ 1.5469016  , 4.550651 ],
                        [ 1.6211523  , 5.064669 ],
                        [ 1.7115484  , 5.8241506],
                        [ 1.8044264  , 6.780392 ],
                        [ 1.8850249  , 7.7789836],
                        [ 1.9490186  , 8.700374 ],
                        [ 1.9941084  , 9.401596 ],
                        [ 2.023349   , 9.8528385],
                        [ 2.042419  , 10.1270075],
                        [ 2.054471  , 10.279838 ],
                        [ 2.0623703 , 10.365189 ],
                        [ 2.0676546 , 10.416825 ],
                        [ 2.0717437 , 10.461722 ],
                        [ 2.0737932 , 10.480462 ]])

    
#print(f'Difference {weight_prof - weight_prof_1}')

mymetrics = ['mean_absolute_error']
valfunc = 'val_mean_absolute_error'


def my_gradient_tf(a):
    return a[:, 1:] - a[:, 0:-1]


def calc_heatingrates_tf_dp(flux_dn, flux_up, dp):
    #  flux_net =   flux_up   - flux_dn
    F = tf.subtract(flux_up, flux_dn)
    dF = my_gradient_tf(F)
    dFdp = tf.divide(dF, dp)
    coeff = -844.2071713147411  # -(24*3600) * (9.81/1004)
    dTdt_day = tf.multiply(coeff, dFdp)
    return dTdt_day


def CustomLoss(y_true, y_pred, dp, rsd_top):
    # err_flux =  tf.math.reduce_mean(tf.math.square(weight_prof*(y_true - y_pred)),axis=-1)
    err_flux = K.mean(K.square(weight_prof*(y_true - y_pred)))
    # err_flux =  K.mean(K.square((y_true - y_pred)))

    rsd_true = tf.math.multiply(y_true[:, :, 0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:, :, 0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:, :, 1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:, :, 1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)
    # err_hr = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))

    # alpha   = 1e-6
    # alpha   = 1e-5
    alpha = 1e-4
    return (alpha) * err_hr + (1 - alpha)*err_flux


def rmse_hr(y_true, y_pred, dp, rsd_top):

    rsd_true = tf.math.multiply(y_true[:, :, 0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:, :, 0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:, :, 1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:, :, 1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    # return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))


def bias_hr(y_true, y_pred, dp, rsd_top):

    rsd_true = tf.math.multiply(y_true[:, :, 0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:, :, 0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:, :, 1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:, :, 1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    # return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    return K.mean(HR_true - HR_pred)


def rmse_flux(y_true, y_pred, dp, rsd_top):

    rsd_top_2 = tf.expand_dims(rsd_top, axis=2)
    y_true_scaled = tf.math.multiply(y_true, rsd_top_2)
    y_pred_scaled = tf.math.multiply(y_pred, rsd_top_2)

    return K.sqrt(K.mean(K.square(y_true_scaled - y_pred_scaled)))


def bias_flux(y_true, y_pred, dp, rsd_top):

    rsd_top_2 = tf.expand_dims(rsd_top, axis=2)
    y_true_scaled = tf.math.multiply(y_true, rsd_top_2)
    y_pred_scaled = tf.math.multiply(y_pred, rsd_top_2)

    return K.mean(y_true_scaled - y_pred_scaled)


# MODEL TRAINING CODE
if True:

    # Model architecture
    # Activation for first
    # activ0 = 'relu'
    activ0 = 'linear'
    # activ0 = 'softsign'
    # activ0 = 'tanh'
    activ_last = 'relu'
    activ_last = 'sigmoid'
    # activ_last   = 'linear'

    epoch_period = 2000
    n_epochs = 0  # 565 #0 # start epoch
    # steps_per_epoch = 924
    epochs = 100000
    patience = 50  # 400 #25
    lossfunc = losses.mean_squared_error
    batch_size = 1024
    # batch_size  = 2048

    nneur = 16
    nlay = 60
    nx_main = 10  # 5-gases + temp + pressure + mu + lwp + iwp
    nx_aux = 1
    ny = 2  # rsd and rsu

    lr = 0.001  # DEFAULT!
    optim = optimizers.Adam(learning_rate=lr)

    # shape(x_tr_m) = (nsamples, nseq, nfeatures_seq)
    # shape(x_tr_s) = (nsamples, nfeatures_aux)
    # shape(y_tr)   = (nsamples, nseq, noutputs)

    # Main inputs associated with RNN layer (sequence dependent)
    # inputs = Input(shape=(None,nx_main),name='inputs_main')
    inputs = Input(shape=(nlay, nx_main), name='inputs_main')

    # Optionally, use auxiliary inputs that do not dependend on sequence?

    # if use_auxinputs:  # commented out cos I need albedo for the loss function, easier to have it separate
    inp_aux_albedo = Input(
        shape=(nx_aux), name='inputs_aux_albedo')  # sfc_albedo

    # Target outputs: these are fed as part of the input to avoid problem with
    # validation data where TF complained about wrong shape
    # target  = Input((None,ny))
    target = Input((nlay+1, ny))

    # other inputs required to compute heating rate
    dpres = Input((nlay,))
    incflux = Input((nlay+1))

    # hidden0,last_state = layers.SimpleRNN(nneur,return_sequences=True,return_state=True)(inputs)
    hidden0, last_state = layers.GRU(
        nneur, return_sequences=True, return_state=True)(inputs)

    last_state_plus_albedo = tf.concat([last_state, inp_aux_albedo], axis=1)

    mlp_surface_outp = Dense(nneur, activation=activ0,
                             name='dense_surface')(last_state_plus_albedo)

    hidden0_lev = tf.concat([hidden0, tf.reshape(
        mlp_surface_outp, [-1, 1, nneur])], axis=1)

    hidden1 = layers.GRU(nneur, return_sequences=True,
                         go_backwards=True)(hidden0_lev)
    # hidden1 = layers.SimpleRNN(nneur,return_sequences=True,go_backwards=True)(hidden0_lev)

    ###### Only processing change from original ######
    hidden1 = tf.reverse(hidden1, [1])
    ##################################################

    # try concatinating hidden0 and hidden 1 instead
    hidden_concat = tf.concat([hidden0_lev, hidden1], axis=2)

    hidden2 = layers.GRU(nneur, return_sequences=True)(hidden_concat)
    # hidden2 = layers.SimpleRNN(nneur,return_sequences=True)(hidden_concat)

    outputs = TimeDistributed(layers.Dense(
        ny, activation=activ_last), name='dense_output')(hidden2)

    model = Model(inputs=[inputs, inp_aux_albedo, target,
                  dpres, incflux], outputs=outputs)

    model.add_metric(rmse_flux(target, outputs, dpres, incflux), 'rmse_flux')
    model.add_metric(rmse_hr(target, outputs, dpres, incflux), 'rmse_hr')
    # model.add_metric(bias_flux(target,outputs,dpres,incflux),'bias_flux')
    # model.add_metric(bias_hr(target,outputs,dpres,incflux),'bias_hr')

    # model.add_metric(rmse_hr(target,outputs,inp_aux_albedo,dpres,incflux),'rmse_hr')
    # model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')

    model.add_loss(CustomLoss(target, outputs, dpres, incflux))
    # model.add_loss(losses.mean_squared_error(target,outputs))
    model.compile(optimizer=optim, loss='mse')

    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss',  patience=patience, verbose=1,
                               mode='min', restore_best_weights=True)]

    train_input_dir = "/data-T1/hws/CAMS/processed_data/training/2008/"
    validation_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1, 13)]
    train_input_files = [
        f'{train_input_dir}Flux_Ukkonen-2008-{month}.nc' for month in months]
    validation_input_files = [
        f'{validation_input_dir}Flux_Ukkonen-2008-{month}.nc' for month in months]

    generator_training = data_generation.InputSequence(
        train_input_files, batch_size)
    # generator_training = generator_training.batch(batch_size)
    generator_validation = data_generation.InputSequence(
        validation_input_files, batch_size)

    # generator_cross_validation = generator_cross_validation.batch(batch_size)

    if n_epochs > 0:
        del model
        model = tf.keras.models.load_model(
            datadir + training_model_name + str(n_epochs))

    if is_train:
        while n_epochs < epochs:

            history = model.fit(generator_training,
                                epochs=epoch_period,
                                shuffle=False, verbose=1,
                                validation_data=generator_validation, callbacks=callbacks)

            # history = model.fit_generator(generator_training, \
            # steps_per_epoch, epochs=epoch_period,   \
            # validation_data=generator_cross_validation, #callbacks=callbacks)

            # print(history.history.keys())
            # print("number of epochs = " + str(history.history['epoch']))
            print(str(history.history['rmse_hr']))

            nn_epochs = len(history.history['rmse_hr'])
            if nn_epochs < epoch_period:
                model.save(datadir + training_model_name + str(n_epochs + nn_epochs))
                print("Writing FINAL model. N_epochs = " +
                      str(n_epochs + nn_epochs))
                break
            else:
                n_epochs = n_epochs + epoch_period
                model.save(datadir + training_model_name + str(n_epochs))
                print(f"Writing model (n_epochs = {n_epochs})")

        del model
        model = tf.keras.models.load_model(
            datadir + training_model_name + str(n_epochs))

    else:
        start_time = 0

        class MyCallback(tf.keras.callbacks.Callback):
            def on_test_batch_begin(self, batch, logs=None):
                global start_time
                global count

                # start_time = time.process_time_ns()
                start_time = time.perf_counter_ns()

            def on_test_batch_end(self, batch, logs=None):
                global start_time
                global total_elapsed_time_ns
                # end_time = time.process_time_ns()
                end_time = time.perf_counter_ns()
                total_elapsed_time_ns += end_time - start_time
        for year in ['2009', '2015', '2020',]:
            total_elapsed_time_ns = 0
            print(f'Year = {year}')
            testing_input_dir = f"/data-T1/hws/CAMS/processed_data/testing/{year}/"
            testing_input_files = [
                f'{testing_input_dir}Flux_Ukkonen-{year}-{month}.nc' for month in months]
            # Best epoch was 515 but writes out model at epoch 565
            # when training terminated
            n_epoch = 565

            generator_testing = data_generation.InputSequence(
                testing_input_files, batch_size)

            print(f"n_epoch = {n_epoch}")
            model = tf.keras.models.load_model(
                datadir + evaluation_model_name + str(n_epoch))
            # model.add_metric(bias_hr(target,outputs,dpres,incflux),'bias_hr')
            # model.compile(optimizer=optim,loss='mse',metrics=[bias_hr,bias_flux])

            model.evaluate(x=generator_testing,
                           callbacks=[MyCallback()])
            print(
                f'Elapsed time in seconds = {float(total_elapsed_time_ns) / float(10e9)}')
            print(f'Elapsed time in ns = {total_elapsed_time_ns}')
