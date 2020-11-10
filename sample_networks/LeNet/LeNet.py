batch_size = 100
num_classes = 10
epochs = 300
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SeparableConv2D, DepthwiseConv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras import metrics
from keras import callbacks
from apscheduler.schedulers.background import BackgroundScheduler
import pickle
import psutil
import time

process = psutil.Process()

mcpu = 0
mmem = 0

def get_info():
    global mmem
    mem = process.memory_info().rss
    if mmem < mem:
        mmem = mem

scheduler = BackgroundScheduler()
scheduler.add_job(get_info, 'interval', seconds=1)


data_augmentation = True

#load saved data
pkl_file = open('/exports/home/j_liu21/projects/genetic_algorithms/x_train.pkl', 'rb')
x_train = pickle.load(pkl_file, encoding='latin1')
pkl_file.close()

############################################################
x_data_len = len(x_train)
#end = int(.2*x_data_len)
#x_train_train = x_train[0:end]
#x_train_valid = x_train[end:]
print(x_data_len)
x_train_train = x_train[0:40000]
x_train_valid = x_train[40000:50000]
############################################################

pkl_file = open('/exports/home/j_liu21/projects/genetic_algorithms/y_train.pkl', 'rb')
y_train = pickle.load(pkl_file, encoding='latin1')
pkl_file.close()

############################################################
#y_train_train = y_train[0:end]
#y_train_valid = y_train[end:]
y_train_train = y_train[0:40000]
y_train_valid = y_train[40000:50000]
############################################################

pkl_file = open('/exports/home/j_liu21/projects/genetic_algorithms/x_test.pkl', 'rb')
x_test = pickle.load(pkl_file, encoding='latin1')
pkl_file.close()

pkl_file = open('/exports/home/j_liu21/projects/genetic_algorithms/y_test.pkl', 'rb')
y_test = pickle.load(pkl_file, encoding='latin1')
pkl_file.close()

# Convert class vectors to binary class matrices.
y_train_train = keras.utils.to_categorical(y_train_train, num_classes)
y_train_valid = keras.utils.to_categorical(y_train_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(20, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(strides=(2,2) , pool_size = (2, 2), padding='same'))
model.add(Conv2D(50, (5, 5), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(strides=(2,2) , pool_size = (2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(500, activation='relu') )
model.add(Dense(num_classes, activation='softmax') )

print(model.summary())

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
#opt = keras.optimizers.SGD(learning_rate = .01, decay=1e-6)

es = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1)

scheduler.start()

model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

x_train_train = x_train_train.astype('float32')
x_train_valid = x_train_valid.astype('float32')
x_test = x_test.astype('float32')
x_train_train /= 255
x_train_valid /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train_train, y_train_train,
              batch_size = batch_size,
              epochs = epochs,
              validation_data = (x_train_valid, y_train_valid),
              shuffle = True)
else:
    print('Using real-time data augmentation.')
                   # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center = False,             # set input mean to 0 over the dataset
        samplewise_center = False,              # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,   # divide each input by its std
        zca_whitening = False,                  # apply ZCA whitening
        zca_epsilon = 1e-06,                    # epsilon for ZCA whitening
        rotation_range = 0,                     # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,                # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,               # randomly shift images vertically (fraction of total height)
        shear_range = 0.,                       # set range for random shear
        zoom_range = 0.,                        # set range for random zoom
        channel_shift_range = 0.,               # set range for random channel shifts
        fill_mode = 'nearest',                  # set mode for filling points outside the input boundaries
        cval = 0.,                              # value used for fill_mode = "constant"
        horizontal_flip = True,                 # randomly flip images
        vertical_flip = False,                  # randomly flip images
        rescale = None,                         # set rescaling factor (applied before any other transformation)
        preprocessing_function = None,          # set function that will be applied on each input
        data_format = None,                     # image data format, either "channels_first" or "channels_last"
        validation_split = 0.0 )                # fraction of images reserved for validation (strictly between 0 and 1)


    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).

    datagen.fit(x_train_train)

    # Fit the model on the batches generated by datagen.flow().

    model.fit_generator(datagen.flow(x_train_train, y_train_train, batch_size = batch_size),
                                     steps_per_epoch = 100,
                                     epochs = epochs,
                                     validation_data = (x_train_valid, y_train_valid),
                                     workers = 8,
                                     callbacks = [es],
                                     verbose = 1 )

    #model.fit_generator(datagen.flow(x_train_train, y_train_train, batch_size = batch_size),
    #                                 steps_per_epoch = 100,
    #                                 epochs = epochs,
    #                                 validation_data = (x_train_valid, y_train_valid),
    #                                 workers = 8 )


# Score trained model.
t0 = time.time()
scores_train_train = model.evaluate(x_train_train, y_train_train, verbose = 0)
t1 = time.time()
train_eval_time = t1-t0
print('train_eval_time: ', train_eval_time)

t2 = time.time()
scores_test = model.evaluate(x_test, y_test, verbose = 0)
t3 = time.time()
test_eval_time = t3-t2
print('test_eval_time: ', test_eval_time)

scheduler.shutdown()
cpu_time = process.cpu_times().user

print('Training_loss: {} Test_accuracy: {} Mem: {} CPU: {}'.format(scores_train_train[0], scores_test[1], mmem, cpu_time) )

