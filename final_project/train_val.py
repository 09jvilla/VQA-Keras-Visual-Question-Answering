from data_processing import *
from keras.callbacks import ModelCheckpoint

def train(epoch=10, batch_size=256, data_limit=215359, model_type="VGG19", model_obj=None):
    dropout_rate = 0.5
    
    if model_type=="VGG19":
        train_X, train_y = read_data(data_limit)        
        val_X, val_y, multi_val_y = get_val_data()
    else:
        train_X, train_y = read_data(data_limit, data_img=resnet_train_img, embedding_img_dim=2048)        
        val_X, val_y, multi_val_y = get_val_data(data_img=resnet_test_img, embedding_img_dim=2048)

    checkpointer = ModelCheckpoint(filepath=ckpt_model_weights_filename, verbose=1, monitor='loss', save_best_only=False)
    #history = model_obj.fit(train_X, train_y, epochs=epoch, batch_size=batch_size,
    #                    callbacks=[checkpointer], shuffle="batch", validation_data=(val_X, val_y))

    history = model_obj.fit(train_X, train_y, epochs=epoch, batch_size=batch_size,
                        callbacks=[checkpointer], shuffle="batch")

    return model_obj, history
    
def val(model_obj, model_type="VGG19"):
    if model_type=="VGG19":
        val_X, val_y, multi_val_y = get_val_data() 
    else:
        val_X, val_y, multi_val_y = get_val_data(data_img=resnet_test_img, embedding_img_dim=2048) 

    metric_vals = model_obj.evaluate(val_X, val_y)
    
    print("")
    for metric_name, metric_val in zip(model_obj.metrics_names, metric_vals):
        print(str(metric_name) + " is " + str(metric_val))

    # Comparing prediction against multiple choice answers
    true_positive = 0
    preds = model.predict(val_X)
    pred_classes = [np.argmax(_) for _ in preds]
    for i, _ in enumerate(pred_classes):
        if _ in multi_val_y[i]:
            true_positive += 1
    print("True positive rate: " +  str(np.float(true_positive)/len(pred_classes)))