from trainers import ease_evaluate, verbose
import time

def ease_runner(args, model, criterion, optimizer, train_data, vad_data_tr, vad_data_te, epoch, N, data_inf):

    epoch_start_time = time.time()

    model.fit(data_inf)

    val_loss, n100, r10, r20, r50 = ease_evaluate(args, model, vad_data_tr, vad_data_te)
    verbose(epoch, epoch_start_time, 0, val_loss, n100, r10, r20, r50)

    # model.reg_weight += 10
    
    return n100