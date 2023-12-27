def testing_all(net, train_loader, test_loader, fold_number):
    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = computeDiceOneHot()
    mseLoss = nn.MSELoss()

    device = torch.device('cuda:2')
    za = 0
    if torch.cuda.is_available():
        net.to(device)
        softMax.to(device)
        CE_loss.to(device)
        Dice_loss.to(device)

    batch_size = 4
    net.eval()
    val_loss = []
    val_dice_0 = []
    val_dice_1 = []
    val_dice_2 = []
    val_dice_3 = []
    val_dice_c = []
    val_iou_c = []

    precision_0 = []
    recall_0 = []
    specificity_0 = []
    accuracy_0 = []
    precision_1 = []
    recall_1 = []
    specificity_1 = []
    accuracy_1 = []
    precision_2 = []
    recall_2 = []
    specificity_2 = []
    accuracy_2 = []
    precision_3 = []
    recall_3 = []
    specificity_3 = []
    accuracy_3 = []
    precision_comb = []
    recall_comb = []
    specificity_comb = []
    accuracy_comb = []
    
    max_dice = []
    max_iou = []
    max_precision = []
    max_recall = []
    max_specificity = []
    max_accuracy = []
    max_f1 = []
    
    ens_dice = []
    ens_iou = []
    ens_precision = []
    ens_recall = []
    ens_specificity = []
    ens_accuracy = []
    ens_f1 = []
    
    f1_0 = []
    f1_1 = []
    f1_2 = []
    f1_3 = []
    f1_comb = []
    
    rms_max = []
    asd_max = []
    hfd_max = []

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()

    for j, data in enumerate(test_loader):
        image, labels = data

        if image.size(0) != batch_size:
            continue

        MRI = image.to(device) #to_var(image)
        Segmentation = labels.to(device) #to_var(labels)
        with torch.no_grad():
            outputs0_2, outputs1_2, outputs2_2, outputs3_2 = net(MRI.float())

        #print('outputs 0: ', outputs0_2.shape)
        #print('outputs 1: ', outputs1_2.shape)
        #print('outputs 2: ', outputs2_2.shape)
        #print('outputs 3: ', outputs3_2.shape)
        ens_pred = weights_ensemble(outputs0_2, outputs1_2, outputs2_2, outputs3_2, Segmentation)
        #ens_pred = ens_pred.to(device)

        segmentation_prediction = (
            outputs0_2 + outputs1_2 + outputs2_2 + outputs3_2
            ) / 4
        predClass_y = softMax(segmentation_prediction)

        loss3_2 = CE_loss(outputs3_2, Segmentation)

        lossG = (loss3_2)

        max_pred0 = torch.argmax(outputs0_2, dim = 1)
        max_pred0 = torch.squeeze(max_pred0, dim = 0)

        max_pred1 = torch.argmax(outputs1_2, dim = 1)
        max_pred1 = torch.squeeze(max_pred1, dim = 0)

        max_pred2 = torch.argmax(outputs2_2, dim = 1)
        max_pred2 = torch.squeeze(max_pred2, dim = 0)

        max_pred3 = torch.argmax(outputs3_2, dim = 1)
        max_pred3 = torch.squeeze(max_pred3, dim = 0)

        max_pred_comb = torch.argmax(segmentation_prediction, dim = 1)
        max_pred_comb = torch.squeeze(max_pred_comb, dim = 0)
        
        m_pred_comb = max_pred_comb.detach().cpu().numpy().astype(np.float64)
        ch_seg = Segmentation.detach().cpu().numpy().astype(np.float64)
        #print('unique: ', np.unique(max_pred_comb))
        #print('max pred shape: ', max_pred_comb.shape, max_pred_comb.dtype)
        #print('segmentation shape: ', Segmentation.shape, Segmentation.dtype)
        #qmeasures = computeQualityMeasures(max_pred_comb[1, :, :], Segmentation[1, :, :], np.array([1,1,1]))
        qmeasures = computeQualityMeasures(m_pred_comb, ch_seg, np.array([1,1,1]))
        rms_max.append(qmeasures['rmse'])
        asd_max.append(qmeasures['msd'])
        hfd_max.append(qmeasures['hd'])
        #print('RMS: ', qmeasures['rmse'])
        #print('ASD: ', qmeasures['msd'])
        #print('HFD: ', qmeasures['hd'])
        
        #print('argmax shape: ', max_pred.detach().cpu().numpy().shape)
        #print('before dice: ', max_pred3.shape, ' ', ens_pred.shape)
        dsc0, iou0 = dice_coef(max_pred0, Segmentation)
        dsc1, iou1 = dice_coef(max_pred1, Segmentation)
        dsc2, iou2 = dice_coef(max_pred2, Segmentation)
        dsc3, iou3 = dice_coef(max_pred3, Segmentation)
        dscc, iouc = dice_coef(max_pred_comb, Segmentation)
        dsce, ioue = dice_coef(ens_pred, Segmentation)
        
        result_array = [max_pred0, max_pred1, max_pred2, max_pred3]
        dice_array = [dsc0.detach().cpu().numpy(), dsc1.detach().cpu().numpy(), dsc2.detach().cpu().numpy(), dsc3.detach().cpu().numpy()]
        iou_array = [iou0.detach().cpu().numpy(), iou1.detach().cpu().numpy(), iou2.detach().cpu().numpy(), iou3.detach().cpu().numpy()]
        #iou_array = np.array(iou_array)
        #result_array = np.array(result_array)
        dice_array = np.array(dice_array)
        
        h_dice = np.max(dice_array)
        #print(dice_array)
        loc_max = np.where(dice_array == h_dice)
        #print(loc_max)
        #print(loc_max[0])
        #if len(loc_max[0]) > 1:
            #h_iou = iou_array[0]
        #else:
            #print('actual one')
        
        #print('max index: ', int(loc_max[0]))
        h_iou = iou_array[int(loc_max[0])]
        h_pred = result_array[int(loc_max[0])]
        
        ens_dice.append(dsce.detach().cpu().numpy())
        ens_iou.append(ioue.detach().cpu().numpy())
        prec, rec, spec, acc, iou = prec_rec(ens_pred.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        ens_precision.append(prec)
        ens_recall.append(rec)
        ens_f1.append(f1_sc)
        ens_specificity.append(spec)
        ens_accuracy.append(acc)
        
        max_dice.append(h_dice)
        max_iou.append(h_iou)
        prec, rec, spec, acc, iou = prec_rec(h_pred.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        max_precision.append(prec)
        max_recall.append(rec)
        max_f1.append(f1_sc)
        max_specificity.append(spec)
        max_accuracy.append(acc)
        
        val_dice_0.append(dsc0.detach().cpu().numpy())
        val_dice_1.append(dsc1.detach().cpu().numpy())
        val_dice_2.append(dsc2.detach().cpu().numpy())
        val_dice_3.append(dsc3.detach().cpu().numpy())
        val_dice_c.append(dscc.detach().cpu().numpy())
        val_iou_c.append(iouc.detach().cpu().numpy())
        val_loss.append(lossG.item())

        prec, rec, spec, acc, iou = prec_rec(max_pred0.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        precision_0.append(prec)
        recall_0.append(rec)
        f1_0.append(f1_sc)
        specificity_0.append(spec)
        accuracy_0.append(acc)

        prec, rec, spec, acc, iou = prec_rec(max_pred1.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        precision_1.append(prec)
        recall_1.append(rec)
        f1_1.append(f1_sc)
        specificity_1.append(spec)
        accuracy_1.append(acc)

        prec, rec, spec, acc, iou = prec_rec(max_pred2.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        precision_2.append(prec)
        recall_2.append(rec)
        f1_2.append(f1_sc)
        specificity_2.append(spec)
        accuracy_2.append(acc)

        prec, rec, spec, acc, iou = prec_rec(max_pred3.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        precision_3.append(prec)
        recall_3.append(rec)
        f1_3.append(f1_sc)
        specificity_3.append(spec)
        accuracy_3.append(acc)

        prec, rec, spec, acc, iou = prec_rec(max_pred_comb.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
        f1_sc = (2*prec*rec) / (prec + rec)
        precision_comb.append(prec)
        recall_comb.append(rec)
        f1_comb.append(f1_sc)
        specificity_comb.append(spec)
        accuracy_comb.append(acc)
    
    file_name = 'Testing_with_densenet_modified.txt'
    with open(file_name, 'a') as f:
        f.write(f'-----------------Fold Number: {fold_number} ---------------------')
        f.write('\n')
        f.write(f'Ensembled Dice: {np.mean(ens_dice)}')
        f.write('\n')
        f.write(f'Ensembled IOU: {np.mean(ens_iou)}')
        f.write('\n')
        f.write(f'Ensembled Precision: {np.mean(ens_precision)}')
        f.write('\n')
        f.write(f'Ensembled recall: {np.mean(ens_recall)}')
        f.write('\n')
        f.write(f'Ensembled specificity: {np.mean(ens_specificity)}')
        f.write('\n')
        f.write(f'Ensembled Accuracy: {np.mean(ens_accuracy)}')
        f.write('\n')
        f.write(f'Ensembled F1: {np.mean(ens_f1)}')
        f.write('\n')
        f.write(f'Max Dice: {np.mean(max_dice)}')
        f.write('\n')
        f.write(f'Max IOU: {np.mean(max_iou)}')
        f.write('\n')
        f.write(f'Max Precision: {np.mean(max_precision)}')
        f.write('\n')
        f.write(f'Max recall: {np.mean(max_recall)}')
        f.write('\n')
        f.write(f'Max specificity: {np.mean(max_specificity)}')
        f.write('\n')
        f.write(f'Max Accuracy: {np.mean(max_accuracy)}')
        f.write('\n')
        f.write(f'Max F1: {np.mean(max_f1)}')
        f.write('\n')
        f.write(f'Max RMS: {np.mean(rms_max)}')
        f.write('\n')
        f.write(f'Max ASD: {np.mean(asd_max)}')
        f.write('\n')
        f.write(f'Max HFD: {np.mean(hfd_max)}')
        f.write('\n')
        f.write(f'Combined Dice: {np.mean(val_dice_c)}')
        f.write('\n')
        f.write(f'Combined IOU: {np.mean(val_iou_c)}')
        f.write('\n')
        f.write(f'Precision combined: {np.mean(precision_comb)}')
        f.write('\n')
        f.write(f'Recall combined: {np.mean(recall_comb)}')
        f.write('\n')
        f.write(f'Specificity combined: {np.mean(specificity_comb)}')
        f.write('\n')
        f.write(f'Accuracy combined: {np.mean(accuracy_comb)}')
        f.write('\n')
        f.write(f'Dice 0: {np.mean(val_dice_0)}')
        f.write('\n')
        f.write(f'Dice 1: {np.mean(val_dice_1)}')
        f.write('\n')
        f.write(f'Dice 2: {np.mean(val_dice_2)}')
        f.write('\n')
        f.write(f'Dice 3: {np.mean(val_dice_3)}')
        f.write('\n')
        f.write(f'Precision 0: {np.mean(precision_0)}')
        f.write('\n')
        f.write(f'Precision 1: {np.mean(precision_1)}')
        f.write('\n')
        f.write(f'Precision 2: {np.mean(precision_2)}')
        f.write('\n')
        f.write(f'Precision 3: {np.mean(precision_3)}')
        f.write('\n')
        f.write(f'Recall 0: {np.mean(recall_0)}')
        f.write('\n')
        f.write(f'Recall 1: {np.mean(recall_1)}')
        f.write('\n')
        f.write(f'Recall 2: {np.mean(recall_2)}')
        f.write('\n')
        f.write(f'Recall 3: {np.mean(recall_3)}')
        f.write('\n')
        f.write(f'Specificity 0: {np.mean(specificity_0)}')
        f.write('\n')
        f.write(f'Specificity 1: {np.mean(specificity_1)}')
        f.write('\n')
        f.write(f'Specificity 2: {np.mean(specificity_2)}')
        f.write('\n')
        f.write(f'Specificity 3: {np.mean(specificity_3)}')
        f.write('\n')
        f.write(f'Accuracy 0: {np.mean(accuracy_0)}')
        f.write('\n')
        f.write(f'Accuracy 1: {np.mean(accuracy_1)}')
        f.write('\n')
        f.write(f'Accuracy 2: {np.mean(accuracy_2)}')
        f.write('\n')
        f.write(f'Accuracy 3: {np.mean(accuracy_3)}')
        f.write('\n')
        f.write('\n')
    
    print('Fold: ', fold_number)
    print('Testing Loss: ', np.mean(val_loss))
    print('Testing Dice 0: ', np.mean(val_dice_0))
    print('Testing Dice 1: ', np.mean(val_dice_1))
    print('Testing Dice 2: ', np.mean(val_dice_2))
    print('Testing Dice 3: ', np.mean(val_dice_3))
    print('Testing Dice combined: ', np.mean(val_dice_c))

    print('Precision 0: ', np.mean(precision_0))
    print('Recall 0: ', np.mean(recall_0))
    print('Precision 1: ', np.mean(precision_1))
    print('Recall 1: ', np.mean(recall_1))
    print('Precision 2: ', np.mean(precision_2))
    print('Recall 2: ', np.mean(recall_2))
    print('Precision 3: ', np.mean(precision_3))
    print('Recall 3: ', np.mean(recall_3))
    print('Precision combined: ', np.mean(precision_comb))
    print('Recall combined: ', np.mean(recall_comb))

    print('F1 0: ', np.mean(f1_0))
    print('F1 1: ', np.mean(f1_1))
    print('F1 2: ', np.mean(f1_2))
    print('F1 3: ', np.mean(f1_3))
    print('F1 comb: ', np.mean(f1_comb))
    print('Specificity comb: ', np.mean(specificity_comb))
    print('Accuracy comb: ', np.mean(accuracy_comb))
    
    return [np.mean(ens_dice), np.mean(ens_iou), np.mean(ens_precision), np.mean(ens_recall),
            np.mean(ens_specificity), np.mean(ens_accuracy), np.mean(ens_f1), np.mean(max_dice), np.mean(max_iou), np.mean(max_precision), np.mean(max_recall), np.mean(max_specificity),
           np.mean(max_accuracy), np.mean(max_f1), np.mean(rms_max), np.mean(asd_max), np.max(np.mean(hfd_max)), np.mean(val_dice_c), np.mean(val_iou_c), np.mean(precision_comb),
           np.mean(recall_comb), np.mean(specificity_comb), np.mean(accuracy_comb), np.mean(val_dice_0), np.mean(val_dice_1),
           np.mean(val_dice_2), np.mean(val_dice_3), np.mean(precision_0), np.mean(precision_1), np.mean(precision_2),
           np.mean(precision_3), np.mean(recall_0), np.mean(recall_1), np.mean(recall_2), np.mean(recall_3),
           np.mean(specificity_0), np.mean(specificity_1), np.mean(specificity_2), np.mean(specificity_3),
           np.mean(accuracy_0), np.mean(accuracy_1), np.mean(accuracy_2), np.mean(accuracy_3)]



train_path = "Thyroid_Nodule/total_data.npy"
test_path = "Thyroid_Nodule/total_data.npy"

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_0_modellll_0.8307888.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 0)
test_set = DataPrepval(test_path, 0)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results1 = testing_all(net, train_loader, test_loader, 0)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_1_modellll_0.83639127.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 1)
test_set = DataPrepval(test_path, 1)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results2 = testing_all(net, train_loader, test_loader, 1)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_2_modellll_0.8104825.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 2)
test_set = DataPrepval(test_path, 2)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results3 = testing_all(net, train_loader, test_loader, 2)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_3_modellll_0.8600036.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 3)
test_set = DataPrepval(test_path, 3)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results4 = testing_all(net, train_loader, test_loader, 3)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_4_modellll_0.8274825.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 4)
test_set = DataPrepval(test_path, 4)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results5 = testing_all(net, train_loader, test_loader, 4)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_5_modellll_0.8401052.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 5)
test_set = DataPrepval(test_path, 5)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results6 = testing_all(net, train_loader, test_loader, 5)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_6_modellll_0.7995591.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 6)
test_set = DataPrepval(test_path, 6)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results7 = testing_all(net, train_loader, test_loader, 6)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_7_modellll_0.8839694.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 7)
test_set = DataPrepval(test_path, 7)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results8 = testing_all(net, train_loader, test_loader, 7)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_8_modellll_0.8517256.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 8)
test_set = DataPrepval(test_path, 8)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results9 = testing_all(net, train_loader, test_loader, 8)

net = DAF_stack()
model_path = "Thyroid_Nodule/fold_saves_densenet/fold_9_modellll_0.8623544.pth"
net.load_state_dict(torch.load(model_path))
train_set = DataPrep(train_path, 9)
test_set = DataPrepval(test_path, 9)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, pin_memory=True)
results10 = testing_all(net, train_loader, test_loader, 9)

results1 = np.array(results1)
results2 = np.array(results2)
results3 = np.array(results3)
results4 = np.array(results4)
results5 = np.array(results5)
results6 = np.array(results6)
results7 = np.array(results7)
results8 = np.array(results8)
results9 = np.array(results9)
results10 = np.array(results10)


results1 = np.expand_dims(results1, axis=0)
results2 = np.expand_dims(results2, axis=0)
results3 = np.expand_dims(results3, axis=0)
results4 = np.expand_dims(results4, axis=0)
results5 = np.expand_dims(results5, axis=0)
results6 = np.expand_dims(results6, axis=0)
results7 = np.expand_dims(results7, axis=0)
results8 = np.expand_dims(results8, axis=0)
results9 = np.expand_dims(results9, axis=0)
results10 = np.expand_dims(results10, axis=0)

#print('results shape: ', results10.shape)
catted = np.concatenate((results1, results2, results3, results4, results5, results6, results7, results8, results9, results10))
#catted = np.concatenate((results7, results8, results9, results10))
#catted = results10
print('catted shape: ', catted.shape)

avg = np.mean(catted, axis = 0)
print('avg dice: ', avg[0])

file_name = 'Testing_with_densenet_modified.txt'
with open(file_name, 'a') as f:
    f.write(f'----------------- Averaged Results ---------------------')
    f.write('\n')
    f.write(f'Ensembled Dice: {avg[0]}')
    f.write('\n')
    f.write(f'Ensembled IOU: {avg[1]}')
    f.write('\n')
    f.write(f'Ensembled Precision: {avg[2]}')
    f.write('\n')
    f.write(f'Ensembled recall: {avg[3]}')
    f.write('\n')
    f.write(f'Ensembled specificity: {avg[4]}')
    f.write('\n')
    f.write(f'Ensembled Accuracy: {avg[5]}')
    f.write('\n')
    f.write(f'Ensembled F1: {avg[6]}')
    f.write('\n')
    f.write(f'Max Dice: {avg[7]}')
    f.write('\n')
    f.write(f'Max IOU: {avg[8]}')
    f.write('\n')
    f.write(f'Max Precision: {avg[9]}')
    f.write('\n')
    f.write(f'Max recall: {avg[10]}')
    f.write('\n')
    f.write(f'Max specificity: {avg[11]}')
    f.write('\n')
    f.write(f'Max Accuracy: {avg[12]}')
    f.write('\n')
    f.write(f'Max F1: {avg[13]}')
    f.write('\n')
    f.write(f'Max RMS: {avg[14]}')
    f.write('\n')
    f.write(f'Max ASD: {avg[15]}')
    f.write('\n')
    f.write(f'Max HFD: {avg[16]}')
    f.write('\n')
    f.write(f'Combined Dice: {avg[17]}')
    f.write('\n')
    f.write(f'Combined IOU: {avg[18]}')
    f.write('\n')
    f.write(f'Precision combined: {avg[19]}')
    f.write('\n')
    f.write(f'Recall combined: {avg[20]}')
    f.write('\n')
    f.write(f'Specificity combined: {avg[21]}')
    f.write('\n')
    f.write(f'Accuracy combined: {avg[22]}')
    f.write('\n')
    f.write(f'Dice 0: {avg[23]}')
    f.write('\n')
    f.write(f'Dice 1: {avg[24]}')
    f.write('\n')
    f.write(f'Dice 2: {avg[25]}')
    f.write('\n')
    f.write(f'Dice 3: {avg[26]}')
    f.write('\n')
    f.write(f'Precision 0: {avg[27]}')
    f.write('\n')
    f.write(f'Precision 1: {avg[28]}')
    f.write('\n')
    f.write(f'Precision 2: {avg[29]}')
    f.write('\n')
    f.write(f'Precision 3: {avg[30]}')
    f.write('\n')
    f.write(f'Recall 0: {avg[31]}')
    f.write('\n')
    f.write(f'Recall 1: {avg[32]}')
    f.write('\n')
    f.write(f'Recall 2: {avg[33]}')
    f.write('\n')
    f.write(f'Recall 3: {avg[34]}')
    f.write('\n')
    f.write(f'Specificity 0: {avg[35]}')
    f.write('\n')
    f.write(f'Specificity 1: {avg[36]}')
    f.write('\n')
    f.write(f'Specificity 2: {avg[37]}')
    f.write('\n')
    f.write(f'Specificity 3: {avg[38]}')
    f.write('\n')
    f.write(f'Accuracy 0: {avg[39]}')
    f.write('\n')
    f.write(f'Accuracy 1: {avg[40]}')
    f.write('\n')
    f.write(f'Accuracy 2: {avg[41]}')
    f.write('\n')
    f.write(f'Accuracy 3: {avg[42]}')
    f.write('\n')
    f.write('\n')
