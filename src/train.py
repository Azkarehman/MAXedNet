from sklearn.metrics import confusion_matrix, average_precision_score
import warnings
warnings.filterwarnings("ignore")

def prec_rec(pred, gt):
    tn, fp, fn, tp = confusion_matrix(pred.ravel(), gt.ravel()).ravel()
    
    prec = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    specificity = (tn)/(tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    
    iou = (tp)/(tp + fp + fn)
    
    return prec, recall, specificity, accuracy, iou

def dice_coef(pred, target):
    smooth = 0.001
    #print('in dice: ', pred.size, ' ', target.size)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    iou = (intersection + smooth) / (m1.sum() + m2.sum() - intersection + smooth)
    return dice, iou

def min_loss(loss_arr, out_arr, feats_arr):
    device = torch.device('cuda:3')
    loss_arr = np.array(loss_arr)
    out_arr = np.array(out_arr)
    feats_arr = np.array(feats_arr)
    #print(loss_arr)
    ml = np.min(loss_arr)
    loc_min = np.where(loss_arr == ml)
    #print(loc_min, int(loc_min[0]))
    
    CE_loss = nn.CrossEntropyLoss()
    CE_loss.to(device)
    
    loss = 0
    teacher = out_arr[int(loc_min[0])]
    teacher = np.squeeze(torch.argmax(teacher, dim = 1))
    
    teacher_feats = feats_arr[int(loc_min[0])]
    teacher_feats = np.squeeze(torch.argmax(teacher_feats, dim = 1))
    #print('teacher shape: ', teacher.shape, ' ', teacher_feats.shape)
    for i in range(4):
        if i != int(loc_min[0]):
            student = out_arr[i]
            student_feats = feats_arr[i]
            #print('student shape: ', student.shape)
            l = CE_loss(student, teacher)
            #ll = CE_loss(student_feats, teacher_feats)
            #dl = torch.dist(student_feats, teacher_feats) * 0.03
            loss = loss + l# + ll
    #mean_loss = loss / 3
    return loss


def train_test(net, train_loader, test_loader, fold_number):
    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = computeDiceOneHot()
    mseLoss = nn.MSELoss()

    device = torch.device('cuda:3')
    za = 0
    if torch.cuda.is_available():
        net.to(device)
        softMax.to(device)
        CE_loss.to(device)
        Dice_loss.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)

    BestDice, BestEpoch = 0, 0
    BestDice3D = [0,0,0,0]

    d1Val = []
    d2Val = []
    d3Val = []
    d4Val = []

    d1Val_3D = []
    d2Val_3D = []
    d3Val_3D = []
    d4Val_3D = []

    d1Val_3D_std = []
    d2Val_3D_std = []
    d3Val_3D_std = []
    d4Val_3D_std = []

    Losses = []
    record_train_dice = []
    record_train_loss = []
    record_val_dice = []
    record_val_loss = []


    epoch = 100
    batch_size = 6

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(0, epoch):
        net.train()
        lossVal = []
        train_dice = []
        train_loss = []
        val_dice = []
        val_loss = []
        val_dice_c = []
        precision_comb = []
        recall_comb = []
        f1_comb = []

        totalImages = len(train_loader)

        for j, data in enumerate(train_loader):
            image, labels = data

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizer.zero_grad()
            MRI = image.to(device) #to_var(image)
            Segmentation = labels.to(device) #to_var(labels)

            net.zero_grad()

            # Network outputs
            semVector_1_1, \
            semVector_2_1, \
            semVector_1_2, \
            semVector_2_2, \
            semVector_1_3, \
            semVector_2_3, \
            semVector_1_4, \
            semVector_2_4, \
            inp_enc0, \
            inp_enc1, \
            inp_enc2, \
            inp_enc3, \
            inp_enc4, \
            inp_enc5, \
            inp_enc6, \
            inp_enc7, \
            out_enc0, \
            out_enc1, \
            out_enc2, \
            out_enc3, \
            out_enc4, \
            out_enc5, \
            out_enc6, \
            out_enc7, \
            outputs0, \
            outputs1, \
            outputs2, \
            outputs3, \
            outputs0_2, \
            outputs1_2, \
            outputs2_2, \
            outputs3_2, refine4, refine3, refine2, refine1 = net(MRI.float())

            segmentation_prediction = (
                outputs0 + outputs1 + outputs2 + outputs3 +\
                outputs0_2 + outputs1_2 + outputs2_2 + outputs3_2
                ) / 8

            predClass_y = softMax(segmentation_prediction)

            # hint loss
            hl1 = torch.dist(refine1, refine4) * 0.01
            hl2 = torch.dist(refine2, refine4) * 0.01
            hl3 = torch.dist(refine3, refine4) * 0.01
            
            #loss0_3 = CE_loss(outputs0_2, outputs3_2) * 0.3
            #loss1_3 = CE_loss(outputs1_2, outputs3_2) * 0.3
            #loss2_3 = CE_loss(outputs2_2, outputs3_2) * 0.3

            # Cross-entropy loss
            #print('segmentation shape: ', Segmentation.shape)
            loss0 = CE_loss(outputs0, Segmentation)
            loss1 = CE_loss(outputs1, Segmentation)
            loss2 = CE_loss(outputs2, Segmentation)
            loss3 = CE_loss(outputs3, Segmentation)
            loss0_2 = CE_loss(outputs0_2, Segmentation)
            loss1_2 = CE_loss(outputs1_2, Segmentation)
            loss2_2 = CE_loss(outputs2_2, Segmentation)
            loss3_2 = CE_loss(outputs3_2, Segmentation)
            loss_arr = [loss0_2.detach().cpu().numpy(), loss1_2.detach().cpu().numpy(),
                        loss2_2.detach().cpu().numpy(), loss3_2.detach().cpu().numpy()]
            out_arr = [outputs0, outputs1, outputs2, outputs3]
            out_arr_refine = [outputs0_2, outputs1_2, outputs2_2, outputs3_2]
            feats_arr = [refine1, refine2, refine3, refine4]
            
            minl = min_loss(loss_arr, out_arr_refine, out_arr)
            
            lossSemantic1 = mseLoss(semVector_1_1, semVector_2_1)
            lossSemantic2 = mseLoss(semVector_1_2, semVector_2_2)
            lossSemantic3 = mseLoss(semVector_1_3, semVector_2_3)
            lossSemantic4 = mseLoss(semVector_1_4, semVector_2_4)

            lossRec0 = mseLoss(inp_enc0, out_enc0)
            lossRec1 = mseLoss(inp_enc1, out_enc1)
            lossRec2 = mseLoss(inp_enc2, out_enc2)
            lossRec3 = mseLoss(inp_enc3, out_enc3)
            lossRec4 = mseLoss(inp_enc4, out_enc4)
            lossRec5 = mseLoss(inp_enc5, out_enc5)
            lossRec6 = mseLoss(inp_enc6, out_enc6)
            lossRec7 = mseLoss(inp_enc7, out_enc7)
            
            #lossG = loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2
            
            lossG = 0.8 * (loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2)\
                + 0.2 * (lossSemantic1 + lossSemantic2 + lossSemantic3 + lossSemantic4) \
                + 0.1 * (lossRec0 + lossRec1 + lossRec2 + lossRec3 + lossRec4 + lossRec5 + lossRec6 + lossRec7) \
                + 0.2 * minl + hl1 + hl2 + hl3# + loss0_3 + loss1_3 + loss2_3
            

            # Compute the DSC
            max_pred = torch.argmax(segmentation_prediction, dim = 1)
            #print('argmax shape: ', max_pred.detach().cpu().numpy().shape)
            dsc, iou = dice_coef(max_pred, Segmentation)

            lossG.backward()
            optimizer.step()

            train_dice.append(dsc.detach().cpu().numpy())
            train_loss.append(lossG.item())

        net.eval()
        for j, data in enumerate(test_loader):
            image, labels = data
            
            if image.size(0) != batch_size:
                continue
            
            with torch.no_grad():
                MRI = image.to(device) #to_var(image)
                Segmentation = labels.to(device) #to_var(labels)
                outputs0_2, outputs1_2, outputs2_2, outputs3_2 = net(MRI.float())
            
            segmentation_prediction = (
                outputs0_2 + outputs1_2 + outputs2_2 + outputs3_2
                ) / 4
            predClass_y = softMax(segmentation_prediction)
            
            loss3_2 = CE_loss(outputs3_2, Segmentation)
            
            lossG = (loss3_2)
            
            #max_pred_comb = torch.argmax(segmentation_prediction, dim = 1)
            #max_pred_comb = torch.squeeze(max_pred_comb, dim = 0)
            #print('argmax shape: ', max_pred.detach().cpu().numpy().shape)
            #dscc, iouc = dice_coef(max_pred_comb, Segmentation)
            
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
            #print('argmax shape: ', max_pred.detach().cpu().numpy().shape)
            dsc0, iou0 = dice_coef(max_pred0, Segmentation)
            dsc1, iou1 = dice_coef(max_pred1, Segmentation)
            dsc2, iou2 = dice_coef(max_pred2, Segmentation)
            dsc3, iou3 = dice_coef(max_pred3, Segmentation)
            dscc, iouc = dice_coef(max_pred_comb, Segmentation)
            
            result_array = [max_pred0, max_pred1, max_pred2, max_pred3]
            dice_array = [dsc0.detach().cpu().numpy(), dsc1.detach().cpu().numpy(), dsc2.detach().cpu().numpy(), dsc3.detach().cpu().numpy()]
            iou_array = [iou0.detach().cpu().numpy(), iou1.detach().cpu().numpy(), iou2.detach().cpu().numpy(), iou3.detach().cpu().numpy()]
            dice_array = np.array(dice_array)
            max_dice = np.max(dice_array)
            #print(dice_array, ' ', max_dice)
            loc_max = np.where(dice_array == max_dice)
            #print(loc_max)
            #print(loc_max[0])
            if len(loc_max[0]) > 1:
                max_iou = iou_array[0]
            else:
                #print('actual one')
                max_iou = iou_array[int(loc_max[0])]
            
            val_dice_c.append(max_dice)
            val_loss.append(lossG.item())
            
            prec, rec, spec, accuracy, iou = prec_rec(max_pred_comb.detach().cpu().numpy(), Segmentation.detach().cpu().numpy())
            #print('all scores: ', prec, ' ', rec, ' ', spec, ' ', iou)
            f1_sc = (2*prec*rec) / (prec + rec)
            precision_comb.append(prec)
            recall_comb.append(rec)
            f1_comb.append(f1_sc)

        # ##########
        print('Epoch: ', i+1)
        print('Training Loss: ', np.mean(train_loss), 'Training Dice: ', np.mean(train_dice))
        print('Validation Loss: ', np.mean(val_loss), 'Validation Dice: ', np.mean(val_dice_c))
        print('Precision combined: ', np.mean(precision_comb))
        print('Recall combined: ', np.mean(recall_comb))
        print('F1 comb: ', np.mean(f1_comb))
        print('\n')
        
        file_name = 'experiment_modified_densenet_' + str(fold_number) + '_data.txt'
        with open(file_name, 'a') as f:
            f.write(f'Epoch: {i+1}')
            f.write('\n')
            f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)}')
            f.write('\n')
            f.write(f'Val Loss: {np.mean(val_loss)} Val Dice: {np.mean(val_dice_c)}')
            f.write('\n')
            f.write(f'Precision combined: {np.mean(precision_comb)}')
            f.write('\n')
            f.write(f'Recall combined: {np.mean(recall_comb)}')
            f.write('\n')
            f.write(f'F1 combined: {np.mean(f1_comb)}')
            f.write('\n')
            f.write('\n')
        
        #if i > 20:
        if np.mean(val_dice_c) > 0.70:
            #print('here', za)
            if np.mean(val_dice_c) > za:
                za = np.mean(val_dice_c)
                save_path = "Thyroid_Nodule/fold_saves_densenet/fold_"+str(fold_number)+"_modellll_"+str(np.mean(val_dice_c))+".pth"
                torch.save(net.state_dict(), save_path)
                print('save: ', save_path)

        record_train_loss.append(np.mean(train_loss))
        record_train_dice.append(np.mean(train_dice))
        record_val_loss.append(np.mean(val_loss))
        record_val_dice.append(np.mean(val_dice))
    
    record_train_loss = np.array(record_train_loss)
    record_train_dice = np.array(record_train_dice)
    record_val_loss = np.array(record_val_loss)
    record_val_dice = np.array(record_val_dice)

    

train_path = "Thyroid_Nodule/total_data.npy"
test_path = "Thyroid_Nodule/total_data.npy"

net = DAF_stack()
train_set = DataPrep(train_path, 0)
test_set = DataPrepval(test_path, 0)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 0)

net = DAF_stack()
train_set = DataPrep(train_path, 1)
test_set = DataPrepval(test_path, 1)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 1)

net = DAF_stack()
train_set = DataPrep(train_path, 2)
test_set = DataPrepval(test_path, 2)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 2)

net = DAF_stack()
train_set = DataPrep(train_path, 3)
test_set = DataPrepval(test_path, 3)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 3)

net = DAF_stack()
train_set = DataPrep(train_path, 4)
test_set = DataPrepval(test_path, 4)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 4)

net = DAF_stack()
train_set = DataPrep(train_path, 5)
test_set = DataPrepval(test_path, 5)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 5)

net = DAF_stack()
train_set = DataPrep(train_path, 6)
test_set = DataPrepval(test_path, 6)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 6)

net = DAF_stack()
train_set = DataPrep(train_path, 7)
test_set = DataPrepval(test_path, 7)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 7)

net = DAF_stack()
train_set = DataPrep(train_path, 8)
test_set = DataPrepval(test_path, 8)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 8)

net = DAF_stack()
train_set = DataPrep(train_path, 9)
test_set = DataPrepval(test_path, 9)
train_loader = DataLoader(train_set, batch_size=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=6, shuffle=True, pin_memory=True)
train_test(net, train_loader, test_loader, 9)

