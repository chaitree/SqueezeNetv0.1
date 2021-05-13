import cv2
import glob
import pandas as pd

print(cv2.__version__)

filenames_train = glob.glob("./cars_train/cars_train/*")
##print(filenames_train)

annots = pd.read_csv("./cars_train_annos.csv")

df_op = pd.DataFrame()
##print(annots.head())
image_no = 0 

for i in range(len(filenames_train)):
##    if (i>5):
##        break
    print(filenames_train[i])
    img = cv2.imread(filenames_train[i])
##    cv2.imshow("img", img)
##    cv2.waitKey(1000)
    image_name =filenames_train[i][-9:]
##    print(filenames_train[i][-9:])

    ht, wt, ch = img.shape
##    print(ht, wt, ch)

    for  j, row in annots.iterrows():
        if ( row['fname'][-9:] == image_name):
##            print (row)
            x1 = row['bbox_x1']
            x2 = row['bbox_x2']
            y1 = row['bbox_y1']
            y2 = row['bbox_y2']
            class_name = row['class']
            cropped_img = img.copy()
            
            cropped_img = cropped_img[y1:y2, x1:x2]
##            print(x1, x2, y1, y2)
##            cv2.rectangle(cropped_img, (x1,x2),(y1,y2), (0,0,255))
            cv2.imwrite("images/train/"+str(image_no)+".jpg", cropped_img)
##            cv2.imshow(str(image_no), cropped_img)
##            cv2.waitKey(1000)
            df_op1 = pd.DataFrame({"image_name":"images/train/"+str(image_no)+".jpg",'class_id':class_name}, index=pd.Index([i]))
            image_no = image_no +  1
            df_op = df_op.append(df_op1)
##print(df_op)
df_op.to_csv("./annotations/train_labels.csv")



filenames_test = glob.glob("./cars_test/cars_test/*")
##print(filenames_train)

annots = pd.read_csv("./cars_test_annos.csv")

df_op = pd.DataFrame()
##print(annots.head())
 

for i in range(len(filenames_test)):
##    if (i>5):
##        break
    print(filenames_test[i])
    img = cv2.imread(filenames_test[i])
    cv2.imshow("img", img)
##    cv2.waitKey(1000)
    image_name =filenames_test[i][-9:]
##    print(filenames_test[i][-9:])

    ht, wt, ch = img.shape
##    print(ht, wt, ch)

    for  j, row in annots.iterrows():
        if ( row['fname'][-9:] == image_name):
##            print (row)
            x1 = row['bbox_x1']
            x2 = row['bbox_x2']
            y1 = row['bbox_y1']
            y2 = row['bbox_y2']
            class_name = row['class']
            cropped_img = img.copy()
            
            cropped_img = cropped_img[y1:y2, x1:x2]
##            print(x1, x2, y1, y2)
##            cv2.rectangle(cropped_img, (x1,x2),(y1,y2), (0,0,255))
            cv2.imwrite("images/test/"+str(image_no)+".jpg", cropped_img)
##            cv2.imshow(str(image_no), cropped_img)
##            cv2.waitKey(1000)
            df_op1 = pd.DataFrame({"image_name":"images/test/"+str(image_no)+".jpg",'class_id':class_name}, index=pd.Index([i]))
            image_no = image_no +  1
            df_op = df_op.append(df_op1)
##print(df_op)
df_op.to_csv("./annotations/test_labels.csv")
