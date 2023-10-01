import os
import glob
import shutil

# dir_path = os.path.dirname(os.path.realpath(__file__))
#↑獲取當前資料夾名稱然後存成dir_path變數

# all_file_name = os.listdir(dir_path)
#↑讀取資料夾內所有檔案名稱然後放進all_file_name這個list裡

# file = glob.glob('.\discussion\*_json')
# for i in file:
#     print(i)

if __name__ == '__main__':
    # path = r'C:\Users\EBILxCPSL_S3\Desktop\Crack_detection\label\discussion'
    # finalpath = r'C:\Users\EBILxCPSL_S3\Desktop\Crack_detection\label\masks'
    path = '.\label_v2'
    finalpath = '.\\red_masks_v2'
    # file = os.listdir(path)
    n=1
    # for i in file:
    #     if os.path.isdir(os.path.join(path,i)):
    #         print(i)

    '''labelme_json_to_dataset轉完json檔後 從每個_json的資料夾拿出label.png'''
    for root, dirs, files in os.walk(path):
        # print(f"root is {root}")
        # print(f"directory is {dirs}")
        # print(f"files is {files}")
        if "_json" in root:
            if(files[1]=='label.png'):
                # print(f"path is {root}")
                name = root.split('\\')
                # print(f"name is {name}")
                # print(f"name is {name[2]}")
                name = name[2]
                name = name.split('_')
                name = name[0]+'_'+name[1]
                # print(f"name is {name}")
                # print(f"files is {files}")
                file = files[1]
                print("\nroot is "+root+"\\"+file)
                print("\t\t|")
                print("final path is "+finalpath+"\\"+name+'.jpg')
                os.rename(root+"\\"+file,finalpath+"\\"+name+'.jpg')

    #         # os.rename(files[1],f'label{n}.jpg')
    #         # shutil.copyfile(files[1], f'C:\\Users\\EBILxCPSL_S3\\Desktop\\Crack_detection\\label\masks\\lable{n}.jpg')
    #         # shutil.copyfile(files[1], f'C:\\Users\\EBILxCPSL_S3\\Desktop\\Crack_detection\\label\masks')
            # n += 1
            # print(n)


    # for root, dirs, files in allList:
    #     print("directory：", dirs)
    #     print("files: ", files)
        # print(files[3])
        # if(files[0]=='img.png'):
        #     print("path：", root)
        #     name = root.split('\\')
            # print(f"name is {name[9]}")
            # name = name[9]
            # name = name.split('_')
            # name = name[0]+'_'+name[1]
            # print(f"name is {name}")
            # print("file：", files[0])
            # file = files[0]
            # print("root is "+root+"\\"+file)
            # print("path is "+finalpath+"\\"+name+'.jpg')
            # os.rename(root+"\\"+file,finalpath+"\\"+name+'.jpg')

            # os.rename(files[1],f'label{n}.jpg')
            # shutil.copyfile(files[1], f'C:\\Users\\EBILxCPSL_S3\\Desktop\\Crack_detection\\label\masks\\lable{n}.jpg')
            # shutil.copyfile(files[1], f'C:\\Users\\EBILxCPSL_S3\\Desktop\\Crack_detection\\label\masks')
            # n += 1
            

