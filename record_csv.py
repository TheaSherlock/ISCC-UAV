import csv

def conv_csv(dir, data1, data2):
    # dir='/Users/thea/Library/CloudStorage/OneDrive-TheChineseUniversityofHongKong/Yao_20220803/Figure_Classifier/FLLS_original/'
    data=[data1, data2]
    name=['training_loss', 'test_accuracy']
    for data_idx in range(len(data)):
        with open (dir + name[data_idx] +'.csv', 'w', newline='') as csvfile:
            for content in data[data_idx]:
                csvfile.write(str(content)+'\n')

