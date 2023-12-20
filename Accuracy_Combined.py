
file1 = open('RF_Binary/RF_Binary_Accuracy_Scores.txt', 'r')
file2 = open('RF_Multiclass/RF_Multi_Accuracy_Scores.txt', 'r')
file3 = open('RF_Outliers/RF_Outlier_Accuracy_Scores.txt', 'r')

content1 = file1.read()
content2 = file2.read()
content3 = file3.read()

file1.close()
file2.close()
file3.close()

destination_file = open('Accuracy_Scores_Combined.txt', 'w')
destination_file.write(content1 + content2 + content3)
destination_file.close()