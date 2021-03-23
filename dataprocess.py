import codecs
import csv
import numpy as np

print('MCY1')
with open('koln-pruned.txt') as f:
    lines = f.readlines()
    list = []
    count = 0
    num = 0
    for line in lines:
        count += 1
        new = line.replace('#', '')
        new = new.replace('_', '')
        new = new.strip('\n')
        # new = new.strip("'")
        # print(new.split(' '))
        content = ['%.7f' % float(i) for i in new.split(' ')]
        print(content)
        list.append(content)
        print(count)
        if count % 5e6 == 0:
            num += 1
            with open('koln-pruned%f.csv' % num, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(list)
            list = []






        #a = line.replace('#', '1111')
'''
with codecs.open('koln-pruned.txt', 'r', encoding='utf-8') as f:
    text = f.read()
with codecs.open('koln-pruned.txt', 'w', encoding='utf8') as f:
    f.write(text.replace('#', '1111'))
    f.close()
'''

#contents = codecs.open('koln-pruned.txt', encoding='utf-8').write()

#contents.write(file.replace('#', '1111'))
#num_lines = sum(1 for line in contents)
##contents = codecs.open('2.txt', encoding='utf-8').read()
#print(num_lines)


#contents = open('koln-pruned.txt','w')
'''
newcontent = contents.replace('#', '1111')
newcontent = newcontent.replace('_', '2222')
newcontent.close()
'''
'''
with open("koln-pruned.txt", 'r+') as dataset:
    file = dataset.read()

with open('koln-pruned.txt','w') as contents:
    contents.write(file.replace('#', '1111'))
    contents.close()
'''