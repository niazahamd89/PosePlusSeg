print('##### readline 함수 #####')
f = open('result2.txt', 'r')
s = f.readline()

for s in f:
    print(s.split(",")[1])
    # print('\n')
f.close()
