import ast

print('##### readline 함수 #####')
f = open('keypoints.txt', 'r', encoding='utf-8')
lines = f.readlines()
# s = ' '.join(s)
# s = ast.literal_eval(s)

cocoformat = [0,14,13,16,15,4,1,5,2,6,3,10,7,11,8,12,9] # 17 points
final = []
for line in lines:

    output = [0 for i in range(17*3)]
    # print(output)
    line = line.split('|')
    id = line[0]
    line = str(line[1])
    line = line.split(', [')
    print(id)
    for lin in line:
        if len(lin) > 80:
            lin = lin.split(',')
            # print(lin[1])
            
            lin_edit = ",".join(lin[0:5])
            # print(lin_edit+"]")
            lin = lin_edit+"]"
        lin = lin.split('{')
        # print(lin[1])
        lin = lin[1].split(" '")
        
        key_id = lin[0].split(":")[1]
        key_id = key_id.replace(",","")
        key_id = key_id.replace(" ","")
        # print(key_id)

        coordinate = lin[1].split(":")[1]
        coordinate = coordinate.split("(")[1]
        coordinate = coordinate.replace("[","")
        coordinate = coordinate.replace("]),","")
        coordinate = coordinate.replace(" ","")
        coordinate = coordinate.split(",")

        # print(coordinate)

        path = cocoformat.index(int(key_id))
        output[path*3+0] = int(coordinate[0])
        output[path*3+1] = int(coordinate[1])
        output[path*3+2] = 2
    
    result = '{"image_id":'+str(id)+',"category_id":1,"score":1.0,"keypoints":'+ str(output)+'}'
    result = result.replace("'","")
    final.append(result)
final = str(final).replace("'","")
fff = open('keypoints_test_fine.json', 'w', encoding='utf-8')
fff.write(final)
fff.close()
# print(final)
        # print("'" + lin[2])
    # line_dict = dict(line[1])
    # for lin in line
        
    # # s = s.replace("\\n'","")
    # print(line[1])
    # print("\n")
# f_as = open('segmentation.txt', 'w+t', encoding='utf-8')
# f_as.write(s)
# f_as.write("\n")
# f_as.close()
# f.close()
